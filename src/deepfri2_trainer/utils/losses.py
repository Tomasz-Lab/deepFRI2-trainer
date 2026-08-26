"""Loss functions.

- ``WeightedFocalLoss`` -- focal loss with per-GO-term class weights; structure model.
- ``MCMLossDAG``        -- hierarchy-aware BCE that max-propagates probabilities along the
  direct GO edges; sequence and fusion models.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedFocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, device=None):
        super().__init__()
        self.gamma = gamma
        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        if alpha is not None:
            if isinstance(alpha, (float, int)):
                self.alpha = torch.tensor([alpha], device=self.device).float()
            elif isinstance(alpha, np.ndarray):
                self.alpha = torch.from_numpy(alpha).float().to(device=self.device)
            else:
                raise TypeError("Alpha must be None, a scalar, or a numpy array")

            # Reshape alpha to [C, 1] if it's a vector of class weights
            if len(self.alpha.shape) > 1:
                self.alpha = self.alpha.view(-1, 1)
        else:
            self.alpha = None

    def forward(self, inputs, targets, model=None):
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)

        if self.alpha is not None:
            focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        else:
            focal_loss = (1 - pt) ** self.gamma * ce_loss

        return focal_loss.mean()


class DAGPropagator:
    """Max-propagate probabilities up the GO DAG: ``parent >= max(children)``.

    The convention CAFA-evaluator uses before scoring, applied here to eval predictions so the
    in-loop Fmax matches the benchmark. Deliberately separate from ``MCMLossDAG``, which runs
    the same propagation inside the loss: keeping them apart means a change here cannot alter
    the training objective.
    """

    def __init__(self, A: torch.Tensor, num_steps: int | None = None):
        A = A.float().t().contiguous()          # input is CHILD -> PARENT
        self.parent_idx, self.child_idx = (A != 0).nonzero(as_tuple=True)
        if num_steps is None:
            num_steps = MCMLossDAG._estimate_dag_depth(self.parent_idx, self.child_idx, A.shape[0])
        self.num_steps = int(max(0, num_steps))

    def __call__(self, probabilities) -> np.ndarray:
        out = torch.as_tensor(np.asarray(probabilities), dtype=torch.float32)
        if self.parent_idx.numel() == 0 or self.num_steps == 0:
            return out.numpy()
        index = self.parent_idx.unsqueeze(0).expand(out.shape[0], -1)
        for _ in range(self.num_steps):
            nxt = out.clone()
            nxt.scatter_reduce_(1, index, out.index_select(1, self.child_idx),
                                reduce="amax", include_self=True)
            out = nxt
        return out.numpy()


class MCMLossDAG(nn.Module):
    """
    MCM loss using *direct* GO adjacency (parent -> child) instead of a transitive closure.

    Convention:
      A[p, c] = 1  iff  GO term p is a (direct) parent of GO term c.

    We enforce the hierarchy constraint via iterative max-propagation:
      out[p] := max(out[p], out[c])  for all edges (p -> c),
    repeated `num_steps` times (defaults to the longest path length in the DAG).

    This is typically much faster than using `adjacency_prop` (transitive closure),
    because the number of direct edges is far smaller than the closure edge count.
    """

    def __init__(
        self,
        A: torch.Tensor,
        num_steps: int | None = None,
        raw_violation_weight: float = 0.0,
        raw_violation_margin: float = 0.0,
    ):
        super().__init__()

        # Input adjacency is CHILD -> PARENT; convert to PARENT -> CHILD for propagation.
        A = A.float().t().contiguous()
        self.register_buffer("A", A)

        parent_idx, child_idx = (A != 0).nonzero(as_tuple=True)
        self.register_buffer("parent_idx", parent_idx.to(torch.long))
        self.register_buffer("child_idx", child_idx.to(torch.long))

        if num_steps is None:
            num_steps = self._estimate_dag_depth(parent_idx, child_idx, A.shape[0])
        self.num_steps = int(max(0, num_steps))
        self.raw_violation_weight = float(raw_violation_weight)
        self.raw_violation_margin = float(raw_violation_margin)

    @staticmethod
    def _estimate_dag_depth(parent_idx: torch.Tensor, child_idx: torch.Tensor, n: int) -> int:
        """
        Estimate longest path length (DAG depth) from direct edges using Kahn topo sort.
        Runs on CPU; n is small (~3k).
        """
        if parent_idx.numel() == 0:
            return 0

        # Build children adjacency lists + indegree
        children: list[list[int]] = [[] for _ in range(n)]
        indeg = [0] * n
        p_list = parent_idx.detach().cpu().tolist()
        c_list = child_idx.detach().cpu().tolist()
        for p, c in zip(p_list, c_list):
            children[p].append(c)
            indeg[c] += 1

        # Kahn topo order
        from collections import deque

        q = deque([i for i in range(n) if indeg[i] == 0])
        topo: list[int] = []
        while q:
            u = q.popleft()
            topo.append(u)
            for v in children[u]:
                indeg[v] -= 1
                if indeg[v] == 0:
                    q.append(v)

        # Longest path DP along topo
        depth = [0] * n
        for u in topo:
            du = depth[u]
            for v in children[u]:
                nv = du + 1
                if nv > depth[v]:
                    depth[v] = nv
        return max(depth)

    def forward(self, logits, targets, model=None):
        outputs = torch.sigmoid(logits)
        targets = targets.to(dtype=outputs.dtype)

        constr_output = self.get_constr_out(outputs)

        train_output = self.get_constr_out(targets * outputs)
        train_output = (1 - targets) * constr_output + targets * train_output

        base = F.binary_cross_entropy(train_output.float(), targets.float())

        # Optional explicit penalty on RAW hierarchy violations (before propagation):
        # enforce outputs[parent] >= outputs[child] on direct edges.
        if self.raw_violation_weight > 0 and self.parent_idx.numel() > 0:
            penalty = self.raw_violation_penalty(outputs)
            return base + self.raw_violation_weight * penalty

        return base

    def raw_violation_penalty(self, outputs: torch.Tensor) -> torch.Tensor:
        """
        Mean ReLU(child - parent - margin) over all direct edges and batch items.
        Computed chunked to avoid materializing (B, E) for large E.
        """
        if self.parent_idx.numel() == 0:
            return outputs.new_tensor(0.0)

        out = outputs.float()
        B, _ = out.shape
        E = int(self.parent_idx.numel())
        m = float(self.raw_violation_margin)

        CHUNK = 200_000
        total = out.new_tensor(0.0)
        count = 0
        for start in range(0, E, CHUNK):
            end = min(start + CHUNK, E)
            p = self.parent_idx[start:end]
            c = self.child_idx[start:end]
            # (B, chunk): positive when child > parent (+margin)
            v = (out.index_select(1, c) - out.index_select(1, p) - m).relu()
            total = total + v.sum()
            count += v.numel()

        return total / max(1, count)

    def get_constr_out(self, outputs: torch.Tensor) -> torch.Tensor:
        """
        Apply iterative max-propagation using direct edges.
        """
        if self.parent_idx.numel() == 0 or self.num_steps == 0:
            return outputs

        # compute in float32 for stability
        out = outputs.float()
        B, _ = out.shape
        E = self.parent_idx.numel()

        # Chunked gather + scatter max
        # IMPORTANT: avoid in-place updates on the same Tensor across multiple scatter_reduce_
        # calls, otherwise autograd can error with "modified by an inplace operation".
        CHUNK = 200_000
        for _ in range(self.num_steps):
            for start in range(0, E, CHUNK):
                end = min(start + CHUNK, E)
                p = self.parent_idx[start:end]
                c = self.child_idx[start:end]

                src = out.index_select(1, c)  # (B, chunk)
                idx = p.unsqueeze(0).expand(B, end - start)

                out_next = out.clone()
                out_next.scatter_reduce_(1, idx, src, reduce="amax", include_self=True)
                out = out_next

        return out.to(dtype=outputs.dtype)
