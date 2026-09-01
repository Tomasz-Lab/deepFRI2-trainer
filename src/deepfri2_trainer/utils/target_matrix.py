"""Target matrices: protein -> GO-term supervision, built from the primitive annotation tables.

:class:`TargetMatrix` builds the training supervision -- which GO terms survive the annotation
threshold, the sparse per-protein label vectors, the class weights and the GO adjacency the
hierarchy loss propagates over. :class:`InferenceTargetMatrix` builds label vectors for an
arbitrary annotation table (the CAZy test set) against a *fixed* GO-term order, so a trained
model can be scored on it.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import torch
from networkx.readwrite import json_graph

#: The three GO namespace roots. A root term annotates every protein, so it carries no signal;
#: ``exclude_roots`` drops it from the label space.
ROOTS = {"MF": "GO:0003674", "CC": "GO:0005575", "BP": "GO:0008150"}

#: Number of annotations at which a GO term's loss weight reaches its floor.
WEIGHT_REFERENCE_COUNT = 5000
WEIGHT_FLOOR = 0.1

#: Names of the eight pickles :meth:`TargetMatrix.create_targets` returns, in order.
TARGET_FILES = (
    "go_indices.pkl",
    "protein_vectors.pkl",
    "protein_vectors_test.pkl",
    "weights.pkl",
    "adjacency.pkl",
    "adjacency_prop.pkl",
    "grand_truth.pkl",
    "grand_truth_test.pkl",
)


def load_go_graphs(graphs_dir: Path | str, ontologies: list[str]):
    """``graph_<ontology>.json`` -> ``(graphs, node sets)``.

    Edge direction is CHILD -> PARENT, as stored (the root has in-edges and no out-edges).
    """
    graphs_dir = Path(graphs_dir)
    graphs, nodes = {}, {}
    for ontology in ontologies:
        with open(graphs_dir / f"graph_{ontology}.json") as handle:
            graphs[ontology] = json_graph.node_link_graph(json.load(handle), edges="edges")
        nodes[ontology] = set(graphs[ontology].nodes())
    return graphs, nodes


class BaseTargetMatrix:
    """Sparse label-vector construction, shared by the training and inference builders."""

    @staticmethod
    def _make_sparse_vector(indices, size):
        if indices:
            index_tensor = torch.tensor(indices, dtype=torch.long).unsqueeze(0)
            value_tensor = torch.ones(len(indices), dtype=torch.float32)
        else:
            index_tensor = torch.empty((1, 0), dtype=torch.long)
            value_tensor = torch.empty((0,), dtype=torch.float32)
        return torch.sparse_coo_tensor(
            indices=index_tensor, values=value_tensor, size=(size,), dtype=torch.float32
        ).coalesce()

    @classmethod
    def _build_protein_vectors(
        cls,
        annotations: pd.DataFrame,
        go_indices: dict,
        protein_col: str = "DB_Object_ID",
        go_col: str = "GO_ID_all",
    ):
        """One sparse multi-hot vector per protein over ``go_indices``."""
        if annotations.empty:
            return {}

        valid = set(go_indices)
        annots = annotations[[protein_col, go_col]].dropna().drop_duplicates()
        annots = annots[
            (annots[protein_col].astype(str).str.len() > 0)
            & (annots[go_col].astype(str).str.len() > 0)
        ]
        return {
            protein_id: cls._make_sparse_vector(
                sorted(go_indices[go_id] for go_id in set(group[go_col]) & valid), len(go_indices)
            )
            for protein_id, group in annots.groupby(protein_col)
        }

    @staticmethod
    def create_annotation_dataframe(protein_vectors, go_indices):
        """Sparse vectors -> a long ``(ProteinID, GO_Term)`` table."""
        go_of_position = {position: go for go, position in go_indices.items()}
        rows = [
            (protein_id, go_of_position[position])
            for protein_id, vector in protein_vectors.items()
            for position in vector.indices()[0].tolist()
            if position in go_of_position
        ]
        return pd.DataFrame(rows, columns=["ProteinID", "GO_Term"])

    @staticmethod
    def _report(ontology: str, title: str, lines: dict[str, int]) -> None:
        """A labelled statistics block -- which ontology, and which artefact it describes."""
        print(f"[{ontology}] {title}")
        for label, value in lines.items():
            print(f"[{ontology}]   {label:<33} {value:>12,}")


class TargetMatrix(BaseTargetMatrix):
    """Training supervision for one or more ontologies.

    Three annotation views are derived from the same primitives, differing only in which
    annotations they admit:

    ``train``     the configured ``qualities`` -- what the model is trained on.
    ``truth``     high-quality only -- the ground truth the CAFA evaluation scores against.
    ``truth_test``high-quality only, from the held-out test proteins.
    """

    def __init__(
        self,
        annotations_dir: Path | str,
        unified_file: Path | str,
        graphs_dir: Path | str,
        ontologies: list[str],
        qualities: list[str],
        threshold: int,
        exclude_roots: bool = False,
        reuse_go_indices: dict[str, dict] | None = None,
    ):
        self.annotations_dir = Path(annotations_dir)
        self.unified_file = Path(unified_file)
        self.graphs_dir = Path(graphs_dir)
        self.ontologies = list(ontologies)
        self.qualities = list(qualities)
        self.threshold = int(threshold)
        self.exclude_roots = exclude_roots
        self.reuse_go_indices = reuse_go_indices or {}

    # ---------- primitives ----------

    UNIFIED_COLUMNS = [
        "Unnamed: 0", "DB_Object_ID", "GO_ID", "division_group", "annotation_quality",
        "Relation", "globalMetricValue", "Testing", "Sequence",
    ]

    def _load_unified(self):
        """Annotation metadata, split into the training and the test half.

        ``NOT`` relations are negative examples and are excluded from both. Training additionally
        requires a structure quality score (``globalMetricValue``).
        """
        print("Loading annotations...")
        unified = pd.read_csv(self.unified_file, index_col=0, usecols=self.UNIFIED_COLUMNS)
        not_relation = unified["Relation"].str.contains("NOT") == True  # noqa: E712
        test = unified[~not_relation & (unified["Testing"] == True)]  # noqa: E712
        train = unified[
            ~not_relation & (unified["Testing"] == False) & (~unified["globalMetricValue"].isna())  # noqa: E712
        ]
        return train, test

    def _load_annotation_views(self, unified_train, unified_test):
        """The three quality-filtered annotation views, per ontology."""
        print("Filtering annotations...")
        views = {"train": {}, "truth": {}, "truth_test": {}}
        selectors = {
            "train": unified_train[unified_train["annotation_quality"].isin(self.qualities)],
            "truth": unified_train[unified_train["annotation_quality"].isin(["HQ"])],
            "truth_test": unified_test[unified_test["annotation_quality"].isin(["HQ"])],
        }
        for ontology in self.ontologies:
            with open(self.annotations_dir / f"annots_{ontology}.pickle", "rb") as handle:
                annots = pickle.load(handle)
            for view, selector in selectors.items():
                subset = annots[annots.index.isin(selector.index)].copy()
                # DB_Object_ID_unique is "<division>__<accession>"; collapse to the accession and
                # deduplicate, so a protein annotated in several divisions counts once
                subset["DB_Object_ID"] = subset["DB_Object_ID_unique"].str.split("__").str[1]
                views[view][ontology] = subset.drop_duplicates(["DB_Object_ID", "GO_ID_all"])
        return views

    # ---------- graph ----------

    def _threshold_graph(self, graph, train_annots):
        """The subgraph of GO terms with at least ``threshold`` annotated proteins.

        Returns ``(subgraph, node order)``. The order is the parent graph's, taken explicitly:
        iterating a networkx subgraph *view* is not order-stable across processes -- when the
        kept set is small relative to the graph, ``FilterAtlas.__iter__`` iterates it as a
        Python set, whose order depends on ``PYTHONHASHSEED``. That order becomes ``go_indices``
        and thus the model's output dimension order, so leaving it to chance made every build of
        the target matrix produce a differently permuted label space.
        """
        annotated = train_annots.groupby("GO_ID_all")["DB_Object_ID"].agg(set).apply(len)
        counts = pd.Series(0, index=graph.nodes)
        counts.update(annotated)  # terms with no annotation stay at 0
        for node in graph.nodes:
            graph.nodes[node]["prots"] = int(counts.loc[node])
        kept = [node for node in graph.nodes if graph.nodes[node].get("prots", 0) >= self.threshold]

        subgraph = graph.subgraph(kept).copy()
        assert len(list(nx.weakly_connected_components(subgraph))) == 1
        assert nx.is_directed_acyclic_graph(subgraph)
        return subgraph, kept

    @staticmethod
    def _adjacencies(subgraph, node_order):
        """``(adjacency, propagated adjacency)`` in the graph's own CHILD -> PARENT direction.

        Both are laid out in ``node_order``, the order ``go_indices`` uses.

        The propagated matrix is the transitive closure plus self-loops, so
        ``adjacency_prop[child, ancestor] == 1``. A hierarchy loss that wants PARENT -> CHILD
        must transpose.
        """
        adjacency = torch.tensor(nx.adjacency_matrix(subgraph, nodelist=node_order).todense())
        closure = nx.algorithms.dag.transitive_closure_dag(subgraph)
        propagated = torch.tensor(
            nx.adjacency_matrix(closure, nodelist=node_order).todense()
        )
        propagated = (propagated != 0).to(dtype=adjacency.dtype)
        propagated.fill_diagonal_(1)
        return adjacency, propagated

    def _weights(self, train_annots, go_indices):
        """Per-term loss weight: rare terms weigh 1, terms with >= 5000 annotations weigh 0.1."""
        counts = (
            train_annots.groupby("GO_ID_all")["DB_Object_ID"].agg("count")
            .loc[list(go_indices)]
            .values
        )
        return np.maximum(1 - np.minimum(counts / WEIGHT_REFERENCE_COUNT, 1), WEIGHT_FLOOR)

    def _adopt_order(self, ontology: str, node_order: list[str]) -> list[str]:
        """Keep an existing ``go_indices`` ordering, when one was supplied.

        A checkpoint's output dimensions are tied to the ordering it was trained with, so
        rebuilding a target matrix for an existing model must reuse that model's order rather
        than the freshly derived one. The term *set* must match exactly -- a different threshold
        or annotation version is a different label space, not a reordering of one.
        """
        existing = (self.reuse_go_indices or {}).get(ontology)
        if not existing:
            return node_order
        if set(existing) != set(node_order):
            only_existing = sorted(set(existing) - set(node_order))[:5]
            only_new = sorted(set(node_order) - set(existing))[:5]
            raise ValueError(
                f"cannot reuse the {ontology} go_indices ordering: the term sets differ "
                f"({len(existing)} vs {len(node_order)} terms; only-existing {only_existing}, "
                f"only-new {only_new}). Rebuild with the threshold and annotation version the "
                "existing label space was built from."
            )
        print(f"[{ontology}]   reusing the supplied go_indices ordering ({len(existing)} terms)")
        return [go_id for go_id, _ in sorted(existing.items(), key=lambda item: item[1])]

    # ---------- root removal ----------

    @classmethod
    def _drop_index(cls, vectors, root_index, old_to_new, size):
        """Rebuild sparse vectors without the root term; returns ``(vectors, emptied proteins)``."""
        rebuilt, emptied = {}, []
        for protein_id, tensor in vectors.items():
            if tensor._indices().numel() == 0:
                continue
            old = tensor._indices().squeeze()
            keep = old != root_index
            kept = old[keep]
            if kept.numel() == 0:
                emptied.append(protein_id)
                continue
            rebuilt[protein_id] = torch.sparse_coo_tensor(
                indices=torch.tensor([old_to_new[int(i)] for i in kept], dtype=torch.long).unsqueeze(0),
                values=tensor._values()[keep],
                size=(size,),
                dtype=torch.float32,
            ).coalesce()
        return rebuilt, emptied

    @staticmethod
    def _drop_row_col(matrix, index):
        keep = torch.arange(matrix.shape[0]) != index
        return matrix[keep, :][:, keep]

    def _exclude_root(self, ontology, go_indices, protein_vectors, protein_vectors_test,
                      weights, adjacency, adjacency_prop, truth, truth_test):
        root = ROOTS.get(ontology)
        if root not in go_indices[ontology]:
            return
        print(f"Excluding root {root} from '{ontology}'...")

        old = go_indices[ontology].copy()
        root_index = old[root]
        kept_go_ids = [go_id for go_id in old if go_id != root]
        go_indices[ontology] = {go_id: i for i, go_id in enumerate(kept_go_ids)}
        old_to_new = {old[go_id]: go_indices[ontology][go_id] for go_id in kept_go_ids}

        adjacency[ontology] = self._drop_row_col(adjacency[ontology], root_index)
        adjacency_prop[ontology] = self._drop_row_col(adjacency_prop[ontology], root_index)
        weights[ontology] = np.delete(weights[ontology], root_index)

        for vectors, annots in ((protein_vectors, truth), (protein_vectors_test, truth_test)):
            vectors[ontology], emptied = self._drop_index(
                vectors[ontology], root_index, old_to_new, len(kept_go_ids)
            )
            annots[ontology] = annots[ontology][
                (~annots[ontology]["DB_Object_ID"].isin(emptied))
                & (annots[ontology]["GO_ID_all"] != root)
            ]

    # ---------- entry point ----------

    def create_targets(self):
        """Build every target-matrix artefact.

        Returns ``(go_indices, protein_vectors, protein_vectors_test, weights, adjacency,
        adjacency_prop, grand_truth, grand_truth_test)``, each a dict keyed by ontology --
        the eight pickles named by :data:`TARGET_FILES`, in that order.
        """
        unified_train, unified_test = self._load_unified()
        views = self._load_annotation_views(unified_train, unified_test)
        train, truth, truth_test = views["train"], views["truth"], views["truth_test"]

        print("Loading GO-graphs...")
        graphs, _ = load_go_graphs(self.graphs_dir, self.ontologies)

        go_indices, protein_vectors, protein_vectors_test = {}, {}, {}
        weights, adjacency, adjacency_prop, subgraphs = {}, {}, {}, {}
        for ontology in self.ontologies:
            print(f"[{ontology}] thresholding the GO graph and building label vectors...")
            subgraphs[ontology], node_order = self._threshold_graph(graphs[ontology], train[ontology])
            node_order = self._adopt_order(ontology, node_order)
            adjacency[ontology], adjacency_prop[ontology] = self._adjacencies(
                subgraphs[ontology], node_order)

            go_indices[ontology] = {go_id: i for i, go_id in enumerate(node_order)}
            protein_vectors[ontology] = self._build_protein_vectors(train[ontology], go_indices[ontology])
            protein_vectors_test[ontology] = self._build_protein_vectors(
                truth_test[ontology], go_indices[ontology]
            )
            weights[ontology] = self._weights(train[ontology], go_indices[ontology])

            # the ground truth spans the FULL graph, not the thresholded subgraph: the CAFA
            # evaluation must not be told to ignore terms the model was never given
            for annots in (truth, truth_test):
                annots[ontology] = annots[ontology][
                    annots[ontology]["GO_ID_all"].isin(graphs[ontology].nodes)
                ]

        if self.exclude_roots:
            for ontology in self.ontologies:
                self._exclude_root(ontology, go_indices, protein_vectors, protein_vectors_test,
                                   weights, adjacency, adjacency_prop, truth, truth_test)

        for ontology in self.ontologies:
            self._report(ontology, "target matrix", {
                "GO terms (nodes)": len(go_indices[ontology]),
                "GO links": len(subgraphs[ontology].edges),
                "GO links (propagated)": int(adjacency_prop[ontology].count_nonzero().item()),
                "proteins (train)": len(protein_vectors[ontology]),
                "proteins (test)": len(protein_vectors_test[ontology]),
                "annotations (train)": int(sum(
                    t.values().sum().item() for t in protein_vectors[ontology].values())),
                "annotations (test)": int(sum(
                    t.values().sum().item() for t in protein_vectors_test[ontology].values())),
            })

        # test-set sequences travel with the ground truth: the test FASTA is written from it
        sequences = unified_test[["DB_Object_ID", "Sequence"]].drop_duplicates().set_index("DB_Object_ID")
        for ontology in self.ontologies:
            truth_test[ontology] = truth_test[ontology].join(sequences, on="DB_Object_ID", how="left")

        return (go_indices, protein_vectors, protein_vectors_test, weights,
                adjacency, adjacency_prop, truth, truth_test)


class InferenceTargetMatrix(BaseTargetMatrix):
    """Label vectors for an arbitrary annotation table, against a *fixed* GO-term order.

    Used for the CAZy test set: the model's ``go_indices`` are taken as given, so the vectors
    line up with what the model predicts. Unlike :class:`TargetMatrix` nothing is thresholded or
    quality-filtered; with ``propagate_terms`` the annotations are first propagated to all their
    ancestors in the full GO graph.
    """

    def __init__(
        self,
        annotations: pd.DataFrame | Path | str,
        go_indices: dict,
        go_graph_nodes: dict | None = None,
        go_graphs: dict | None = None,
        protein_column: str = "protein_id",
        go_column: str = "go_id",
        ontology_column: str | None = None,
        csv_kwargs: dict | None = None,
        propagate_terms: bool = False,
    ):
        self.annotations = annotations if isinstance(annotations, pd.DataFrame) else Path(annotations)
        self.go_indices = go_indices
        self.go_graph_nodes = go_graph_nodes or {o: set(idx) for o, idx in go_indices.items()}
        self.go_graphs = go_graphs or {}
        self.protein_column = protein_column
        self.go_column = go_column
        self.ontology_column = ontology_column
        self.csv_kwargs = csv_kwargs or {}
        self.propagate_terms = propagate_terms
        self.ontologies = list(go_indices)

    def _ontology_of_go_term(self):
        ontology_of, duplicated = {}, set()
        for ontology, go_ids in self.go_graph_nodes.items():
            for go_id in go_ids:
                if ontology_of.setdefault(go_id, ontology) != ontology:
                    duplicated.add(go_id)
        if duplicated:
            raise ValueError(f"GO terms in multiple ontologies: {sorted(duplicated)[:10]}")
        return ontology_of

    def _load(self):
        """Read the table, explode list-valued GO columns to one row per (protein, term)."""
        annots = (self.annotations.copy() if isinstance(self.annotations, pd.DataFrame)
                  else pd.read_csv(self.annotations, **self.csv_kwargs))

        required = {self.protein_column, self.go_column} | (
            {self.ontology_column} if self.ontology_column else set())
        missing = required - set(annots.columns)
        if missing:
            raise ValueError(f"Missing required columns in {self.annotations}: {sorted(missing)}")

        annots = annots.dropna(subset=[self.protein_column, self.go_column])
        annots[self.protein_column] = annots[self.protein_column].astype(str).str.strip()
        annots = annots[annots[self.protein_column] != ""]

        if not annots.empty:
            collection = (list, tuple, set, np.ndarray, pd.Series)
            annots[self.go_column] = annots[self.go_column].apply(
                lambda value: list(value) if isinstance(value, collection) else [value])
            annots = annots.explode(self.go_column, ignore_index=True)
            annots = annots.dropna(subset=[self.go_column])
            annots[self.go_column] = annots[self.go_column].astype(str).str.strip()
            annots = annots[annots[self.go_column] != ""].copy()

        annots["_is_original_go_term"] = True
        annots["_ontology"] = (
            annots[self.go_column].map(self._ontology_of_go_term()) if self.ontology_column is None
            else annots[self.ontology_column].astype(str).str.strip()
        )
        return annots

    def _propagate(self, annots: pd.DataFrame) -> pd.DataFrame:
        """Add every ancestor of every annotated term, flagging which were original."""
        if not self.propagate_terms or annots.empty:
            return annots

        frames = []
        for ontology in self.ontologies:
            subset = annots[annots["_ontology"] == ontology]
            if subset.empty:
                continue
            if ontology not in self.go_graphs:
                raise ValueError(f"GO graph for '{ontology}' is required when propagate_terms=True.")
            graph = self.go_graphs[ontology]
            rows = []
            for protein_id, go_ids in subset.groupby(self.protein_column)[self.go_column].agg(set).items():
                propagated = set()
                for go_id in go_ids:
                    if go_id in graph:
                        propagated |= {go_id} | nx.descendants(graph, go_id)
                rows += [{self.protein_column: protein_id, self.go_column: go_id,
                          "_ontology": ontology, "_is_original_go_term": go_id in go_ids}
                         for go_id in sorted(propagated)]
            frames.append(pd.DataFrame(rows))

        frames = [frame for frame in frames if not frame.empty]
        return pd.concat(frames, ignore_index=True) if frames else annots.iloc[0:0].copy()

    def create_targets(self):
        """Returns ``(go_indices, protein_vectors, grand_truth)``, each keyed by ontology."""
        print(f"[{self.ontologies[0] if len(self.ontologies) == 1 else ','.join(self.ontologies)}]"
              " loading CAZy protein:GO annotations...")
        annots = self._propagate(self._load())

        unknown = int(annots["_ontology"].isna().sum())
        if unknown:
            print(f"Dropping {unknown} annotations with GO terms missing from the GO graph.")
        stray = sorted(set(annots["_ontology"].dropna()) - set(self.ontologies))
        if stray:
            raise ValueError(f"Found ontologies not present in go_indices: {stray}")

        protein_vectors, grand_truth = {}, {}
        for ontology in self.ontologies:
            subset = annots[annots["_ontology"] == ontology].drop_duplicates(
                [self.protein_column, self.go_column])

            in_graph = subset[self.go_column].isin(self.go_graph_nodes[ontology])
            if (~in_graph).sum():
                print(f"Dropping {int((~in_graph).sum())} annotations for '{ontology}' that are "
                      "not present in the original GO graph.")
            subset = subset[in_graph].copy()

            # the ground truth keeps every term of the full graph; only the label vectors are
            # restricted to the terms the model actually has an output for
            in_indices = subset[self.go_column].isin(set(self.go_indices[ontology]))
            subset["is_in_go_indices"] = in_indices
            grand_truth[ontology] = subset.rename(
                columns={"_is_original_go_term": "is_original_go_term"}).drop(columns="_ontology")
            protein_vectors[ontology] = self._build_protein_vectors(
                subset[in_indices], self.go_indices[ontology],
                protein_col=self.protein_column, go_col=self.go_column)

            self._report(ontology, "CAZy target matrix", {
                "GO terms (nodes)": len(self.go_indices[ontology]),
                "proteins": len(protein_vectors[ontology]),
                "annotations": int(sum(
                    t.values().sum().item() for t in protein_vectors[ontology].values())),
                "ground-truth annotations": len(grand_truth[ontology]),
            })

        return self.go_indices, protein_vectors, grand_truth
