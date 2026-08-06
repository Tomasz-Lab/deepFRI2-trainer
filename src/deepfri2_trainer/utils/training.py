"""The deepFRI2 training loop.

Optimizer, loss selection, AMP handling, gradient clipping and metrics are unchanged from the
code that produced the released checkpoints. ``wandb.init`` lives in
``deepfri2_trainer.outputs.start_run`` -- run outputs are named after the wandb run, so the
session must be open before training begins; per-epoch metric logging still happens here.
"""

import numpy as np
import torch
import torch.optim as optim
import tqdm
import wandb
from scipy.special import expit as sigmoid

from .losses import MCMLossDAG, WeightedFocalLoss


def count_trainable_parameters(model):
    count = 0
    for i in model.named_parameters():
        if i[1].requires_grad:
            count += i[1].numel()
    return count


def initialize_training(
    model,
    learning_rate,
    weights=None,
    loss_fn_name=None,
    loss_fn_kwargs=None,
):
    """Initialize the optimizer and the loss function."""
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    if loss_fn_name == "MCMLossDAG":
        assert isinstance(loss_fn_kwargs, dict)
        loss_fn = MCMLossDAG(
            A=loss_fn_kwargs["A"],
            num_steps=loss_fn_kwargs.get("num_steps"),
            raw_violation_weight=loss_fn_kwargs.get("raw_violation_weight", 0.0),
            raw_violation_margin=loss_fn_kwargs.get("raw_violation_margin", 0.0),
        )
    elif loss_fn_name is None:
        loss_fn = WeightedFocalLoss(alpha=weights)
    else:
        raise ValueError(
            f"unknown loss_fn_name {loss_fn_name!r}; expected 'MCMLossDAG' or None "
            "(None selects WeightedFocalLoss)"
        )

    return optimizer, loss_fn


def process_batch(batch, device, use_embeddings, use_distograms):
    """Move a batch to ``device``, dropping the modalities the model does not consume."""
    _, embeds, disto, targets, masks = batch

    if use_embeddings:
        embeds = embeds.to(device=device)
    else:
        embeds = None

    if use_distograms:
        disto = disto.to(device=device)
    else:
        disto = None

    targets = targets.to(device=device)
    masks = masks.to(device=device)

    return embeds, disto, targets, masks


def train_epoch(
    model,
    train_dataloader,
    optimizer,
    loss_fn,
    device,
    use_embeddings,
    use_distograms,
    pbar: tqdm.tqdm = None,
    max_steps_per_epoch: int | None = None,
    grad_clip_max_norm: float | None = 1.0,
):
    """Run a single training epoch."""
    model.train()
    total_loss = 0
    num_batches = 0
    all_predictions = []
    all_targets = []

    scaler = torch.amp.GradScaler()

    for idx, batch in enumerate(train_dataloader):
        embeds, disto, targets, masks = process_batch(
            batch, device, use_embeddings, use_distograms
        )

        optimizer.zero_grad()

        with torch.amp.autocast(torch.device(device).type):
            logits = model(embeds, disto, masks)

        loss = loss_fn(logits, targets, model)

        scaler.scale(loss).backward()
        if grad_clip_max_norm is not None and grad_clip_max_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_max_norm)
        scaler.step(optimizer)
        scaler.update()

        # epoch-level metrics are computed over all batches
        all_predictions.append(logits.float().cpu().detach().numpy())
        all_targets.append(targets.cpu().detach().numpy())

        total_loss += loss.item()
        num_batches += 1

        if pbar is not None:
            pbar.set_postfix(
                loss=loss.item(), progress=f"{idx}/{max_steps_per_epoch or len(train_dataloader)}"
            )

        if max_steps_per_epoch is not None and (idx + 1) >= max_steps_per_epoch:
            break

    all_predictions = np.concatenate(all_predictions)
    all_targets = np.concatenate(all_targets)
    train_loss = total_loss / num_batches

    return train_loss, all_predictions, all_targets


def evaluate_model(
    model,
    eval_dataloader,
    loss_fn,
    device,
    use_embeddings,
    use_distograms,
    max_steps_per_epoch: int | None = None,
):
    """Evaluate the model on the evaluation dataset."""
    model.eval()
    eval_predictions = []
    eval_targets = []
    eval_loss = 0
    eval_batches = 0

    with torch.no_grad():
        for idx, batch in enumerate(tqdm.tqdm(eval_dataloader)):
            embeds, disto, targets, masks = process_batch(
                batch, device, use_embeddings, use_distograms
            )

            logits = model(embeds, disto, masks)
            loss = loss_fn(logits, targets, model)

            eval_predictions.append(logits.float().cpu().detach().numpy())
            eval_targets.append(targets.cpu().numpy())
            eval_loss += loss.item()
            eval_batches += 1

            if max_steps_per_epoch is not None and (idx + 1) >= max_steps_per_epoch:
                break

    eval_predictions = np.concatenate(eval_predictions)
    eval_targets = np.concatenate(eval_targets)
    eval_loss = eval_loss / eval_batches

    return eval_loss, eval_predictions, eval_targets


def calculate_metrics(predictions, targets, threshold=0.3):
    """Macro-averaged precision / recall / F1 over GO terms, plus micro averages.

    Returns ``(macro_precision, macro_recall, macro_f1, extras)``.

    Macro averaging over a very sparse label space needs care about undefined classes, and
    this is where the previous implementation was wrong. Conventions (sklearn's, with
    ``zero_division=np.nan``):

    - precision is undefined for a GO term the model never predicts (tp + fp == 0);
      such terms are excluded from the macro precision.
    - recall is undefined for a GO term with no ground truth (tp + fn == 0); such terms are
      excluded from the macro recall.
    - **F1 is undefined only for terms that are absent from both** -- no ground truth and no
      prediction. A term with ground truth that was never predicted scores F1 = 0 and *must*
      be counted. F1 is therefore computed as ``2 * tp / (predicted + support)``, which is
      the harmonic mean wherever both are defined and gives the right answer in the
      degenerate cases (this is also how sklearn computes it).

    The previous version computed ``f1 = 2pr/(p+r)`` elementwise, so a never-predicted term
    inherited precision's NaN and was dropped by ``nanmean``. With a few hundred predicted
    terms out of a few thousand, that averaged F1 over only the terms the model happened to
    fire on and inflated it several-fold (e.g. macro F1 0.17 where sklearn reports 0.005,
    alongside macro recall 0.006). Reported F1 values from before this fix are not comparable
    with the ones after it.

    The three macro numbers are still averaged over different sets of terms, so F1 is not
    ``2PR/(P+R)`` of the reported P and R -- that is inherent to nan-skipping macro
    averaging. ``extras`` therefore also carries the micro averages (which are mutually
    consistent) and the class counts each macro average is over.

    Vectorized numpy, ~17x faster than sklearn at this problem size (~110K proteins x 5467 GO
    terms) and numerically identical to it (asserted in tests/test_metrics.py).
    """
    preds_bin = (sigmoid(predictions) >= threshold).astype(np.float64)
    t = targets.astype(np.float64)
    # float64 accumulation: float32 sums over ~10^5 proteins lose enough precision to show up
    # against sklearn.
    tp = (preds_bin * t).sum(axis=0)
    fp = (preds_bin * (1 - t)).sum(axis=0)
    fn = ((1 - preds_bin) * t).sum(axis=0)

    predicted = tp + fp   # terms the model fired on -> precision defined
    support = tp + fn     # terms with ground truth  -> recall defined

    with np.errstate(invalid="ignore", divide="ignore"):
        precision = np.where(predicted > 0, tp / np.where(predicted > 0, predicted, 1), np.nan)
        recall = np.where(support > 0, tp / np.where(support > 0, support, 1), np.nan)
        f1_denominator = predicted + support
        f1 = np.where(
            f1_denominator > 0, 2 * tp / np.where(f1_denominator > 0, f1_denominator, 1), np.nan
        )

        total_tp, total_fp, total_fn = float(tp.sum()), float(fp.sum()), float(fn.sum())
        micro_precision = total_tp / (total_tp + total_fp) if total_tp + total_fp > 0 else np.nan
        micro_recall = total_tp / (total_tp + total_fn) if total_tp + total_fn > 0 else np.nan
        micro_denominator = micro_precision + micro_recall
        micro_f1 = (
            2 * micro_precision * micro_recall / micro_denominator
            if np.isfinite(micro_denominator) and micro_denominator > 0
            else np.nan
        )

    extras = {
        "precision_micro": float(micro_precision),
        "recall_micro": float(micro_recall),
        "f1_micro": float(micro_f1),
        "classes_predicted": int((predicted > 0).sum()),
        "classes_with_support": int((support > 0).sum()),
        "classes_total": int(tp.shape[0]),
    }
    return (
        float(np.nanmean(precision)),
        float(np.nanmean(recall)),
        float(np.nanmean(f1)),
        extras,
    )


def log_metrics(
    train_loss,
    eval_loss,
    all_predictions,
    all_targets,
    eval_predictions,
    eval_targets,
    prfs_train,
    prfs_eval,
    epoch,
    log_wandb=True,
):
    """Log metrics to console and wandb if enabled."""
    metrics = {
        "train_loss": train_loss,
        "eval_loss": eval_loss,
        "train_predictions_mean": np.mean(all_predictions),
        "train_predictions_std": np.std(all_predictions),
        "eval_predictions_mean": np.mean(eval_predictions),
        "eval_predictions_std": np.std(eval_predictions),
        "train/precision": prfs_train[0],
        "train/recall": prfs_train[1],
        "train/f1": prfs_train[2],
        "eval/precision": prfs_eval[0],
        "eval/recall": prfs_eval[1],
        "eval/f1": prfs_eval[2],
    }
    # Micro averages and the class counts behind each macro average: the macro numbers are
    # each over a different subset of GO terms, so they cannot be combined with each other.
    for split, prfs in (("train", prfs_train), ("eval", prfs_eval)):
        for key, value in (prfs[3] or {}).items():
            metrics[f"{split}/{key}"] = value

    if log_wandb:
        wandb.log(metrics, step=epoch + 1)

    print(f"Epoch {epoch + 1}")
    print(f"Train - Loss: {train_loss:.4f}")
    print(f"Eval  - Loss: {eval_loss:.4f}")
    for split, prfs in (("Train", prfs_train), ("Eval ", prfs_eval)):
        extras = prfs[3] or {}
        print(
            f"{split} - macro P/R/F1: {prfs[0]:.4f} / {prfs[1]:.4f} / {prfs[2]:.4f}"
            f" | micro P/R/F1: {extras.get('precision_micro', float('nan')):.4f}"
            f" / {extras.get('recall_micro', float('nan')):.4f}"
            f" / {extras.get('f1_micro', float('nan')):.4f}"
            f" | GO terms predicted/with support/total:"
            f" {extras.get('classes_predicted')}/{extras.get('classes_with_support')}"
            f"/{extras.get('classes_total')}"
        )


def train_model(
    model,
    train_dataloader,
    eval_dataloader,
    num_epochs: int = 20,
    learning_rate=1e-4,
    use_distograms=True,
    use_embeddings=True,
    threshold=0.3,
    log_wandb=True,
    weights=None,
    loss_fn_name=None,
    loss_fn_kwargs=None,
    grad_clip_max_norm: float | None = 1.0,
    max_steps_per_epoch: int | None = None,
):
    """
    Universal training function that can use embeddings, distograms, or both.

    Args:
        model: The model to train
        train_dataloader: DataLoader for training data
        eval_dataloader: DataLoader for evaluation data
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        use_distograms: Whether to use distogram data
        use_embeddings: Whether to use embedding data
        threshold: Threshold for binary classification
        log_wandb: Whether to log per-epoch metrics to the active wandb run
        grad_clip_max_norm: Per-step gradient clipping max norm
        max_steps_per_epoch: Optional cap on train/eval batches per epoch (smoke tests)

    Returns:
        Trained model and evaluation metrics
    """
    device = next(model.parameters()).device

    optimizer, loss_fn = initialize_training(
        model,
        learning_rate,
        weights,
        loss_fn_name=loss_fn_name,
        loss_fn_kwargs=loss_fn_kwargs,
    )
    total_epochs = max(1, int(num_epochs))

    # Training loop
    with tqdm.trange(total_epochs, desc="Training") as pbar:
        for epoch in pbar:
            train_loss, all_predictions, all_targets = train_epoch(
                model,
                train_dataloader,
                optimizer,
                loss_fn,
                device,
                use_embeddings,
                use_distograms,
                pbar,
                max_steps_per_epoch=max_steps_per_epoch,
                grad_clip_max_norm=grad_clip_max_norm,
            )

            eval_loss, eval_predictions, eval_targets = evaluate_model(
                model,
                eval_dataloader,
                loss_fn,
                device,
                use_embeddings,
                use_distograms,
                max_steps_per_epoch=max_steps_per_epoch,
            )

            prfs_train = calculate_metrics(all_predictions, all_targets, threshold)
            prfs_eval = calculate_metrics(eval_predictions, eval_targets, threshold)

            log_metrics(
                train_loss,
                eval_loss,
                all_predictions,
                all_targets,
                eval_predictions,
                eval_targets,
                prfs_train,
                prfs_eval,
                epoch,
                log_wandb,
            )

    return model, {
        "train_loss": train_loss,
        "eval_loss": eval_loss,
        "train_metrics": prfs_train,
        "eval_metrics": prfs_eval,
        "all_predictions": all_predictions,
        "all_targets": all_targets,
        "eval_predictions": eval_predictions,
        "eval_targets": eval_targets,
        "experiment_name": wandb.run.name if wandb.run is not None else None,
    }
