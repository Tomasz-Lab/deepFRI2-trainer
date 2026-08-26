"""Orchestration: train one stage, or a whole ontology.

Order of operations in a stage -- the wandb run is opened before training because every output
path is named after it::

    parity check -> targets -> loaders -> model -> sanity -> wandb.init
    -> config + source snapshot -> START -> train -> weights/labels -> predictions -> DONE

The stage runs inside a :class:`~deepfri2_trainer.outputs.RunLogger`, so its console output
ends up in ``<run_dir>/log.txt``, including the part printed before the run directory name was
known. Training all three stages produces three run directories, three ``log.txt`` files and
three START/DONE pairs in the shared ``training.log``.
"""

from __future__ import annotations

import gc
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn

from .config import MODEL_TYPES, RunConfig, load_config
from .data import Loaders, Targets, build_loaders, load_targets
from .load_model import build_model, load_weights
from .outputs import (
    RunLogger,
    append_training_log,
    log_selection,
    log_run_artifacts,
    save_config,
    save_labels,
    save_source_snapshot,
    save_weights,
    start_run,
)
from .parity import BackendSensitivity, ParityReport, check_parity, probe_backend_sensitivity
from .predict import write_all_predictions
from .sanity import (
    check_fusion_branches,
    check_label_space,
    check_prediction_file,
    report_trainable_parameters,
)
from .train import run_training
from .utils.training import format_duration, seed_everything

# Sub-models must exist before the fusion gate can be trained.
STAGE_ORDER = ("sequence", "structure", "fusion")


@dataclass
class StageResult:
    """Everything one stage produced."""

    cfg: RunConfig
    run_id: str
    run_dir: Path
    metrics: dict = field(repr=False)
    parity: ParityReport | None = None
    sensitivity: BackendSensitivity | None = None
    selected_epoch: int | None = None
    model: nn.Module | None = field(default=None, repr=False)
    loaders: Loaders | None = field(default=None, repr=False)
    targets: Targets | None = field(default=None, repr=False)
    prediction_files: dict[str, Path] = field(default_factory=dict)

    @property
    def summary(self) -> str:
        record = self.metrics.get("selected") or self.metrics
        return (
            f"{self.run_id}: epoch={self.selected_epoch}/{len(self.metrics.get('history', []))} "
            f"({self.cfg.selection}) train_loss={record['train_loss']:.4f} "
            f"eval_loss={record['eval_loss']:.4f} eval_fmax={record['eval_fmax']:.4f} "
            f"time={format_duration(self.metrics.get('stage_seconds', 0))} "
            f"({format_duration(self.metrics.get('seconds_per_epoch', 0))}/epoch) "
            f"-> {self.run_dir}"
        )


#: selection metric -> (key, "higher is better"?)
SELECTION_METRICS = {"eval_fmax": ("eval_fmax", True), "eval_loss": ("eval_loss", False)}


def select_epoch(
    history: list[dict], selection: str, metric: str = "eval_fmax", tolerance: float = 0.0
) -> tuple[dict, str]:
    """The epoch whose weights the run ships, and which rolling checkpoint holds them.

    Training keeps two checkpoints only -- the optimum of ``metric`` so far (``_best.pth``) and
    the most recent epoch (``_last.pth``) -- so those are the only two epochs a run can ship,
    and every selection rule has to resolve to one of them:

    ``last``        the final epoch.
    ``best_strict`` the optimum of ``metric``, whenever it occurred.
    ``best``        the final epoch when it comes within ``tolerance`` of the optimum, the
                    optimum otherwise. Fmax wobbles by a few thousandths between epochs, and
                    a run that stops on an early lucky epoch while the model is still
                    improving generalises worse than one that trained to the end; when the
                    final epoch is as good as the peak within that noise, it is the safer
                    checkpoint. Once the drop is larger than the noise, the optimum wins.

    Returns ``(record, kind)`` where ``kind`` is ``"best"`` or ``"last"`` -- the checkpoint the
    caller must load. Do not infer the file from the epoch number: the tolerance rule used to
    return an interior epoch that was never written to disk, and the run then silently shipped
    ``_best.pth`` while reporting the interior epoch's metrics.
    """
    if not history:
        raise ValueError("training produced no epochs")
    if selection == "last":
        return history[-1], "last"

    key, higher_is_better = SELECTION_METRICS[metric]
    optimum_record = (max if higher_is_better else min)(history, key=lambda r: r[key])
    if optimum_record["epoch"] == history[-1]["epoch"]:
        return history[-1], "last"
    if selection == "best_strict":
        return optimum_record, "best"

    slack = (history[-1][key] - optimum_record[key]) * (1 if higher_is_better else -1)
    if slack >= -tolerance:
        return history[-1], "last"
    return optimum_record, "best"


def correlate(history: list[dict], left: str, right: str) -> tuple[float, float]:
    """Pearson and Spearman between two per-epoch series."""
    from scipy.stats import pearsonr, spearmanr  # noqa: PLC0415

    a = [record[left] for record in history]
    b = [record[right] for record in history]
    if len(a) < 3 or len(set(a)) < 2 or len(set(b)) < 2:
        return float("nan"), float("nan")
    return float(pearsonr(a, b)[0]), float(spearmanr(a, b)[0])


def run_stage(
    cfg: RunConfig,
    device: str | torch.device = "cuda:0",
    log_wandb: bool = True,
    max_steps_per_epoch: int | None = None,
    check_architecture_parity: bool = True,
    logger: list | None = None,
) -> StageResult:
    """Train one model end to end: parity -> data -> model -> training -> outputs.

    ``logger``, when given a list, collects this stage's :class:`RunLogger` so a caller can
    append to the log after the stage has finished (``run_stages`` appends the summary).
    """
    with RunLogger() as run_logger:
        if logger is not None:
            logger.append(run_logger)
        stage_started = time.perf_counter()
        # Stamped once: the config is written twice (before training, and again once the
        # shipped epoch is known) and both writes must report the same start time.
        started_at = f"{datetime.now():%Y-%m-%d %H:%M:%S}"
        print(f"\n{'=' * 88}\n{cfg.describe()}\n{'=' * 88}")

        parity = None
        if check_architecture_parity:
            parity = check_parity(cfg, device=device)
            print(parity.summary())

        # Seed here, not earlier: the parity check builds throwaway models and would
        # otherwise consume the RNG that initialises the model we actually train.
        print(f"seed: {seed_everything(cfg.seed)}")

        targets = load_targets(cfg)
        loaders = build_loaders(cfg, targets)
        print(f"{targets.num_labels} GO terms | embedding size {loaders.emb_size}")

        # build_model applies the run's backend flags, so probe after it.
        model = build_model(cfg, targets.num_labels, loaders.emb_size, device)

        sensitivity = probe_backend_sensitivity(cfg, device) if check_architecture_parity else None
        if sensitivity is not None:
            print(sensitivity.summary())

        check_label_space(targets, model)
        report_trainable_parameters(
            model, expect_only="refine_gate" if cfg.model_type == "fusion" else None
        )
        if cfg.model_type == "fusion":
            check_fusion_branches(model, loaders, cfg)

        # Opens the wandb run and fixes cfg.run_name -> run directory and output file names.
        start_run(cfg, model, targets, log_wandb=log_wandb)
        run_logger.attach(cfg.log_path)

        save_config(
            cfg, model=model, parity=parity, sensitivity=sensitivity,
            num_labels=targets.num_labels, timestamp=started_at,
        )
        save_source_snapshot(cfg, parity=parity)
        log_run_artifacts(cfg)

        append_training_log(
            cfg,
            "START",
            num_labels=targets.num_labels,
            epochs=cfg.training["num_epochs"],
            lr=cfg.training["learning_rate"],
            loss=cfg.training["loss"].get("name") or "WeightedFocalLoss",
            train_on=cfg.train_on,
            train_batches=len(loaders.train),
            parity=None if parity is None else parity.status,
            cudnn_tf32=torch.backends.cudnn.allow_tf32 if cfg.use_distograms else None,
                weights=",".join(f"{k}={v}" for k, v in sorted(cfg.weights.items())) or None,
        )

        # Two rolling checkpoints: the best epoch so far, by cfg.selection_metric, and the most
        # recent one. Keeping every epoch would cost 20x the disk for weights nothing reads.
        key, higher_is_better = SELECTION_METRICS[cfg.selection_metric]
        best_so_far: dict = {}

        def improved(record: dict) -> bool:
            if not best_so_far:
                return True
            return (record[key] > best_so_far[key]) if higher_is_better else (
                record[key] < best_so_far[key])

        def checkpoint_epoch(epoch: int, trained, record: dict) -> None:
            torch.save(trained.state_dict(), cfg.candidate_checkpoint_path("last"))
            kept = "last"
            if improved(record):
                best_so_far.clear()
                best_so_far.update(record)
                torch.save(trained.state_dict(), cfg.candidate_checkpoint_path("best"))
                kept = "best+last"
            print(
                f"epoch {epoch}: {key}={record[key]:.4f} -> {kept} "
                f"(best so far: epoch {best_so_far['epoch']}, {best_so_far[key]:.4f})"
            )

        try:
            model, metrics = run_training(
                cfg, model, loaders, targets,
                log_wandb=log_wandb, max_steps_per_epoch=max_steps_per_epoch,
                on_epoch_end=checkpoint_epoch,
            )
        except Exception as error:
            append_training_log(cfg, "FAILED", error=type(error).__name__)
            raise

        # `selection` decides which of the two the run ships and predicts with. The last
        # epoch is not necessarily the best: eval loss bottoms out early and then rises, and
        # out-of-distribution sets (CAZy) feel that first.
        history = metrics["history"]
        tolerance = float(cfg.training.get("selection_tolerance", 0.0))
        selected, checkpoint = select_epoch(
            history, cfg.selection, cfg.selection_metric, tolerance
        )
        metrics["selected"] = selected
        metrics["selected_checkpoint"] = checkpoint
        metrics["best"] = select_epoch(history, "best_strict", cfg.selection_metric)[0]
        final = history[-1]
        print(
            f"selection={cfg.selection} by {cfg.selection_metric} (tolerance {tolerance}): "
            f"ships epoch {selected['epoch']}/{len(history)} from "
            f"{cfg.candidate_checkpoint_path(checkpoint).name} "
            f"(eval_fmax={selected['eval_fmax']:.4f} train_fmax={selected['train_fmax']:.4f} "
            f"eval_loss={selected['eval_loss']:.4f})"
        )
        print(
            f"  optimum epoch {metrics['best']['epoch']} ({key}={metrics['best'][key]:.4f}), "
            f"final epoch {final['epoch']} ({key}={final[key]:.4f}), "
            f"gap {abs(final[key] - metrics['best'][key]):.4f} vs tolerance {tolerance}"
        )
        if cfg.train_on == "train+eval":
            collapsed = "yes" if checkpoint == "last" else "no"
            print(
                f"  NOTE: train_on=train+eval -- the eval split is inside the training set, so "
                f"every eval_* number above (including the {cfg.selection_metric} this selection "
                f"is based on) is a training metric, not a held-out one. With no held-out signal "
                f"the metric normally improves monotonically and selection collapses to the final "
                f"epoch (collapsed here: {collapsed}). Judge this run on the test / CAZy sets."
            )

        # Does the train split track eval at all? If these are strongly correlated the model is
        # still learning generalisable structure; once they diverge, later epochs only fit train.
        for name, (left, right) in {
            "Fmax  train vs eval": ("train_fmax", "eval_fmax"),
            "loss  train vs eval": ("train_loss", "eval_loss"),
        }.items():
            pearson, spearman = correlate(history, left, right)
            metrics[f"correlation_{left}_{right}"] = pearson
            print(f"  {name}: pearson={pearson:+.3f} spearman={spearman:+.3f}")
        if checkpoint != "last":
            load_weights(model, cfg.candidate_checkpoint_path(checkpoint))
        log_selection(
            cfg, selected, len(history), checkpoint=checkpoint,
            log_wandb=log_wandb, timings=metrics,
        )
        # Rewrite the config now that the shipped epoch is known, so `config_<run>.yaml`
        # records which epoch `<run>.pth` actually holds.
        save_config(
            cfg, model=model, parity=parity, sensitivity=sensitivity,
            num_labels=targets.num_labels, timestamp=started_at,
            finished_at=f"{datetime.now():%Y-%m-%d %H:%M:%S}",
            selected_epoch=selected["epoch"], selected_checkpoint=checkpoint,
            total_epochs=len(history), optimum_epoch=metrics["best"]["epoch"],
            # train+eval trains on the eval split, so eval_* above are not held out.
            eval_is_held_out=cfg.train_on == "train",
        )

        save_weights(model, cfg)
        save_labels(targets, cfg)

        prediction_files = write_all_predictions(model, loaders, cfg, targets)
        for path in prediction_files.values():
            check_prediction_file(path, targets)

        metrics["stage_seconds"] = time.perf_counter() - stage_started
        print(
            f"stage time: {format_duration(metrics['stage_seconds'])} total "
            f"({format_duration(metrics['training_seconds'])} training, "
            f"{format_duration(metrics['seconds_per_epoch'])} per epoch)"
        )

        append_training_log(
            cfg,
            "DONE",
            epoch=f"{selected['epoch']}/{len(metrics['history'])}",
            checkpoint=f"{cfg.candidate_checkpoint_path(checkpoint).name}",
            optimum_epoch=metrics["best"]["epoch"],
            selection=cfg.selection,
            train_on=cfg.train_on,
            seed=cfg.seed,
            train_loss=f"{selected['train_loss']:.4f}",
            eval_loss=f"{selected['eval_loss']:.4f}",
            eval_fmax=f"{selected['eval_fmax']:.4f}",
            train_fmax=f"{selected['train_fmax']:.4f}",
            eval_f1=f"{selected['eval_metrics'][2]:.4f}",
            time=format_duration(metrics["stage_seconds"]),
            per_epoch=format_duration(metrics["seconds_per_epoch"]),
            dir=cfg.run_dir,
        )

        return StageResult(
            cfg=cfg,
            run_id=cfg.run_id,
            run_dir=cfg.run_dir,
            metrics=metrics,
            parity=parity,
            sensitivity=sensitivity,
            selected_epoch=selected["epoch"],
            model=model,
            loaders=loaders,
            targets=targets,
            prediction_files=prediction_files,
        )


def run_stages(
    ontology: str,
    stages: tuple[str, ...] = STAGE_ORDER,
    train_on: str = "train",
    device: str | torch.device = "cuda:0",
    log_wandb: bool = True,
    config_dir: Path | str | None = None,
    overrides: dict | None = None,
    max_steps_per_epoch: int | None = None,
    check_architecture_parity: bool = True,
    keep_models: bool = False,
) -> dict[str, StageResult]:
    """Train the requested stages for one ontology, in dependency order.

    When ``sequence`` / ``structure`` are trained in the same call, their run names are passed
    to the fusion stage automatically; otherwise fusion uses ``weights`` from
    ``configs/fusion.yaml``.

    ``keep_models=False`` (the default) drops each model and its dataloaders once its stage is
    done, so training all three in one process does not accumulate GPU/host memory.
    """
    unknown = [stage for stage in stages if stage not in MODEL_TYPES]
    if unknown:
        raise ValueError(f"unknown stage(s) {unknown}; expected any of {MODEL_TYPES}")
    ordered = [stage for stage in STAGE_ORDER if stage in stages]

    results: dict[str, StageResult] = {}
    loggers: list[RunLogger] = []
    for stage in ordered:
        stage_overrides = dict(overrides or {})

        if stage == "fusion":
            trained = {
                sub: results[sub].cfg.run_name
                for sub in ("sequence", "structure")
                if sub in results
            }
            if trained:
                # Deep-merged onto configs/fusion.yaml, so a sub-model that was not retrained
                # here keeps the reference declared in the config.
                print(f"fusion: using sub-models trained in this run: {trained}")
                stage_overrides["weights"] = {ontology: trained}
            else:
                print("fusion: using the sub-models declared in configs/fusion.yaml")

        cfg = load_config(stage, ontology, train_on, config_dir, overrides=stage_overrides)
        results[stage] = run_stage(
            cfg,
            device=device,
            log_wandb=log_wandb,
            max_steps_per_epoch=max_steps_per_epoch,
            check_architecture_parity=check_architecture_parity,
            logger=loggers,
        )

        if not keep_models:
            result = results[stage]
            result.model = None
            result.loaders = None
            result.targets = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    summary = "\n".join(
        ["", "===== summary =====", *(f"{stage:<10} {r.summary}" for stage, r in results.items())]
    )
    print(summary)
    # The stage loggers have already closed, so append the summary to the last stage's log
    # directly: it is the one place that shows how the whole invocation went.
    if loggers:
        loggers[-1].append(summary)

    return results
