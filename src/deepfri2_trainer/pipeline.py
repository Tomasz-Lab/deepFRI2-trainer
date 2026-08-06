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
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn

from .config import MODEL_TYPES, RunConfig, load_config
from .data import Loaders, Targets, build_loaders, load_targets
from .load_model import build_model
from .outputs import (
    RunLogger,
    append_training_log,
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
    model: nn.Module | None = field(default=None, repr=False)
    loaders: Loaders | None = field(default=None, repr=False)
    targets: Targets | None = field(default=None, repr=False)
    prediction_files: dict[str, Path] = field(default_factory=dict)

    @property
    def summary(self) -> str:
        eval_f1 = self.metrics["eval_metrics"][2]
        return (
            f"{self.run_id}: train_loss={self.metrics['train_loss']:.4f} "
            f"eval_loss={self.metrics['eval_loss']:.4f} eval_f1={eval_f1:.4f} -> {self.run_dir}"
        )


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
        print(f"\n{'=' * 88}\n{cfg.describe()}\n{'=' * 88}")

        parity = None
        if check_architecture_parity:
            parity = check_parity(cfg, device=device)
            print(parity.summary())

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
            num_labels=targets.num_labels,
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

        try:
            model, metrics = run_training(
                cfg, model, loaders, targets,
                log_wandb=log_wandb, max_steps_per_epoch=max_steps_per_epoch,
            )
        except Exception as error:
            append_training_log(cfg, "FAILED", error=type(error).__name__)
            raise

        save_weights(model, cfg)
        save_labels(targets, cfg)

        prediction_files = write_all_predictions(model, loaders, cfg, targets)
        for path in prediction_files.values():
            check_prediction_file(path, targets)

        append_training_log(
            cfg,
            "DONE",
            train_loss=f"{metrics['train_loss']:.4f}",
            eval_loss=f"{metrics['eval_loss']:.4f}",
            eval_f1=f"{metrics['eval_metrics'][2]:.4f}",
            dir=cfg.run_dir,
        )

        return StageResult(
            cfg=cfg,
            run_id=cfg.run_id,
            run_dir=cfg.run_dir,
            metrics=metrics,
            parity=parity,
            sensitivity=sensitivity,
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
