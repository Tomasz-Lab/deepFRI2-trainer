"""deepFRI2-trainer: training, retraining and fine-tuning of deepFRI2 models.

Command line: ``python train.py --help``. From Python::

    from deepfri2_trainer import run_stages

    results = run_stages(ontology="MF", stages=("sequence", "structure", "fusion"))

Or one stage at a time, with the intermediate objects in hand::

    from deepfri2_trainer import load_config, load_targets, build_loaders, build_model
    from deepfri2_trainer import outputs, run_training

    cfg = load_config(model_type="sequence", ontology="MF", train_on="train")
    targets = load_targets(cfg)
    loaders = build_loaders(cfg, targets)
    model = build_model(cfg, targets.num_labels, loaders.emb_size, device="cuda:0")
    outputs.start_run(cfg, model, targets)          # opens wandb, fixes the run name
    model, metrics = run_training(cfg, model, loaders, targets)
"""

from . import import_released, model, outputs, parity, preprocess, sanity
from .config import MODEL_TYPES, ONTOLOGIES, TRAIN_ON, RunConfig, load_config
from .data import Loaders, Targets, build_loaders, load_targets
from .load_model import (
    apply_backend_settings,
    build_fusion_model,
    build_model,
    build_sequence_model,
    build_structure_model,
)
from .outputs import (
    RunLogger,
    save_config,
    save_labels,
    save_source_snapshot,
    save_weights,
    start_run,
)
from .parity import BackendSensitivity, ParityReport, check_parity, probe_backend_sensitivity
from .import_released import import_released_runs
from .preprocess import PreprocessConfig
from .pipeline import (
    STAGE_ORDER,
    StageResult,
    correlate,
    run_stage,
    run_stages,
    select_epoch,
)
from .predict import prediction_path, write_all_predictions, write_predictions
from .train import build_loss_kwargs, run_training

__all__ = [
    "MODEL_TYPES",
    "ONTOLOGIES",
    "STAGE_ORDER",
    "TRAIN_ON",
    "BackendSensitivity",
    "Loaders",
    "ParityReport",
    "RunLogger",
    "RunConfig",
    "StageResult",
    "Targets",
    "build_fusion_model",
    "build_loaders",
    "build_loss_kwargs",
    "build_model",
    "build_sequence_model",
    "build_structure_model",
    "check_parity",
    "correlate",
    "load_config",
    "load_targets",
    "model",
    "PreprocessConfig",
    "import_released",
    "import_released_runs",
    "preprocess",
    "outputs",
    "parity",
    "prediction_path",
    "probe_backend_sensitivity",
    "run_stage",
    "run_stages",
    "run_training",
    "select_epoch",
    "sanity",
    "save_config",
    "save_labels",
    "save_source_snapshot",
    "save_weights",
    "start_run",
    "write_all_predictions",
    "write_predictions",
]
