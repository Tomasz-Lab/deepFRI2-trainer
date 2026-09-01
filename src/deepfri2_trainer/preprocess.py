"""
The step between FRIdata and ``train.py``. FRIdata turns a list of protein IDs into sequences,
distograms and embeddings; this module turns the annotation tables and the GO graph into the
*supervision*: which GO terms the model has an output for, the sparse label vector of every
protein, the class weights, the GO adjacency, the homology-aware train/eval split, and the
ground-truth tables the CAFA evaluation scores against.

Three steps, independently runnable (``python preprocess.py --help``):

``targets``  the eight target-matrix pickles, plus the test-set FASTA
``split``    the train/eval FASTA, MMseqs2 clustering and the balanced cluster split
``cazy``     label vectors for the CAZy test set, against the ``go_indices`` ``targets`` produced

Outputs are written per ontology, under the same names and directory shape the previous notebook
produced, so a trainer configured at the new root finds everything where it expects it.
"""

from __future__ import annotations

import json
import pickle
import shutil
import socket
import sys
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Sequence

import pandas as pd
import yaml

from .config import CONFIG_DIR, ONTOLOGIES, _deep_merge
from .utils.split import fasta_to_dict, run_split_pipeline, write_fasta
from .utils.target_matrix import TARGET_FILES, InferenceTargetMatrix, TargetMatrix, load_go_graphs

STEPS = ("targets", "split", "cazy")

#: Split files copied by ``split.adopt_from``. Only the two TSVs are read downstream (by the
#: dataloader and by the CAFA evaluation); the rest travel along as provenance.
SPLIT_ARTEFACTS = ("train.tsv", "eval.tsv", "train.fasta", "eval.fasta", "clusters.tsv")

#: How protein accessions appear in the FRIdata sequence index and in the predictions.
AFDB_ID = "AF-{accession}-F1-model_v4_A"

WIDTH = 88

def rule(title: str) -> None:
    print(f"\n{'-' * 8} {title} {'-' * max(0, WIDTH - len(title) - 10)}")


def note(ontology: str, message: str) -> None:
    """One console line, tagged with the ontology it belongs to."""
    print(f"[{ontology}] {message}")


class _Tee:
    """Write to the console and to the transcript at once."""

    def __init__(self, stream, handle):
        self._stream, self._handle = stream, handle

    def write(self, text):
        self._stream.write(text)
        self._handle.write(text)
        return len(text)

    def flush(self):
        self._stream.flush()
        self._handle.flush()

    def __getattr__(self, name):
        return getattr(self._stream, name)


class Transcript:
    """Append everything a run prints to ``data.log``, under a header naming the run.

    Target-matrix construction is not atomic the way a training run is -- one invocation writes
    several derivatives across several ontologies, and which of them a file came from is not
    recoverable from a summary line. So the log keeps the whole console output instead: the
    command, the resolved configuration, every count printed along the way, and the traceback if
    it failed. That is enough for someone who did not run it to see how the data was made.
    """

    def __init__(self, path: Path, command: str):
        self.path, self.command = path, command
        self._handle = None
        self._saved: tuple = ()
        self._started = None

    def __enter__(self) -> "Transcript":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = open(self.path, "a", buffering=1)
        self._started = datetime.now()
        self._handle.write(
            f"\n{'=' * WIDTH}\n"
            f"{self._started:%Y-%m-%d %H:%M:%S} | {self.command}\n"
            f"{'=' * WIDTH}\n"
            # f"host: {socket.gethostname()}   cwd: {Path.cwd()}\n"
        )
        self._saved = (sys.stdout, sys.stderr)
        sys.stdout, sys.stderr = _Tee(sys.stdout, self._handle), _Tee(sys.stderr, self._handle)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._saved:
            sys.stdout, sys.stderr = self._saved
        elapsed = str(datetime.now() - self._started).split(".")[0]
        if exc_type is not None:
            self._handle.write("\n" + "".join(traceback.format_exception(exc_type, exc, tb)))
        self._handle.write(
            f"\n{datetime.now():%Y-%m-%d %H:%M:%S} | "
            f"{'FAILED' if exc_type else 'DONE'} in {elapsed}\n"
        )
        self._handle.close()


@dataclass
class PreprocessConfig:
    """Paths and parameters for one preprocessing run, from ``configs/``.

    ``paths`` holds the templates of ``paths.yaml :: preprocess``; every ``{placeholder}`` is
    filled from the fields below, so relocating the data tree is a config edit.
    """

    project_location: Path
    data_version: str
    go_version: str
    annotation_threshold: int
    dataset_name: str
    paths: dict[str, str]
    qualities: list[str] = field(default_factory=lambda: ["HQ"])
    exclude_roots: bool = False
    split: dict = field(default_factory=dict)
    cazy: dict = field(default_factory=dict)
    go_indices_from: str | None = None

    @classmethod
    def from_configs(cls, config_dir: Path | str | None = None, **overrides) -> "PreprocessConfig":
        config_dir = Path(config_dir) if config_dir is not None else CONFIG_DIR
        paths = yaml.safe_load((config_dir / "paths.yaml").read_text()) or {}
        raw = yaml.safe_load((config_dir / "data.yaml").read_text()) or {}
        data, settings = raw["data"], raw.get("preprocess", {})

        version = str(data["data_version"])
        # nested overrides merge into the config block rather than replacing it, so
        # `--set cazy.mapping=...` keeps the rest of the cazy settings
        defaults = {
            "project_location": Path(paths["project_location"]),
            "data_version": version,
            "go_version": str(data["go_version"]),
            "annotation_threshold": int(data["annotation_threshold"]),
            "dataset_name": data["dataset_name"].format(data_version=version),
            "paths": paths["preprocess"],
            "qualities": settings.get("qualities", ["HQ"]),
            "exclude_roots": bool(settings.get("exclude_roots", False)),
            "split": settings.get("split", {}),
            "cazy": settings.get("cazy", {}),
            "go_indices_from": settings.get("go_indices_from"),
        }
        return cls(**_deep_merge(defaults, overrides))

    # ---------- names ----------

    def params(self, ontology: str) -> str:
        """``20250722__MF__50`` -- the same key the trainer's ``RunConfig.params`` builds."""
        return f"{self.go_version}__{ontology}__{self.annotation_threshold}"

    def path(self, key: str, **extra) -> Path:
        return Path(self.paths[key].format(
            project_location=self.project_location,
            data_version=self.data_version,
            go_version=self.go_version,
            dataset_name=self.dataset_name,
            **extra,
        ))

    # ---------- inputs ----------

    @property
    def inputs_dir(self) -> Path:
        return self.path("inputs_dir")

    @property
    def graphs_dir(self) -> Path:
        """GO DAGs (``graph_<ontology>.json``) -- a light primitive, kept in the repo tree."""
        return self.inputs_dir / "graphs" / self.go_version

    @property
    def annotations_dir(self) -> Path:
        """``annots_<ontology>.pickle`` -- gigabytes, left where the annotation pipeline wrote it."""
        return self.path("annotations_dir")

    @property
    def unified_file(self) -> Path:
        """Annotation metadata -- gigabytes, left in place."""
        return self.path("unified_file")

    @property
    def sequences_index(self) -> Path:
        """FRIdata's ``sequences.idx``: protein id -> the FASTA holding it."""
        return self.path("sequences_idx")

    @property
    def fridata_dir(self) -> Path:
        """Root the ``sequences.idx`` values are relative to."""
        return self.path("fridata_dir")

    @property
    def mmseqs_bin(self) -> Path:
        return self.path("mmseqs_bin")

    @property
    def log_file(self) -> Path:
        """Append-only record of every preprocessing run, like ``runs_dir/training.log``."""
        return self.path("log_file")

    def cazy_input(self, kind: str) -> Path:
        """``data`` (the UniProt subset) or ``mapping`` (protein -> CAZy family + GO terms)."""
        mapping = self.cazy["mapping"]
        dataset = self.cazy["dataset_name"].format(data_version=self.data_version)
        names = {
            "data": f"data__{mapping}__seqid{self.cazy['min_seq_id']}"
                    f"__cov{self.cazy['coverage']}__uniprot.csv",
            "mapping": f"mapped_sequences_{dataset}_{mapping}.csv",
        }
        return self.inputs_dir / "cazy" / names[kind]

    # ---------- outputs ----------

    def ontology_dir(self, ontology: str) -> Path:
        return self.path("out_dir") / self.dataset_name / self.params(ontology)

    def cazy_dir(self, ontology: str) -> Path:
        return (self.path("out_dir") / "cazy" / self.data_version / "uniprot"
                / f"{ontology}_{self.annotation_threshold}")

    def existing_go_indices(self, ontologies) -> dict[str, dict]:
        """The ``go_indices`` orderings to adopt, from ``go_indices_from``, if it is set.

        The template may use ``{dataset_name}``, ``{params}`` and ``{ontology}``; it points at a
        directory holding a ``go_indices.pkl``.
        """
        if not self.go_indices_from:
            return {}
        existing = {}
        for ontology in ontologies:
            directory = Path(self.go_indices_from.format(
                project_location=self.project_location,
                data_version=self.data_version,
                go_version=self.go_version,
                dataset_name=self.dataset_name,
                params=self.params(ontology),
                ontology=ontology,
                threshold=self.annotation_threshold,
            ))
            existing[ontology] = _load(directory, "go_indices.pkl")[ontology]
        return existing

    def _resolve(self, template: str | None, ontology: str = "<ontology>") -> str | None:
        """Fill a path template for display; ``{params}`` needs an ontology, hence the default."""
        if not template:
            return None
        return template.format(
            project_location=self.project_location, data_version=self.data_version,
            go_version=self.go_version, dataset_name=self.dataset_name,
            params=self.params(ontology), ontology=ontology,
            threshold=self.annotation_threshold)

    def describe(self) -> str:
        return "\n".join([
            f"dataset            : {self.dataset_name}",
            f"data / GO version  : {self.data_version} / {self.go_version}",
            f"threshold          : {self.annotation_threshold}   qualities: {self.qualities}",
            f"exclude roots      : {self.exclude_roots}",
            f"GO graphs          : {self.graphs_dir}",
            f"annotations        : {self.annotations_dir}",
            f"unified table      : {self.unified_file}",
            f"sequence index     : {self.sequences_index}",
            f"reuse go_indices   : {self._resolve(self.go_indices_from) or '(no -- derived from the GO graph)'}",
            f"adopt split        : {self._resolve(self.split.get('adopt_from')) or '(no -- computed with mmseqs, seed ' + str(self.split.get('seed')) + ')'}",
            f"output root        : {self.path('out_dir')}",
            f"log                : {self.log_file}",
        ])


def _save(directory: Path, name: str, obj) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    with open(directory / name, "wb") as handle:
        pickle.dump(obj, handle)
    return directory / name


def _load(directory: Path, name: str):
    with open(directory / name, "rb") as handle:
        return pickle.load(handle)


# --------------------------------------------------------------------------- steps


def build_targets(cfg: PreprocessConfig, ontologies: Sequence[str]) -> dict[str, dict]:
    """Step ``targets``: the eight pickles per ontology, plus the test-set FASTA.

    ``TargetMatrix`` runs once for all requested ontologies, so the multi-gigabyte annotation
    metadata is read once; the results are then written one directory per ontology, each pickle
    keyed by that ontology -- the shape the trainer and the CAFA evaluation read.
    """
    artefacts = TargetMatrix(
        annotations_dir=cfg.annotations_dir,
        unified_file=cfg.unified_file,
        graphs_dir=cfg.graphs_dir,
        ontologies=list(ontologies),
        qualities=cfg.qualities,
        threshold=cfg.annotation_threshold,
        exclude_roots=cfg.exclude_roots,
        reuse_go_indices=cfg.existing_go_indices(ontologies),
    ).create_targets()

    by_ontology = {}
    for ontology in ontologies:
        out = cfg.ontology_dir(ontology)
        targets = {name: artefact[ontology] for name, artefact in zip(TARGET_FILES, artefacts)}
        for name, artefact in targets.items():
            _save(out / "target_matrix", name, {ontology: artefact})

        # the test FASTA is written from the UniProt sequences carried by the ground truth
        sequences = targets["grand_truth_test.pkl"][["DB_Object_ID", "Sequence"]].drop_duplicates()
        write_fasta(out / "sequences_test.fasta",
                    dict(zip(sequences["DB_Object_ID"], sequences["Sequence"])))
        note(ontology, f"wrote {len(TARGET_FILES)} pickles + sequences_test.fasta -> {out}")
        by_ontology[ontology] = {name: {ontology: artefact} for name, artefact in targets.items()}
    return by_ontology


def annotation_table(go_indices: dict, protein_vectors: dict) -> pd.DataFrame:
    """Label vectors -> ``(ProteinID, GO_Term)``, with accessions in their AFDB form."""
    annotations = TargetMatrix.create_annotation_dataframe(protein_vectors, go_indices)
    annotations["ProteinID"] = annotations["ProteinID"].map(
        lambda accession: AFDB_ID.format(accession=accession))
    return annotations


def build_trainval_fasta(cfg: PreprocessConfig, protein_ids: set[str], destination: Path) -> Path:
    """Collect the sequences of ``protein_ids`` out of the FRIdata FASTA files.

    ``sequences.idx`` maps each protein to the (large, shared) FASTA holding it, so each file is
    read once rather than once per protein.
    """
    with open(cfg.sequences_index) as handle:
        index = {key.replace(".pdb", ""): value for key, value in json.load(handle).items()}

    mapping = {key: value for key, value in index.items() if key in protein_ids}
    missing = protein_ids - set(mapping)
    if missing:
        raise KeyError(
            f"{len(missing)} annotated proteins are absent from {cfg.sequences_index} "
            f"(e.g. {sorted(missing)[:3]}); regenerate the FRIdata dataset for these IDs.")

    sequences = {}
    for fasta in sorted(set(mapping.values())):
        source = fasta_to_dict(cfg.fridata_dir / fasta)
        sequences.update({key: source[key] for key, value in mapping.items() if value == fasta})
    return write_fasta(destination, sequences)


def _split_size(tsv: Path) -> int:
    """Distinct proteins in a ``(protein, GO term)`` split file."""
    with open(tsv) as handle:
        return len({line.split("\t", 1)[0] for line in handle if line.strip()})


def adopt_split(cfg: PreprocessConfig, ontology: str) -> None:
    """Copy an existing split instead of computing one.

    Splits produced before the trial loop was seeded cannot be regenerated, so a label space
    that already has trained models against it must keep the split those models were trained and
    evaluated on -- recomputing would quietly move proteins between train and eval and make
    every existing score incomparable.
    """
    source = Path(cfg.split["adopt_from"].format(
        project_location=cfg.project_location, data_version=cfg.data_version,
        go_version=cfg.go_version, dataset_name=cfg.dataset_name,
        params=cfg.params(ontology), ontology=ontology,
        threshold=cfg.annotation_threshold))
    destination = cfg.ontology_dir(ontology) / "mmseqs_output"
    destination.mkdir(parents=True, exist_ok=True)

    missing = [name for name in ("train.tsv", "eval.tsv") if not (source / name).is_file()]
    if missing:
        raise FileNotFoundError(f"cannot adopt the split from {source}: {missing} not found")

    adopted = []
    for name in SPLIT_ARTEFACTS:
        if (source / name).is_file():
            shutil.copy2(source / name, destination / name)
            adopted.append(name)
    trainval = source.parent / "sequences_trainval.fasta"
    if trainval.is_file():
        shutil.copy2(trainval, cfg.ontology_dir(ontology) / trainval.name)
        adopted.append(trainval.name)

    counts = {name: _split_size(destination / f"{name}.tsv") for name in ("train", "eval")}
    note(ontology, f"adopted {len(adopted)} files from {source}")
    note(ontology, f"  {', '.join(adopted)}")
    note(ontology, f"  {counts['train']:,} train / {counts['eval']:,} eval proteins")


def build_split(cfg: PreprocessConfig, ontology: str, targets: dict | None = None) -> None:
    """Step ``split``: trainval FASTA, MMseqs2 clustering, and the balanced cluster split.

    With ``split.adopt_from`` set, an existing split is copied instead -- see :func:`adopt_split`.
    """
    if cfg.split.get("adopt_from"):
        return adopt_split(cfg, ontology)

    out = cfg.ontology_dir(ontology)
    if targets is None:
        targets = {name: _load(out / "target_matrix", name)
                   for name in ("go_indices.pkl", "protein_vectors.pkl")}

    annotations = annotation_table(targets["go_indices.pkl"][ontology],
                                   targets["protein_vectors.pkl"][ontology])
    proteins_by_go = annotations.groupby("GO_Term")["ProteinID"].agg(list).to_dict()

    fasta = build_trainval_fasta(cfg, set(annotations["ProteinID"]),
                                 out / "sequences_trainval.fasta")
    train_ids, eval_ids, score = run_split_pipeline(
        fasta_file=fasta,
        proteins_by_go=proteins_by_go,
        mmseqs_bin=cfg.mmseqs_bin,
        output_dir=out / "mmseqs_output",
        tmp_dir=out / "tmp",
        min_seq_id=cfg.split.get("min_seq_id", 0.5),
        eval_fraction=cfg.split.get("eval_fraction", 0.1),
        num_trials=cfg.split.get("num_trials", 1000),
        seed=cfg.split.get("seed"),
    )
    note(ontology, f"split -> {out / 'mmseqs_output'}")


def build_cazy(cfg: PreprocessConfig, ontology: str, targets: dict | None = None) -> None:
    """Step ``cazy``: label vectors for the CAZy test set, in the model's own GO-term order."""
    out = cfg.ontology_dir(ontology)
    if targets is None:
        targets = {"go_indices.pkl": _load(out / "target_matrix", "go_indices.pkl")}
    go_indices = {ontology: targets["go_indices.pkl"][ontology]}

    data = pd.read_csv(cfg.cazy_input("data"))
    mapping = pd.read_csv(cfg.cazy_input("mapping"))
    mapping["GO"] = mapping["GO"].str.strip("[]").str.replace("'", "").str.split()

    # keep the UniProt subset, then one row per protein carrying all its families and GO terms
    mapping = mapping[mapping["protein"].isin(data["sequence_id"])]
    mapping = mapping.groupby("protein").agg({
        "cazy": lambda values: list(values.unique()),
        "GO": lambda values: list(set(sum(values, []))),
    }).reset_index()
    mapping = mapping.join(
        data[["sequence_id", "AlphaFoldDB_Entry"]].set_index("sequence_id"), on="protein")
    assert len(data) == len(mapping)

    graphs, nodes = load_go_graphs(cfg.graphs_dir, [ontology])
    cazy_go_indices, protein_vectors, grand_truth = InferenceTargetMatrix(
        annotations=mapping[["AlphaFoldDB_Entry", "GO"]].copy(),
        go_indices=go_indices,
        go_graph_nodes=nodes,
        go_graphs=graphs,
        protein_column="AlphaFoldDB_Entry",
        go_column="GO",
        propagate_terms=True,
    ).create_targets()
    assert cazy_go_indices[ontology] == go_indices[ontology]

    destination = cfg.cazy_dir(ontology)
    for name, artefact in (("go_indices.pkl", cazy_go_indices),
                           ("protein_vectors.pkl", protein_vectors),
                           ("grand_truth.pkl", grand_truth)):
        _save(destination, name, artefact)
    note(ontology, f"wrote 3 pickles -> {destination}")


def run(
    cfg: PreprocessConfig,
    ontologies: Sequence[str] = ONTOLOGIES,
    steps: Sequence[str] = STEPS,
    command: str | None = None,
) -> None:
    """Run the requested steps for the requested ontologies, transcribing to ``data.log``.

    Grouped by step rather than by ontology -- `targets` is one pass over the annotation tables
    for all of them, and reading the console top to bottom then follows the pipeline rather than
    interleaving unrelated work.
    """
    unknown = set(steps) - set(STEPS)
    if unknown:
        raise ValueError(f"unknown steps {sorted(unknown)}; expected any of {STEPS}")

    ontologies, steps = list(ontologies), list(steps)
    command = command or (f"preprocess.run(ontologies={ontologies}, steps={steps})")

    with Transcript(cfg.log_file, command):
        print(cfg.describe())
        print(f"ontologies         : {', '.join(ontologies)}")
        print(f"steps              : {', '.join(steps)}")

        targets = {}
        if "targets" in steps:
            rule(f"targets ({', '.join(ontologies)})")
            targets = build_targets(cfg, ontologies)
        if "split" in steps:
            rule(f"split ({', '.join(ontologies)})")
            for ontology in ontologies:
                build_split(cfg, ontology, targets.get(ontology))
        if "cazy" in steps:
            rule(f"cazy ({', '.join(ontologies)})")
            for ontology in ontologies:
                build_cazy(cfg, ontology, targets.get(ontology))

        rule("summary")
        for ontology in ontologies:
            note(ontology, f"{cfg.ontology_dir(ontology)}")
            if "cazy" in steps:
                note(ontology, f"{cfg.cazy_dir(ontology)}")
        print(f"\nlog: {cfg.log_file}")
