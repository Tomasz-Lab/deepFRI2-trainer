"""Homology-aware train / eval split.

Proteins are clustered by sequence identity with MMseqs2 and split **cluster-wise**, so no
evaluation protein has a close homologue in training -- a random per-protein split would leak
and inflate every metric. Among random cluster assignments the one is kept whose per-GO-term
evaluation fraction is closest to the target, so rare terms stay represented on both sides.
"""

from __future__ import annotations

import csv
import random
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path


def fasta_to_dict(path: Path | str) -> dict[str, str]:
    """``{name: sequence}`` from a multi-FASTA."""
    sequences, name, lines = {}, None, []
    with open(path) as handle:
        for line in handle:
            if line.startswith(">"):
                if name is not None:
                    sequences[name] = "\n".join(lines)
                name, lines = line[1:].strip(), []
            else:
                lines.append(line.strip())
    if name is not None:
        sequences[name] = "\n".join(lines)
    return sequences


def write_fasta(path: Path, sequences: dict[str, str]) -> Path:
    with open(path, "w") as handle:
        for name, sequence in sequences.items():
            handle.write(f">{name}\n{sequence}\n")
    return path


def run_clustering(mmseqs_bin, input_fasta, output_dir: Path, tmp_dir: Path, min_seq_id=0.3) -> Path:
    """Cluster ``input_fasta`` with MMseqs2; returns the ``clusters.tsv`` path.

    The output directory is emptied first -- MMseqs2 refuses to overwrite its own databases.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    for entry in output_dir.glob("*"):
        if entry.name == "tmp":
            continue
        entry.unlink() if entry.is_file() else shutil.rmtree(entry)

    database, clusters = output_dir / "seqDB", output_dir / "clusters"
    tsv = output_dir / "clusters.tsv"
    with open(output_dir / "mmseqs.log", "w") as log:
        for command in (
            [str(mmseqs_bin), "createdb", str(input_fasta), str(database)],
            [str(mmseqs_bin), "cluster", str(database), str(clusters), str(tmp_dir),
             "--min-seq-id", str(min_seq_id)],
            [str(mmseqs_bin), "createtsv", str(database), str(database), str(clusters), str(tsv)],
        ):
            subprocess.run(command, stdout=log, stderr=log, check=True)
    return tsv


def parse_clusters(tsv: Path) -> list[list[str]]:
    clusters = defaultdict(set)
    with open(tsv) as handle:
        for line in handle:
            representative, member = line.split()
            clusters[representative] |= {representative, member}
    return [list(members) for members in clusters.values()]


def choose_split(clusters, proteins_by_go, eval_fraction=0.1, num_trials=100, seed=None):
    """Best of ``num_trials`` random cluster assignments, by per-GO-term evaluation balance.

    Whole clusters are assigned to evaluation until the target size is reached; the score is the
    mean squared deviation of each term's evaluation fraction from ``eval_fraction``.

    ``seed`` makes the search reproducible. The original was unseeded, so splits produced before
    this refactor cannot be regenerated -- reuse the ``train.tsv`` / ``eval.tsv`` on disk for
    those, and seed from here on.
    """
    rng = random.Random(seed)
    target = int(sum(len(cluster) for cluster in clusters) * eval_fraction)

    best_split, best_score = None, float("inf")
    for _ in range(num_trials):
        shuffled = clusters[:]
        rng.shuffle(shuffled)

        evaluation, train, count = set(), set(), 0
        for cluster in shuffled:
            if count < target:
                evaluation.update(cluster)
                count += len(cluster)
            else:
                train.update(cluster)

        deviations = [
            (sum(1 for protein in proteins if protein in evaluation) / len(proteins) - eval_fraction) ** 2
            for proteins in proteins_by_go.values() if proteins
        ]
        score = sum(deviations) / len(deviations) if deviations else float("inf")
        if score < best_score:
            best_score, best_split = score, (train.copy(), evaluation.copy())

    return best_split, best_score


def write_split(train_ids, eval_ids, proteins_by_go, sequences, output_dir: Path) -> None:
    """``train.tsv`` / ``eval.tsv`` (protein, GO term) and the matching FASTA files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, ids in (("train", train_ids), ("eval", eval_ids)):
        with open(output_dir / f"{name}.tsv", "w", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t")
            for go_term, proteins in proteins_by_go.items():
                writer.writerows([protein, go_term] for protein in proteins if protein in ids)
        write_fasta(output_dir / f"{name}.fasta",
                    {i: sequences[i] for i in set(ids) if i in sequences})


def run_split_pipeline(
    fasta_file: Path,
    proteins_by_go: dict[str, list[str]],
    mmseqs_bin: Path | str,
    output_dir: Path,
    tmp_dir: Path,
    min_seq_id: float = 0.5,
    eval_fraction: float = 0.1,
    num_trials: int = 100,
    seed: int | None = None,
):
    """Cluster, choose the best split, write it. Returns ``(train ids, eval ids, score)``."""
    clusters = parse_clusters(run_clustering(mmseqs_bin, fasta_file, output_dir, tmp_dir, min_seq_id))
    (train_ids, eval_ids), score = choose_split(
        clusters, proteins_by_go, eval_fraction, num_trials, seed)
    write_split(train_ids, eval_ids, proteins_by_go, fasta_to_dict(fasta_file), output_dir)
    print(f"Split: {len(train_ids)} train / {len(eval_ids)} eval proteins. Score: {score:.6f}")
    return train_ids, eval_ids, score
