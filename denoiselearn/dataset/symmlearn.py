"""Rebuild the SymmLearn image channel directly from the seed registry.

The historical eight-channel generator calls ``compute_symm_maps`` only after
``PGLattice.get_image`` returns.  This module intentionally stops at
``PGImage.img`` and therefore never computes or stores symmetry maps.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
import subprocess
import sys
from typing import Iterable

import numpy as np
import pandas as pd

from .contracts import DEFAULT_IMAGE_SIZE, ImageContract, validate_2d_image


@dataclass(frozen=True)
class SymmLearnSeed:
    registry_row: int
    pg_number: int
    seed: int
    structure: str
    original_image_size: int
    label: int


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_seed_registry(path: str | Path) -> list[SymmLearnSeed]:
    """Load the five-column, headerless SymmLearn seed registry."""

    frame = pd.read_excel(
        path,
        header=None,
        names=["PG_number", "Seeds", "Structures", "Image_size", "Labels"],
    )
    if frame.isna().any().any():
        raise ValueError("SymmLearn seed registry contains missing values")
    records = []
    for index, row in frame.iterrows():
        records.append(
            SymmLearnSeed(
                registry_row=int(index),
                pg_number=int(row.PG_number),
                seed=int(row.Seeds),
                structure=str(row.Structures),
                original_image_size=int(row.Image_size),
                label=int(row.Labels),
            )
        )
    return records


def select_stratified_records(
    records: Iterable[SymmLearnSeed], count: int = 10
) -> list[SymmLearnSeed]:
    """Select deterministic smoke records spread across plane groups."""

    records = list(records)
    groups = sorted({record.pg_number for record in records})
    if count > len(groups):
        raise ValueError(f"count={count} exceeds the {len(groups)} available plane groups")
    selected_groups = [groups[i] for i in np.linspace(0, len(groups) - 1, count, dtype=int)]
    return [next(record for record in records if record.pg_number == group) for group in selected_groups]


def _import_generator(repo_path: str | Path):
    repo_path = Path(repo_path).resolve()
    if not (repo_path / "symmlearn" / "lattice" / "_pg_image.py").exists():
        raise FileNotFoundError(f"not a symmetry-learn repository: {repo_path}")
    repo_text = str(repo_path)
    if repo_text not in sys.path:
        sys.path.insert(0, repo_text)
    from symmlearn.lattice import PlaneGroup, WyckoffStructure

    return PlaneGroup, WyckoffStructure


def _generator_provenance(repo_path: str | Path) -> dict:
    repo = Path(repo_path).resolve()
    result = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    status = subprocess.run(
        ["git", "-C", str(repo), "status", "--porcelain"],
        check=False,
        capture_output=True,
        text=True,
    )
    source_files = [
        repo / "symmlearn/lattice/_pg_image.py",
        repo / "symmlearn/lattice/_plane_group.py",
        repo / "symmlearn/lattice/_wyckoff_structure.py",
        repo / "symmlearn/lattice/_mixin_plane_group.py",
    ]
    return {
        "generator_git_commit": result.stdout.strip() if result.returncode == 0 else "unknown",
        "generator_worktree_dirty": bool(status.stdout.strip()) if status.returncode == 0 else None,
        "generator_source_sha256": {
            str(path.relative_to(repo)).replace("\\", "/"): sha256_file(path)
            for path in source_files
        },
    }


def generate_image_only(
    record: SymmLearnSeed,
    *,
    image_size: int = DEFAULT_IMAGE_SIZE,
    repo_path: str | Path,
) -> tuple[np.ndarray, dict]:
    """Generate one large clean image without computing any symmetry map."""

    ImageContract(image_size=image_size).validate()
    PlaneGroup, WyckoffStructure = _import_generator(repo_path)
    rng = np.random.RandomState(record.seed)
    structure = WyckoffStructure(
        pg_num=record.pg_number,
        structure_letters=record.structure,
    )
    structure_dict = structure.to_structure_dict(seed=rng)
    plane_group = PlaneGroup(plane_group=record.pg_number)
    lattice = plane_group.generate_lattice(
        structure_dict,
        size=image_size,
        seed=rng,
        max_samples=5,
        angle_deg=0,
        sigma_method="min",
        verbose=False,
    )
    pg_image = lattice.get_image(image_size=image_size, sigma_map=None, seed=rng)
    image = validate_2d_image(pg_image.img)
    if pg_image.has_symm_maps or any(
        value is not None
        for value in (pg_image.rot_maps, pg_image.ref_map, pg_image.theta_map)
    ):
        raise RuntimeError("image-only path unexpectedly computed symmetry maps")
    metadata = {
        **asdict(record),
        "source_id": "symmlearn",
        "generator": "symmetry-learn:PGLattice.get_image",
        "generator_repo": str(Path(repo_path).resolve()),
        **_generator_provenance(repo_path),
        "output_image_size": image_size,
        "generated_channels": ["image"],
        "symmetry_maps_computed": False,
        "resolved_pg_number": int(lattice.pg_number),
        "angle_deg": float(lattice.angle_deg),
        "sigma_map": float(lattice.sigma_map),
        "amplitude_map": {key: float(value) for key, value in lattice.amplitude_map.items()},
        "structure_dict": structure_dict,
    }
    return image, metadata
