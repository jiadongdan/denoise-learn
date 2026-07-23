"""Versioned clean-master assembly with structure-level split isolation."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import h5py
import numpy as np


SPLIT_NAMES = ("train", "valid", "test")
VARIANT_TYPES = ("perfect", "defect_only", "vacuum_only", "defect_and_vacuum")


@dataclass(frozen=True)
class SplitConfig:
    """Deterministic group-aware split settings."""

    train_fraction: float = 0.8
    valid_fraction: float = 0.1
    test_fraction: float = 0.1
    seed: int = 20260722
    source_balance_weight: float = 0.25
    variant_balance_weight: float = 0.25

    @property
    def fractions(self) -> dict[str, float]:
        return {
            "train": self.train_fraction,
            "valid": self.valid_fraction,
            "test": self.test_fraction,
        }

    def validate(self) -> None:
        values = tuple(self.fractions.values())
        if any(value <= 0.0 or value >= 1.0 for value in values):
            raise ValueError("all split fractions must be in (0, 1)")
        if not np.isclose(sum(values), 1.0):
            raise ValueError("split fractions must sum to 1")
        if self.source_balance_weight < 0 or self.variant_balance_weight < 0:
            raise ValueError("balance weights must be non-negative")


@dataclass(frozen=True)
class CleanMasterRecord:
    """One already-generated clean image and its structure identity."""

    image_id: str
    base_structure_id: str
    split_group_id: str
    source_id: str
    variant_type: str
    image: np.ndarray
    metadata: Mapping[str, Any] = field(default_factory=dict)
    masks: Mapping[str, np.ndarray] | None = None

    def validate(self) -> None:
        for name in ("image_id", "base_structure_id", "split_group_id", "source_id"):
            if not getattr(self, name):
                raise ValueError(f"{name} must be non-empty")
        if self.variant_type not in VARIANT_TYPES:
            raise ValueError(f"unsupported variant_type: {self.variant_type}")
        image = np.asarray(self.image)
        if image.ndim != 2:
            raise ValueError(f"image {self.image_id!r} must be 2-D")
        if image.dtype != np.int16:
            raise ValueError(f"image {self.image_id!r} must already use int16 storage")
        if image.size == 0 or int(image.min()) < 0:
            raise ValueError(f"image {self.image_id!r} must be non-empty and non-negative")


@dataclass(frozen=True)
class SplitPlan:
    """Immutable image-to-split assignment and within-split write order."""

    assignments: Mapping[str, str]
    ordered_image_ids: Mapping[str, tuple[str, ...]]
    config: SplitConfig
    statistics: Mapping[str, Any]


def _validated_records(records: Iterable[CleanMasterRecord]) -> list[CleanMasterRecord]:
    items = list(records)
    if not items:
        raise ValueError("at least one clean record is required")
    seen: set[str] = set()
    shape = np.asarray(items[0].image).shape
    for item in items:
        item.validate()
        if item.image_id in seen:
            raise ValueError(f"duplicate image_id: {item.image_id}")
        seen.add(item.image_id)
        if np.asarray(item.image).shape != shape:
            raise ValueError("all clean-master images must share one shape")
    return items


def _category_counts(records: Iterable[CleanMasterRecord], attribute: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        value = str(getattr(record, attribute))
        counts[value] = counts.get(value, 0) + 1
    return counts


def build_structure_split(
    records: Iterable[CleanMasterRecord],
    config: SplitConfig = SplitConfig(),
) -> SplitPlan:
    """Shuffle indivisible structure groups and balance counts across splits.

    Masks are deliberately not inspected, so enabling or disabling mask storage
    cannot change any assignment.
    """

    config.validate()
    items = _validated_records(records)
    groups: dict[str, list[CleanMasterRecord]] = {}
    for item in items:
        groups.setdefault(item.split_group_id, []).append(item)

    rng = np.random.default_rng(config.seed)
    variant_group_ids = sorted(
        group_id
        for group_id, group in groups.items()
        if any(item.variant_type != "perfect" for item in group)
    )
    perfect_group_ids = sorted(set(groups) - set(variant_group_ids))
    rng.shuffle(variant_group_ids)
    rng.shuffle(perfect_group_ids)

    total_count = len(items)
    total_sources = _category_counts(items, "source_id")
    total_variants = _category_counts(items, "variant_type")
    assigned: dict[str, list[CleanMasterRecord]] = {name: [] for name in SPLIT_NAMES}

    def objective(candidate_split: str, group: list[CleanMasterRecord]) -> float:
        projected = {
            split: assigned[split] + (group if split == candidate_split else [])
            for split in SPLIT_NAMES
        }
        score = 0.0
        for split in SPLIT_NAMES:
            fraction = config.fractions[split]
            target_total = max(total_count * fraction, 1.0)
            score += ((len(projected[split]) - target_total) / target_total) ** 2
            projected_sources = _category_counts(projected[split], "source_id")
            for category, total in total_sources.items():
                target = max(total * fraction, 1.0)
                score += config.source_balance_weight * (
                    (projected_sources.get(category, 0) - target) / target
                ) ** 2
            projected_variants = _category_counts(projected[split], "variant_type")
            for category, total in total_variants.items():
                target = max(total * fraction, 1.0)
                score += config.variant_balance_weight * (
                    (projected_variants.get(category, 0) - target) / target
                ) ** 2
        return score

    def allocate_count_quotas(count: int, *, require_each: bool) -> dict[str, int]:
        raw = {split: count * config.fractions[split] for split in SPLIT_NAMES}
        quotas = {split: int(np.floor(raw[split])) for split in SPLIT_NAMES}
        for split in sorted(SPLIT_NAMES, key=lambda name: raw[name] - quotas[name], reverse=True):
            if sum(quotas.values()) >= count:
                break
            quotas[split] += 1
        if require_each and count >= len(SPLIT_NAMES):
            missing = [split for split in SPLIT_NAMES if quotas[split] == 0]
            for missing_split in missing:
                donor = max(SPLIT_NAMES, key=lambda name: quotas[name])
                quotas[donor] -= 1
                quotas[missing_split] += 1
        return quotas

    target_image_counts = allocate_count_quotas(total_count, require_each=True)
    variant_quotas = allocate_count_quotas(len(variant_group_ids), require_each=True)
    assigned_variant_groups = {split: 0 for split in SPLIT_NAMES}
    for group_id in variant_group_ids:
        group = groups[group_id]
        eligible = [
            split
            for split in SPLIT_NAMES
            if assigned_variant_groups[split] < variant_quotas[split]
        ]
        scores = np.asarray([objective(split, group) for split in eligible], dtype=np.float64)
        minimum = float(scores.min())
        candidates = [
            split for split, score in zip(eligible, scores) if np.isclose(score, minimum)
        ]
        chosen = candidates[int(rng.integers(0, len(candidates)))]
        assigned[chosen].extend(group)
        assigned_variant_groups[chosen] += 1

    for group_id in perfect_group_ids:
        group = groups[group_id]
        eligible = [
            split
            for split in SPLIT_NAMES
            if len(assigned[split]) + len(group) <= target_image_counts[split]
        ]
        if not eligible:
            eligible = list(SPLIT_NAMES)
        scores = np.asarray([objective(split, group) for split in eligible], dtype=np.float64)
        minimum = float(scores.min())
        candidates = [
            split for split, score in zip(eligible, scores) if np.isclose(score, minimum)
        ]
        chosen = candidates[int(rng.integers(0, len(candidates)))]
        assigned[chosen].extend(group)

    assignments: dict[str, str] = {}
    ordered_image_ids: dict[str, tuple[str, ...]] = {}
    for split in SPLIT_NAMES:
        split_items = assigned[split]
        order = np.arange(len(split_items))
        rng.shuffle(order)
        ordered = tuple(split_items[index].image_id for index in order)
        ordered_image_ids[split] = ordered
        for image_id in ordered:
            assignments[image_id] = split

    plan = SplitPlan(
        assignments=assignments,
        ordered_image_ids=ordered_image_ids,
        config=config,
        statistics=_split_statistics(items, assignments),
    )
    validate_no_split_leakage(items, plan)
    return plan


def _split_statistics(
    records: Iterable[CleanMasterRecord], assignments: Mapping[str, str]
) -> dict[str, Any]:
    stats: dict[str, Any] = {}
    items = list(records)
    for split in SPLIT_NAMES:
        selected = [item for item in items if assignments[item.image_id] == split]
        stats[split] = {
            "count": len(selected),
            "source_counts": _category_counts(selected, "source_id"),
            "variant_counts": _category_counts(selected, "variant_type"),
            "group_count": len({item.split_group_id for item in selected}),
        }
    return stats


def validate_no_split_leakage(records: Iterable[CleanMasterRecord], plan: SplitPlan) -> None:
    """Reject missing assignments or any structure group spanning splits."""

    items = list(records)
    image_ids = {item.image_id for item in items}
    if set(plan.assignments) != image_ids:
        raise ValueError("split plan must assign every image exactly once")
    group_splits: dict[str, set[str]] = {}
    for item in items:
        split = plan.assignments[item.image_id]
        if split not in SPLIT_NAMES:
            raise ValueError(f"unsupported split: {split}")
        group_splits.setdefault(item.split_group_id, set()).add(split)
    leaked = {group: splits for group, splits in group_splits.items() if len(splits) != 1}
    if leaked:
        raise ValueError(f"structure-level split leakage detected: {leaked}")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_clean_master_bundle(
    path: str | Path,
    records: Iterable[CleanMasterRecord],
    plan: SplitPlan,
    *,
    dataset_version: str = "clean_v1",
    include_masks: bool = False,
    compression: str = "gzip",
    compression_level: int = 4,
) -> dict[str, Path | str]:
    """Write one split-keyed HDF5 plus JSONL manifest and checksum sidecar."""

    items = _validated_records(records)
    validate_no_split_leakage(items, plan)
    by_id = {item.image_id: item for item in items}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = path.with_suffix(".manifest.jsonl")
    checksums_path = path.with_suffix(".checksums.json")
    compression_opts = compression_level if compression == "gzip" else None
    string_dtype = h5py.string_dtype(encoding="utf-8")
    shape = np.asarray(items[0].image).shape
    manifest_rows: list[dict[str, Any]] = []
    mask_names = sorted(
        {name for item in items for name in ((item.masks or {}).keys())}
    ) if include_masks else []

    with h5py.File(path, "w") as handle:
        handle.attrs.update(
            {
                "schema_version": "atomic-stem-denoise-h5-v1",
                "dataset_version": dataset_version,
                "split_seed": int(plan.config.seed),
                "split_fractions_json": json.dumps(plan.config.fractions, sort_keys=True),
                "include_masks": bool(include_masks),
                "stored_dtype": "int16",
                "image_layout": "NCHW",
                "dequantize_scale": 1.0 / 32767.0,
                "dequantize_offset": 0.0,
            }
        )
        for split in SPLIT_NAMES:
            split_group = handle.create_group(split)
            ordered = plan.ordered_image_ids[split]
            stack = np.stack([np.asarray(by_id[image_id].image) for image_id in ordered])[:, None]
            images_dataset = split_group.create_dataset(
                "images",
                data=stack,
                dtype=np.int16,
                chunks=(1, 1, shape[0], shape[1]),
                compression=compression,
                compression_opts=compression_opts,
                shuffle=True,
                fletcher32=True,
            )
            images_dataset.attrs["semantic_channel"] = "clean_image"
            metadata_group = split_group.create_group("metadata")
            json_rows: list[str] = []
            for index, image_id in enumerate(ordered):
                item = by_id[image_id]
                row = {
                    **dict(item.metadata),
                    "dataset_version": dataset_version,
                    "image_id": item.image_id,
                    "base_structure_id": item.base_structure_id,
                    "split_group_id": item.split_group_id,
                    "source_id": item.source_id,
                    "variant_type": item.variant_type,
                    "split": split,
                    "h5_key": f"/{split}/images",
                    "h5_index": index,
                    "height": shape[0],
                    "width": shape[1],
                    "dtype": "int16",
                }
                manifest_rows.append(row)
                json_rows.append(json.dumps(row, ensure_ascii=False, sort_keys=True))
            metadata_group.create_dataset(
                "json", data=np.asarray(json_rows, dtype=object), dtype=string_dtype
            )
            if include_masks and mask_names:
                masks_group = split_group.create_group("masks")
                for name in mask_names:
                    mask_stack: list[np.ndarray] = []
                    for image_id in ordered:
                        mask = np.asarray((by_id[image_id].masks or {}).get(name, np.zeros(shape)))
                        if mask.shape != shape or not np.isin(mask, (0, 1)).all():
                            raise ValueError(f"invalid binary mask {name!r} for {image_id!r}")
                        mask_stack.append(mask.astype(np.uint8))
                    masks_group.create_dataset(
                        name,
                        data=np.stack(mask_stack)[:, None],
                        dtype=np.uint8,
                        chunks=(1, 1, shape[0], shape[1]),
                        compression=compression,
                        compression_opts=compression_opts,
                        shuffle=True,
                        fletcher32=True,
                    )

    manifest_text = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in manifest_rows
    )
    manifest_path.write_text(manifest_text, encoding="utf-8")
    h5_checksum = sha256_file(path)
    manifest_checksum = sha256_file(manifest_path)
    checksums = {
        "dataset_version": dataset_version,
        "h5": {"path": path.name, "sha256": h5_checksum},
        "manifest": {"path": manifest_path.name, "sha256": manifest_checksum},
    }
    checksums_path.write_text(
        json.dumps(checksums, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "h5": path,
        "manifest": manifest_path,
        "checksums": checksums_path,
        "h5_sha256": h5_checksum,
        "manifest_sha256": manifest_checksum,
    }
