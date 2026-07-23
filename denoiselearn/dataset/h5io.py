"""HDF5 storage for versioned single-channel clean images."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import h5py
import numpy as np

from .contracts import (
    INT16_SCALE,
    SCHEMA_VERSION,
    dequantize_int16,
    normalize_clean_image,
    quantize_int16,
    validate_2d_image,
)


def write_clean_h5(
    path: str | Path,
    images: Iterable[np.ndarray],
    metadata: Iterable[Mapping[str, Any]],
    *,
    generator: str,
    source_registry_checksum: str = "",
    compression: str = "gzip",
    compression_level: int = 4,
) -> Path:
    """Write clean images as ``[N, 1, H, W]`` int16 plus JSON provenance."""

    image_list = list(images)
    metadata_list = [dict(item) for item in metadata]
    if not image_list:
        raise ValueError("at least one image is required")
    if len(image_list) != len(metadata_list):
        raise ValueError("images and metadata must have identical lengths")

    normalized_images: list[np.ndarray] = []
    enriched_metadata: list[dict[str, Any]] = []
    shape = np.asarray(image_list[0]).shape
    for index, (image, item) in enumerate(zip(image_list, metadata_list)):
        normalized, range_metadata = normalize_clean_image(image)
        if normalized.shape != shape:
            raise ValueError(f"image {index} has shape {normalized.shape}, expected {shape}")
        normalized_images.append(quantize_int16(normalized))
        enriched_metadata.append({**item, **range_metadata, "index": index})

    stack = np.stack(normalized_images, axis=0)[:, None, :, :]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    string_dtype = h5py.string_dtype(encoding="utf-8")
    compression_opts = compression_level if compression == "gzip" else None
    with h5py.File(path, "w") as handle:
        dataset = handle.create_dataset(
            "images",
            data=stack,
            dtype=np.int16,
            chunks=(1, 1, shape[0], shape[1]),
            compression=compression,
            compression_opts=compression_opts,
            shuffle=True,
            fletcher32=True,
        )
        metadata_group = handle.create_group("metadata")
        metadata_group.create_dataset(
            "json",
            data=np.asarray(
                [json.dumps(item, ensure_ascii=False, sort_keys=True) for item in enriched_metadata],
                dtype=object,
            ),
            dtype=string_dtype,
        )
        handle.attrs.update(
            {
                "schema_version": SCHEMA_VERSION,
                "generator": generator,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "image_layout": "NCHW",
                "channels": 1,
                "height": shape[0],
                "width": shape[1],
                "stored_dtype": "int16",
                "stored_min": 0,
                "stored_max": int(INT16_SCALE),
                "dequantize_scale": 1.0 / float(INT16_SCALE),
                "dequantize_offset": 0.0,
                "normalization": "per-image min-max before quantization",
                "source_registry_checksum": source_registry_checksum,
            }
        )
        dataset.attrs["semantic_channel"] = "clean_image_only"
    return path


def read_clean_h5(path: str | Path) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    """Read and dequantize a clean-image HDF5 file."""

    with h5py.File(path, "r") as handle:
        stored = handle["images"][:]
        if stored.ndim != 4 or stored.shape[1] != 1:
            raise ValueError(f"expected [N,1,H,W], got {stored.shape}")
        images = dequantize_int16(stored[:, 0])
        metadata = [json.loads(value) for value in handle["metadata/json"].asstr()[:]]
        attrs = {key: value.item() if hasattr(value, "item") else value for key, value in handle.attrs.items()}
    return images, metadata, attrs


def write_defect_h5(
    path: str | Path,
    base_images: Iterable[np.ndarray],
    defect_images: Iterable[np.ndarray],
    masks: Iterable[Mapping[str, np.ndarray]],
    metadata: Iterable[Mapping[str, Any]],
    *,
    generator: str = "column-mask-defects-v3",
    compression: str = "gzip",
    compression_level: int = 4,
) -> Path:
    """Write paired base/defect images and aligned uint8 masks.

    Each pair uses one shared affine normalization before int16 quantization, so
    weak/strong/vacancy intensity relations are not changed independently.
    """

    bases = list(base_images)
    defects = list(defect_images)
    mask_items = [{key: np.asarray(value) for key, value in item.items()} for item in masks]
    metadata_items = [dict(item) for item in metadata]
    count = len(bases)
    if not count:
        raise ValueError("at least one base/defect pair is required")
    if not (len(defects) == len(mask_items) == len(metadata_items) == count):
        raise ValueError("base_images, defect_images, masks and metadata must have identical lengths")
    mask_names = tuple(sorted(mask_items[0]))
    if not mask_names:
        raise ValueError("at least one mask channel is required")

    shape = validate_2d_image(bases[0]).shape
    stored_bases: list[np.ndarray] = []
    stored_defects: list[np.ndarray] = []
    stored_masks: dict[str, list[np.ndarray]] = {name: [] for name in mask_names}
    enriched_metadata: list[dict[str, Any]] = []
    for index, (base, defect, item_masks, item_metadata) in enumerate(
        zip(bases, defects, mask_items, metadata_items)
    ):
        base_array = validate_2d_image(base)
        defect_array = validate_2d_image(defect)
        if base_array.shape != shape or defect_array.shape != shape:
            raise ValueError(f"pair {index} does not match expected shape {shape}")
        if tuple(sorted(item_masks)) != mask_names:
            raise ValueError(f"pair {index} has inconsistent mask channels")
        pair_min = float(min(base_array.min(), defect_array.min()))
        pair_max = float(max(base_array.max(), defect_array.max()))
        if pair_max <= pair_min:
            raise ValueError(f"pair {index} is constant")
        scale = pair_max - pair_min
        stored_bases.append(quantize_int16((base_array - pair_min) / scale))
        stored_defects.append(quantize_int16((defect_array - pair_min) / scale))
        for name in mask_names:
            mask = np.asarray(item_masks[name])
            if mask.shape != shape:
                raise ValueError(f"mask {name!r} in pair {index} has shape {mask.shape}, expected {shape}")
            if not np.isin(mask, (0, 1)).all():
                raise ValueError(f"mask {name!r} in pair {index} is not binary")
            stored_masks[name].append(mask.astype(np.uint8))
        enriched_metadata.append(
            {
                **item_metadata,
                "index": index,
                "pair_normalization_min": pair_min,
                "pair_normalization_max": pair_max,
                "pair_dequantize_scale": scale / float(INT16_SCALE),
                "pair_dequantize_offset": pair_min,
            }
        )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    compression_opts = compression_level if compression == "gzip" else None
    string_dtype = h5py.string_dtype(encoding="utf-8")
    with h5py.File(path, "w") as handle:
        for name, data, semantic in (
            ("base_images", stored_bases, "perfect_or_base_clean_image"),
            ("images", stored_defects, "clean_image_with_controlled_defects"),
        ):
            dataset = handle.create_dataset(
                name,
                data=np.stack(data, axis=0)[:, None, :, :],
                dtype=np.int16,
                chunks=(1, 1, shape[0], shape[1]),
                compression=compression,
                compression_opts=compression_opts,
                shuffle=True,
                fletcher32=True,
            )
            dataset.attrs["semantic_channel"] = semantic
        mask_group = handle.create_group("masks")
        for name in mask_names:
            mask_group.create_dataset(
                name,
                data=np.stack(stored_masks[name], axis=0)[:, None, :, :],
                dtype=np.uint8,
                chunks=(1, 1, shape[0], shape[1]),
                compression=compression,
                compression_opts=compression_opts,
                shuffle=True,
                fletcher32=True,
            )
        metadata_group = handle.create_group("metadata")
        metadata_group.create_dataset(
            "json",
            data=np.asarray(
                [json.dumps(item, ensure_ascii=False, sort_keys=True) for item in enriched_metadata],
                dtype=object,
            ),
            dtype=string_dtype,
        )
        handle.attrs.update(
            {
                "schema_version": "defect-image-h5-v1",
                "generator": generator,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "image_layout": "NCHW",
                "channels": 1,
                "height": shape[0],
                "width": shape[1],
                "stored_dtype": "int16",
                "mask_dtype": "uint8",
                "stored_min": 0,
                "stored_max": int(INT16_SCALE),
                "normalization": "shared affine normalization per base/defect pair",
            }
        )
    return path


def read_defect_h5(
    path: str | Path,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], list[dict[str, Any]], dict[str, Any]]:
    """Read paired defect data and restore each pair to its recorded source range."""

    with h5py.File(path, "r") as handle:
        stored_base = handle["base_images"][:, 0]
        stored_defect = handle["images"][:, 0]
        metadata = [json.loads(value) for value in handle["metadata/json"].asstr()[:]]
        bases: list[np.ndarray] = []
        defects: list[np.ndarray] = []
        for index, item in enumerate(metadata):
            scale = float(item["pair_dequantize_scale"])
            offset = float(item["pair_dequantize_offset"])
            bases.append(stored_base[index].astype(np.float32) * scale + offset)
            defects.append(stored_defect[index].astype(np.float32) * scale + offset)
        masks = {name: handle[f"masks/{name}"][:, 0].astype(np.uint8) for name in handle["masks"]}
        attrs = {key: value.item() if hasattr(value, "item") else value for key, value in handle.attrs.items()}
    return np.stack(bases), np.stack(defects), masks, metadata, attrs
