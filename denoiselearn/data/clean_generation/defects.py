"""Seeded point-defect and vacuum variants for clean S/TEM images.

Raster sources are partitioned into non-overlapping atomic-column masks. Selected
column masks are then set to background (vacancy/vacuum) or scaled by one seeded
factor per substitution category. Twisted bilayers use structure-aware per-layer
column deletion before rotation and overlap.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Literal

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter, maximum_filter
from scipy.spatial import cKDTree

from .contracts import validate_2d_image


SourceFamily = Literal["symmlearn", "multislice", "quasi"]
VacuumShape = Literal["random", "polygon", "double_ellipse"]
VariantMode = Literal["defect_only", "vacuum_only", "defect_and_vacuum"]


@dataclass(frozen=True)
class DefectVariantMix:
    """Variant counts expressed as fractions of the perfect/base clean count."""

    defect_only_fraction_of_base: float = 0.20
    vacuum_only_fraction_of_base: float = 0.05
    defect_and_vacuum_fraction_of_base: float = 0.05

    def validate(self) -> None:
        values = (
            self.defect_only_fraction_of_base,
            self.vacuum_only_fraction_of_base,
            self.defect_and_vacuum_fraction_of_base,
        )
        if any(value < 0.0 for value in values) or sum(values) > 1.0:
            raise ValueError("variant fractions must be non-negative and sum to at most 1")

    @property
    def total_fraction_of_base(self) -> float:
        return (
            self.defect_only_fraction_of_base
            + self.vacuum_only_fraction_of_base
            + self.defect_and_vacuum_fraction_of_base
        )


DEFAULT_VARIANT_MIX = DefectVariantMix()


@dataclass(frozen=True)
class DefectConfig:
    """Configuration for one deterministic defect/vacuum variant.

    The numeric defaults are smoke-review defaults, not frozen production
    distributions.  Every range is serialized so later datasets can change the
    defaults without losing provenance.
    """

    seed: int
    source_family: SourceFamily = "symmlearn"
    variant_mode: VariantMode = "defect_and_vacuum"
    point_defect_fraction_range: tuple[float, float] = (0.005, 0.050)
    point_defect_category_weights: tuple[float, float, float] = (1.0, 1.0, 1.0)
    weak_factor_range: tuple[float, float] = (0.35, 0.75)
    strong_factor_range: tuple[float, float] = (1.25, 1.80)
    vacuum_area_fraction_range: tuple[float, float] = (0.05, 0.20)
    vacuum_shape: VacuumShape = "random"
    vacuum_region_count_range: tuple[int, int] = (1, 2)
    boundary_connected_probability: float = 0.5
    enable_vacancy: bool = True
    enable_weak: bool = True
    enable_strong: bool = True
    enable_vacuum: bool = True
    detection_min_distance_pixels: int = 3
    detection_threshold_rel: float = 0.12
    detection_smoothing_sigma: float = 1.0
    column_sigma_fraction_of_spacing: float = 0.22
    min_column_sigma_pixels: float = 1.25
    max_column_sigma_pixels: float = 6.0
    column_mask_threshold_rel: float = 0.02
    max_detected_columns: int = 20_000

    def validate(self) -> None:
        if self.source_family not in {"symmlearn", "multislice", "quasi"}:
            raise ValueError(f"unsupported source_family: {self.source_family}")
        for name in (
            "point_defect_fraction_range",
            "vacuum_area_fraction_range",
        ):
            low, high = getattr(self, name)
            if not (0.0 <= low <= high <= 1.0):
                raise ValueError(f"{name} must lie within [0, 1]")
        weak_low, weak_high = self.weak_factor_range
        strong_low, strong_high = self.strong_factor_range
        if not (0.0 < weak_low <= weak_high < 1.0):
            raise ValueError("weak_factor_range must be positive and below 1")
        if not (1.0 < strong_low <= strong_high):
            raise ValueError("strong_factor_range must be above 1")
        if not (0.0 <= self.boundary_connected_probability <= 1.0):
            raise ValueError("boundary_connected_probability must lie within [0, 1]")
        if self.vacuum_shape not in {"random", "polygon", "double_ellipse"}:
            raise ValueError(f"unsupported vacuum_shape: {self.vacuum_shape}")
        if self.variant_mode not in {"defect_only", "vacuum_only", "defect_and_vacuum"}:
            raise ValueError(f"unsupported variant_mode: {self.variant_mode}")
        if len(self.point_defect_category_weights) != 3 or any(
            value < 0 for value in self.point_defect_category_weights
        ) or sum(self.point_defect_category_weights) <= 0:
            raise ValueError("point_defect_category_weights must contain three non-negative values")
        low_count, high_count = self.vacuum_region_count_range
        if low_count < 1 or high_count < low_count:
            raise ValueError("vacuum_region_count_range must contain positive integers")
        if self.detection_min_distance_pixels < 1:
            raise ValueError("detection_min_distance_pixels must be positive")
        if not (0.0 < self.detection_threshold_rel < 1.0):
            raise ValueError("detection_threshold_rel must lie within (0, 1)")
        if not (0.0 < self.column_mask_threshold_rel < 1.0):
            raise ValueError("column_mask_threshold_rel must lie within (0, 1)")


@dataclass(frozen=True)
class DetectedColumns:
    coordinates_yx: np.ndarray
    peak_intensities: np.ndarray
    background: float
    threshold: float
    median_nearest_spacing_pixels: float
    estimated_sigma_pixels: float


@dataclass(frozen=True)
class DefectResult:
    image: np.ndarray
    masks: dict[str, np.ndarray]
    metadata: dict


@dataclass(frozen=True)
class ColumnPartition:
    """Non-overlapping raster assignment of foreground pixels to column IDs."""

    labels: np.ndarray
    foreground: np.ndarray
    background: float
    threshold: float


def build_defect_variant_modes(
    base_clean_count: int,
    *,
    seed: int,
    mix: DefectVariantMix = DEFAULT_VARIANT_MIX,
) -> list[VariantMode]:
    """Build a shuffled 20%/5%/5% variant plan relative to base clean images."""

    if base_clean_count < 0:
        raise ValueError("base_clean_count must be non-negative")
    mix.validate()
    counts = {
        "defect_only": int(round(base_clean_count * mix.defect_only_fraction_of_base)),
        "vacuum_only": int(round(base_clean_count * mix.vacuum_only_fraction_of_base)),
        "defect_and_vacuum": int(
            round(base_clean_count * mix.defect_and_vacuum_fraction_of_base)
        ),
    }
    modes: list[VariantMode] = []
    for mode, count in counts.items():
        modes.extend([mode] * count)
    rng = np.random.default_rng(seed)
    rng.shuffle(modes)
    return modes


def detect_atomic_columns(image: np.ndarray, config: DefectConfig) -> DetectedColumns:
    """Detect reproducible local maxima and estimate their common pixel scale."""

    config.validate()
    array = validate_2d_image(image)
    smooth = gaussian_filter(array, sigma=config.detection_smoothing_sigma)
    background = float(np.percentile(smooth, 5.0))
    high = float(np.percentile(smooth, 99.9))
    threshold = background + config.detection_threshold_rel * (high - background)
    size = 2 * config.detection_min_distance_pixels + 1
    local_max = smooth == maximum_filter(smooth, size=size, mode="nearest")
    margin = max(config.detection_min_distance_pixels, 2)
    eligible = local_max & (smooth >= threshold)
    eligible[:margin, :] = False
    eligible[-margin:, :] = False
    eligible[:, :margin] = False
    eligible[:, -margin:] = False
    coordinates = np.argwhere(eligible)
    if not len(coordinates):
        raise ValueError("no atomic-column peaks detected; review threshold/min-distance settings")
    intensities = smooth[coordinates[:, 0], coordinates[:, 1]]
    order = np.argsort(intensities)[::-1][: config.max_detected_columns]
    coordinates = coordinates[order].astype(np.int32)
    intensities = intensities[order].astype(np.float32)

    if len(coordinates) >= 2:
        distances, _ = cKDTree(coordinates).query(coordinates, k=2)
        nearest = distances[:, 1]
        median_spacing = float(np.median(nearest[np.isfinite(nearest)]))
    else:
        median_spacing = float(2 * config.detection_min_distance_pixels + 1)
    sigma = float(
        np.clip(
            median_spacing * config.column_sigma_fraction_of_spacing,
            config.min_column_sigma_pixels,
            config.max_column_sigma_pixels,
        )
    )
    return DetectedColumns(
        coordinates_yx=coordinates,
        peak_intensities=intensities,
        background=background,
        threshold=threshold,
        median_nearest_spacing_pixels=median_spacing,
        estimated_sigma_pixels=sigma,
    )


def build_column_partition(
    image: np.ndarray,
    detection: DetectedColumns,
    config: DefectConfig,
) -> ColumnPartition:
    """Assign every foreground pixel to its nearest detected atomic column.

    Label 0 is background. Labels 1..N correspond to row indices in
    ``detection.coordinates_yx``. This Voronoi-style partition prevents masks for
    neighboring or overlapping raster columns from sharing pixels.
    """

    base = validate_2d_image(image).astype(np.float32, copy=False)
    background = float(base.min())
    high = float(np.percentile(base, 99.9))
    threshold = background + config.column_mask_threshold_rel * (high - background)
    foreground = base > threshold
    markers = np.zeros(base.shape, dtype=np.int32)
    for column_id, (y, x) in enumerate(detection.coordinates_yx, start=1):
        markers[int(y), int(x)] = column_id
    _, nearest_indices = distance_transform_edt(markers == 0, return_indices=True)
    labels = markers[tuple(nearest_indices)]
    labels[~foreground] = 0
    return ColumnPartition(
        labels=labels.astype(np.int32, copy=False),
        foreground=foreground,
        background=background,
        threshold=float(threshold),
    )
def _ellipse_mask(
    shape: tuple[int, int],
    *,
    center_yx: tuple[float, float],
    radii_yx: tuple[float, float],
) -> np.ndarray:
    y, x = np.ogrid[: shape[0], : shape[1]]
    cy, cx = center_yx
    ry, rx = radii_yx
    return ((y - cy) / max(ry, 1.0)) ** 2 + ((x - cx) / max(rx, 1.0)) ** 2 <= 1.0


def _polygon_mask(shape: tuple[int, int], vertices_yx: np.ndarray) -> np.ndarray:
    """Vectorized even/odd point-in-polygon test inside the vertex bounding box."""

    h, w = shape
    y0 = max(0, int(np.floor(vertices_yx[:, 0].min())))
    y1 = min(h, int(np.ceil(vertices_yx[:, 0].max())) + 1)
    x0 = max(0, int(np.floor(vertices_yx[:, 1].min())))
    x1 = min(w, int(np.ceil(vertices_yx[:, 1].max())) + 1)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    inside = np.zeros_like(yy, dtype=bool)
    py = vertices_yx[:, 0]
    px = vertices_yx[:, 1]
    j = len(vertices_yx) - 1
    for i in range(len(vertices_yx)):
        crosses = (py[i] > yy) != (py[j] > yy)
        x_cross = (px[j] - px[i]) * (yy - py[i]) / (py[j] - py[i] + 1e-12) + px[i]
        inside ^= crosses & (xx < x_cross)
        j = i
    mask = np.zeros(shape, dtype=bool)
    mask[y0:y1, x0:x1] = inside
    return mask


def _sample_center(
    rng: np.random.Generator,
    shape: tuple[int, int],
    radii_yx: tuple[float, float],
    boundary_connected: bool,
) -> tuple[float, float]:
    h, w = shape
    ry, rx = radii_yx
    if not boundary_connected:
        return (
            float(rng.uniform(min(ry, h / 3), max(min(ry, h / 3) + 1, h - min(ry, h / 3)))),
            float(rng.uniform(min(rx, w / 3), max(min(rx, w / 3) + 1, w - min(rx, w / 3)))),
        )
    edge = int(rng.integers(0, 4))
    if edge == 0:
        return float(rng.uniform(0, h)), float(rng.uniform(-0.45 * rx, 0.45 * rx))
    if edge == 1:
        return float(rng.uniform(0, h)), float(w - 1 + rng.uniform(-0.45 * rx, 0.45 * rx))
    if edge == 2:
        return float(rng.uniform(-0.45 * ry, 0.45 * ry)), float(rng.uniform(0, w))
    return float(h - 1 + rng.uniform(-0.45 * ry, 0.45 * ry)), float(rng.uniform(0, w))


def generate_vacuum_mask(
    shape: tuple[int, int], config: DefectConfig, rng: np.random.Generator
) -> tuple[np.ndarray, dict]:
    """Generate polygon or double-ellipse regions with an exact visible target area."""

    target_fraction = float(rng.uniform(*config.vacuum_area_fraction_range))
    boundary_connected = bool(rng.random() < config.boundary_connected_probability)
    region_count = int(rng.integers(config.vacuum_region_count_range[0], config.vacuum_region_count_range[1] + 1))
    shape_name = config.vacuum_shape
    if shape_name == "random":
        shape_name = str(rng.choice(["polygon", "double_ellipse"]))
    if shape_name == "double_ellipse":
        region_count = 2

    mask = np.zeros(shape, dtype=bool)
    records: list[dict] = []
    area_each = target_fraction * shape[0] * shape[1] / region_count
    for region_index in range(region_count):
        aspect = float(np.exp(rng.uniform(np.log(0.55), np.log(1.8))))
        ry = float(np.sqrt(area_each / (np.pi * aspect)))
        rx = float(ry * aspect)
        ry = min(ry, 0.46 * shape[0])
        rx = min(rx, 0.46 * shape[1])
        connects = boundary_connected and region_index == 0
        center = _sample_center(rng, shape, (ry, rx), connects)
        if shape_name == "polygon":
            count = int(rng.integers(6, 10))
            angles = np.sort(rng.uniform(0.0, 2.0 * np.pi, size=count))
            scales = rng.uniform(0.68, 1.18, size=count)
            vertices = np.column_stack(
                [center[0] + ry * scales * np.sin(angles), center[1] + rx * scales * np.cos(angles)]
            )
            region = _polygon_mask(shape, vertices)
            record = {
                "kind": "polygon",
                "center_yx": [float(center[0]), float(center[1])],
                "vertices_yx": vertices.tolist(),
                "boundary_connected_requested": connects,
            }
        else:
            region = _ellipse_mask(shape, center_yx=center, radii_yx=(ry, rx))
            record = {
                "kind": "ellipse",
                "center_yx": [float(center[0]), float(center[1])],
                "radii_yx": [ry, rx],
                "boundary_connected_requested": connects,
            }
        mask |= region
        records.append(record)
    target_pixels = min(mask.size, max(1, int(np.floor(target_fraction * mask.size))))
    area_adjustment = "none"
    if int(mask.sum()) < target_pixels:
        distance = distance_transform_edt(~mask).ravel()
        add_count = target_pixels - int(mask.sum())
        candidates = np.flatnonzero(~mask.ravel())
        nearest = candidates[np.argpartition(distance[candidates], add_count - 1)[:add_count]]
        expanded = mask.ravel().copy()
        expanded[nearest] = True
        mask = expanded.reshape(shape)
        area_adjustment = "distance_expand_to_visible_target"
    elif int(mask.sum()) > target_pixels:
        distance = distance_transform_edt(mask).ravel()
        remove_count = int(mask.sum()) - target_pixels
        protected = np.zeros(mask.shape, dtype=bool)
        if boundary_connected:
            protected[0] = mask[0]
            protected[-1] = mask[-1]
            protected[:, 0] |= mask[:, 0]
            protected[:, -1] |= mask[:, -1]
        candidates = np.flatnonzero((mask & ~protected).ravel())
        if len(candidates) < remove_count:
            candidates = np.flatnonzero(mask.ravel())
        nearest_boundary = candidates[
            np.argpartition(distance[candidates], remove_count - 1)[:remove_count]
        ]
        trimmed = mask.ravel().copy()
        trimmed[nearest_boundary] = False
        mask = trimmed.reshape(shape)
        area_adjustment = "distance_trim_to_visible_target"
    if int(mask.sum()) != target_pixels:
        raise RuntimeError("vacuum geometry area adjustment did not reach target")
    touches_boundary = bool(mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any())
    return mask, {
        "shape_family": shape_name,
        "target_area_fraction": target_fraction,
        "target_pixel_count": target_pixels,
        "realized_area_fraction": float(mask.mean()),
        "realized_pixel_count": int(mask.sum()),
        "visible_area_adjustment": area_adjustment,
        "boundary_connected": touches_boundary,
        "regions": records,
    }


def _sample_count(rng: np.random.Generator, count: int, fraction_range: tuple[float, float], enabled: bool) -> tuple[int, float | None]:
    if not enabled or count <= 0:
        return 0, None
    fraction = float(rng.uniform(*fraction_range))
    return min(count, max(1, int(round(count * fraction)))), fraction


def _allocate_point_defect_counts(
    total_count: int,
    enabled_names: list[str],
    weights: tuple[float, float, float],
    rng: np.random.Generator,
) -> dict[str, int]:
    counts = {name: 0 for name in ("vacancy", "substitution_weak", "substitution_strong")}
    if total_count <= 0 or not enabled_names:
        return counts
    weight_map = dict(zip(counts, weights))
    probabilities = np.asarray([weight_map[name] for name in enabled_names], dtype=float)
    probabilities /= probabilities.sum()
    if total_count < len(enabled_names):
        chosen = rng.choice(enabled_names, size=total_count, replace=False, p=probabilities)
        for name in chosen:
            counts[str(name)] += 1
        return counts
    for name in enabled_names:
        counts[name] = 1
    remaining = total_count - len(enabled_names)
    if remaining:
        extras = rng.multinomial(remaining, probabilities)
        for name, extra in zip(enabled_names, extras):
            counts[name] += int(extra)
    return counts


def _selected_column_mask(labels: np.ndarray, column_ids: np.ndarray) -> np.ndarray:
    if not len(column_ids):
        return np.zeros(labels.shape, dtype=bool)
    return np.isin(labels, column_ids.astype(np.int64) + 1)


def apply_defects(image: np.ndarray, config: DefectConfig) -> DefectResult:
    """Apply seeded vacancy/substitution/vacuum changes to one clean image."""

    config.validate()
    base = validate_2d_image(image).astype(np.float32, copy=False)
    if config.source_family == "quasi":
        raise ValueError(
            "quasi defects require apply_twisted_layer_vacancies(); "
            "post-overlap raster peak deletion is disabled"
        )
    detection = detect_atomic_columns(base, config)
    partition = build_column_partition(base, detection, config)
    rng = np.random.default_rng(config.seed)
    coordinates = detection.coordinates_yx
    total = len(coordinates)
    enable_point_defects = config.variant_mode in {"defect_only", "defect_and_vacuum"}
    enable_weak = config.enable_weak and enable_point_defects
    enable_strong = config.enable_strong and enable_point_defects
    enable_vacuum = config.enable_vacuum and config.variant_mode in {
        "vacuum_only",
        "defect_and_vacuum",
    }

    masks = {
        "defect": np.zeros(base.shape, dtype=bool),
        "vacancy": np.zeros(base.shape, dtype=bool),
        "substitution_weak": np.zeros(base.shape, dtype=bool),
        "substitution_strong": np.zeros(base.shape, dtype=bool),
        "vacuum": np.zeros(base.shape, dtype=bool),
        "vacuum_columns": np.zeros(base.shape, dtype=bool),
    }
    vacuum_metadata: dict = {"enabled": False, "deleted_column_ids": []}
    vacuum_indices = np.empty(0, dtype=np.int64)
    if enable_vacuum:
        vacuum_mask, sampled = generate_vacuum_mask(base.shape, config, rng)
        masks["vacuum"] = vacuum_mask
        inside = vacuum_mask[coordinates[:, 0], coordinates[:, 1]]
        vacuum_indices = np.flatnonzero(inside)
        vacuum_metadata = {
            "enabled": True,
            **sampled,
            "deleted_column_ids": vacuum_indices.astype(int).tolist(),
        }

    available = np.setdiff1d(np.arange(total, dtype=np.int64), vacuum_indices, assume_unique=False)
    rng.shuffle(available)
    enabled_names = []
    if config.enable_vacancy and enable_point_defects:
        enabled_names.append("vacancy")
    if enable_weak:
        enabled_names.append("substitution_weak")
    if enable_strong:
        enabled_names.append("substitution_strong")
    point_count, sampled_point_fraction = _sample_count(
        rng,
        len(available),
        config.point_defect_fraction_range,
        bool(enabled_names),
    )
    allocated_counts = _allocate_point_defect_counts(
        point_count,
        enabled_names,
        config.point_defect_category_weights,
        rng,
    )
    cursor = 0
    category_indices: dict[str, np.ndarray] = {}
    for name in ("vacancy", "substitution_weak", "substitution_strong"):
        n = allocated_counts[name]
        category_indices[name] = available[cursor : cursor + n]
        cursor += n

    weak_factor = float(rng.uniform(*config.weak_factor_range)) if len(category_indices["substitution_weak"]) else None
    strong_factor = float(rng.uniform(*config.strong_factor_range)) if len(category_indices["substitution_strong"]) else None
    output = base.copy()
    records: dict[str, list[dict]] = {
        "vacancy": [],
        "substitution_weak": [],
        "substitution_strong": [],
        "vacuum_columns": [],
    }
    factors = {
        "vacancy": 0.0,
        "substitution_weak": weak_factor,
        "substitution_strong": strong_factor,
        "vacuum_columns": 0.0,
    }
    all_indices = {**category_indices, "vacuum_columns": vacuum_indices}
    for name, indices in all_indices.items():
        factor = factors[name]
        if factor is None:
            continue
        selected_mask = _selected_column_mask(partition.labels, indices)
        masks[name] = selected_mask
        if factor == 0.0:
            output[selected_mask] = partition.background
        else:
            output[selected_mask] = partition.background + (
                base[selected_mask] - partition.background
            ) * float(factor)
        for column_id in indices:
            y, x = coordinates[column_id].astype(int)
            column_mask = partition.labels == int(column_id) + 1
            records[name].append(
                {
                    "column_id": int(column_id),
                    "yx": [int(y), int(x)],
                    "factor": float(factor),
                    "mask_pixel_count": int(column_mask.sum()),
                    "original_peak_above_background": float(base[y, x] - partition.background),
                    "new_peak_above_background": float(output[y, x] - partition.background),
                }
            )

    output = output.astype(np.float32, copy=False)
    masks["defect"] = (
        masks["vacancy"]
        | masks["substitution_weak"]
        | masks["substitution_strong"]
        | masks["vacuum_columns"]
    )
    if vacuum_metadata["enabled"]:
        vacuum_metadata.update(
            {
                "deleted_column_count": int(len(vacuum_indices)),
                "deleted_column_fraction_of_detected": (
                    float(len(vacuum_indices) / total) if total else 0.0
                ),
                "deleted_column_mask_fraction": float(
                    masks["vacuum_columns"].mean()
                ),
            }
        )
    metadata = {
        "generator": "column-mask-defects-v3",
        "seed": int(config.seed),
        "source_family": config.source_family,
        "raster_method": "foreground_nearest_column_partition_then_mask_edit",
        "raster_approximation": config.source_family == "multislice",
        "config": asdict(config),
        "detector": {
            "detected_column_count": total,
            "background": detection.background,
            "threshold": detection.threshold,
            "median_nearest_spacing_pixels": detection.median_nearest_spacing_pixels,
            "estimated_sigma_pixels": detection.estimated_sigma_pixels,
            "coordinates_yx": coordinates.tolist(),
            "column_mask_background": partition.background,
            "column_mask_threshold": partition.threshold,
            "column_mask_foreground_fraction": float(partition.foreground.mean()),
        },
        "policy": {
            "one_factor_per_image_per_substitution_category": True,
            "non_overlapping_column_masks": True,
            "vacancy_and_vacuum_set_full_column_mask_to_background": True,
            "substitution_scales_full_column_mask_relative_to_background": True,
            "quasi_requires_pre_rotation_layer_deletion": True,
        },
        "resolved": {
            "weak_factor": weak_factor,
            "strong_factor": strong_factor,
            "point_defect_fraction_sampled": sampled_point_fraction,
            "point_defect_count": int(point_count),
            "point_defect_fraction_of_available_columns": (
                float(point_count / len(available)) if len(available) else 0.0
            ),
            "counts": {name: len(items) for name, items in records.items()},
        },
        "columns": records,
        "vacuum": vacuum_metadata,
    }
    return DefectResult(
        image=output,
        masks={name: value.astype(np.uint8) for name, value in masks.items()},
        metadata=metadata,
    )


def apply_twisted_layer_vacancies(
    twisted_config,
    defect_config: DefectConfig,
) -> tuple[np.ndarray, DefectResult]:
    """Delete quasi columns independently per layer before rotation and overlap."""

    defect_config.validate()
    if defect_config.source_family != "quasi":
        raise ValueError("defect_config.source_family must be 'quasi'")
    from .twisted_bilayer import generate_twisted_bilayer_pair

    rng = np.random.default_rng(defect_config.seed)
    if defect_config.variant_mode != "defect_only":
        raise ValueError("quasi supports only variant_mode='defect_only'")
    if defect_config.enable_vacancy:
        bottom_fraction = float(rng.uniform(*defect_config.point_defect_fraction_range))
        top_fraction = float(rng.uniform(*defect_config.point_defect_fraction_range))
    else:
        bottom_fraction = top_fraction = 0.0
    resolved_config = replace(
        twisted_config,
        bottom_vacancy_fraction=bottom_fraction,
        top_vacancy_fraction=top_fraction,
        vacancy_seed=defect_config.seed,
    )
    perfect, defect, layer_masks, twisted_metadata = generate_twisted_bilayer_pair(
        resolved_config
    )
    zeros = np.zeros(perfect.shape, dtype=np.uint8)
    vacancy = layer_masks["vacancy"].astype(np.uint8)
    masks = {
        "defect": vacancy.copy(),
        "vacancy": vacancy,
        "bottom_layer_vacancy": layer_masks["bottom_layer_vacancy"].astype(np.uint8),
        "top_layer_vacancy": layer_masks["top_layer_vacancy"].astype(np.uint8),
        "substitution_weak": zeros.copy(),
        "substitution_strong": zeros.copy(),
        "vacuum": zeros.copy(),
        "vacuum_columns": zeros.copy(),
    }
    metadata = {
        "generator": "twisted-layer-vacancies-v1",
        "seed": int(defect_config.seed),
        "source_family": "quasi",
        "config": asdict(defect_config),
        "twisted": twisted_metadata,
        "resolved": {
            "bottom_vacancy_fraction": bottom_fraction,
            "top_vacancy_fraction": top_fraction,
            "bottom_visible_deleted_count": twisted_metadata["bottom_layer"]["vacancy"]["visible_deleted_count"],
            "top_visible_deleted_count": twisted_metadata["top_layer"]["vacancy"]["visible_deleted_count"],
        },
        "policy": {
            "quasi_vacancy_only": True,
            "selection_before_rotation_and_overlap": True,
            "independent_layer_seed_streams": True,
            "single_layer_vacancy_can_retain_other_layer_signal": True,
        },
    }
    return perfect, DefectResult(image=defect, masks=masks, metadata=metadata)
