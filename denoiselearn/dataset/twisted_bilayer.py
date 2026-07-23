"""Configurable projected-column STEM-like twisted bilayer generator.

This lightweight generator is intended for structural coverage and pipeline
validation.  It is not a replacement for a multislice electron-scattering
simulation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Mapping, Sequence

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter

from .contracts import DEFAULT_IMAGE_SIZE, ImageContract, validate_2d_image


@dataclass(frozen=True)
class ColumnSite:
    fractional_xy: tuple[float, float]
    species: str


@dataclass(frozen=True)
class Material:
    name: str
    lattice_constant_angstrom: float
    sites: tuple[ColumnSite, ...]


@dataclass(frozen=True)
class TwistedSpecialCase:
    case_id: str
    bottom_material: str
    top_material: str
    twist_angle_deg: float


# Material names retain lattice/species provenance and allow homo/heterobilayer
# pairing. They do not select material-specific intensity or sigma presets:
# graphene has equal A/B columns; every other entry uses the same seeded ranges.
MATERIALS: Mapping[str, Material] = {
    "graphene": Material(
        name="graphene",
        lattice_constant_angstrom=2.46,
        sites=(ColumnSite((0.0, 0.0), "C_A"), ColumnSite((1 / 3, 1 / 3), "C_B")),
    ),
    "MoS2": Material(
        name="MoS2",
        lattice_constant_angstrom=3.16,
        sites=(ColumnSite((0.0, 0.0), "Mo"), ColumnSite((1 / 3, 1 / 3), "S2")),
    ),
    "WS2": Material(
        name="WS2",
        lattice_constant_angstrom=3.15,
        sites=(ColumnSite((0.0, 0.0), "W"), ColumnSite((1 / 3, 1 / 3), "S2")),
    ),
    "MoSe2": Material(
        name="MoSe2",
        lattice_constant_angstrom=3.29,
        sites=(ColumnSite((0.0, 0.0), "Mo"), ColumnSite((1 / 3, 1 / 3), "Se2")),
    ),
    "WSe2": Material(
        name="WSe2",
        lattice_constant_angstrom=3.28,
        sites=(ColumnSite((0.0, 0.0), "W"), ColumnSite((1 / 3, 1 / 3), "Se2")),
    ),
}


DEFAULT_SPECIAL_CASES: tuple[TwistedSpecialCase, ...] = (
    TwistedSpecialCase("graphene_magic_1p1", "graphene", "graphene", 1.1),
    TwistedSpecialCase("tmd_anchor_6", "MoS2", "MoS2", 6.0),
    TwistedSpecialCase("tmd_anchor_19", "MoS2", "WS2", 19.0),
    TwistedSpecialCase("graphene_commensurate_21p8", "graphene", "graphene", 21.8),
    TwistedSpecialCase("graphene_quasicrystal_30", "graphene", "graphene", 30.0),
    TwistedSpecialCase("tmd_anchor_38p2", "MoSe2", "WSe2", 38.2),
    TwistedSpecialCase("tmd_quasicrystal_57p1", "MoS2", "MoS2", 57.1),
)

DEFAULT_MATERIAL_PAIRS: tuple[tuple[str, str], ...] = (
    ("graphene", "graphene"),
    ("MoS2", "MoS2"),
    ("WS2", "WS2"),
    ("MoSe2", "MoSe2"),
    ("WSe2", "WSe2"),
    ("MoS2", "WS2"),
    ("MoS2", "WSe2"),
    ("MoSe2", "WSe2"),
    ("graphene", "MoS2"),
)


@dataclass(frozen=True)
class TwistedBilayerConfig:
    bottom_material: str = "graphene"
    top_material: str = "graphene"
    twist_angle_deg: float = 1.1
    translation_angstrom: tuple[float, float] = (0.0, 0.0)
    image_size: int = DEFAULT_IMAGE_SIZE
    pixel_size_angstrom: float | None = None
    lattice_pixels_range: tuple[float, float] = (19.0, 30.0)
    sigma_fraction_range: tuple[float, float] = (0.16, 0.357)
    min_sigma_pixels: float = 2.0
    ab_intensity_ratio_range: tuple[float, float] = (0.2, 0.8)
    bottom_ab_intensity_ratio: float | None = None
    top_ab_intensity_ratio: float | None = None
    bottom_sigma_angstrom: float | None = None
    top_sigma_angstrom: float | None = None
    bottom_gain: float = 1.0
    top_gain: float = 0.90
    background: float = 0.0
    seed: int = 0
    angle_source: str = "explicit"
    special_case_id: str | None = None
    sampling_seed: int | None = None
    sample_index: int | None = None
    bottom_vacancy_fraction: float = 0.0
    top_vacancy_fraction: float = 0.0
    vacancy_seed: int | None = None
    vacancy_support_sigma: float = 3.0


def _rotation(angle_deg: float) -> np.ndarray:
    angle = np.deg2rad(angle_deg)
    return np.asarray([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])


def _validate_range(name: str, values: tuple[float, float]) -> None:
    low, high = values
    if low <= 0 or high < low:
        raise ValueError(f"{name} must satisfy 0 < low <= high, got {values}")


def _nearest_column_spacing(material: Material) -> float:
    """Return the nearest projected-column spacing in Angstrom."""

    a = material.lattice_constant_angstrom
    vectors = np.asarray([[a, 0.0], [0.5 * a, np.sqrt(3.0) * 0.5 * a]])
    central = [np.asarray(site.fractional_xy) @ vectors for site in material.sites]
    candidates = []
    for i in (-1, 0, 1):
        for j in (-1, 0, 1):
            shift = np.asarray([i, j]) @ vectors
            for site in material.sites:
                candidates.append(np.asarray(site.fractional_xy) @ vectors + shift)
    distances = [
        float(np.linalg.norm(candidate - origin))
        for origin in central
        for candidate in candidates
        if float(np.linalg.norm(candidate - origin)) > 1e-9
    ]
    return min(distances)


def estimate_moire_period_pixels(
    bottom_lattice_pixels: float,
    top_lattice_pixels: float,
    twist_angle_deg: float,
) -> float:
    """Estimate moire period from the difference of two hexagonal reciprocal vectors."""

    if bottom_lattice_pixels <= 0 or top_lattice_pixels <= 0:
        raise ValueError("lattice lengths must be positive")
    if not 0.0 <= twist_angle_deg <= 60.0:
        raise ValueError("twist_angle_deg must be within [0, 60]")
    angle = np.deg2rad(twist_angle_deg)
    denominator_sq = (
        bottom_lattice_pixels**2
        + top_lattice_pixels**2
        - 2.0 * bottom_lattice_pixels * top_lattice_pixels * np.cos(angle)
    )
    if denominator_sq <= np.finfo(float).eps:
        return float("inf")
    return (
        bottom_lattice_pixels
        * top_lattice_pixels
        / np.sqrt(denominator_sq)
    )


def _lattice_points(
    material: Material,
    field_angstrom: float,
    padding: float,
    ab_intensity_ratio: float,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    a = material.lattice_constant_angstrom
    vectors = np.asarray([[a, 0.0], [0.5 * a, np.sqrt(3.0) * 0.5 * a]])
    extent = field_angstrom * np.sqrt(2.0) / 2.0 + padding + 2.0 * a
    n = int(np.ceil(extent / (a * np.sqrt(3.0) / 2.0))) + 3
    lattice_indices = np.stack(
        np.meshgrid(np.arange(-n, n + 1), np.arange(-n, n + 1), indexing="ij"),
        axis=-1,
    ).reshape(-1, 2)
    origins = lattice_indices @ vectors
    positions = []
    amplitudes = []
    species = []
    for site_index, site in enumerate(material.sites):
        offset = np.asarray(site.fractional_xy) @ vectors
        positions.append(origins + offset)
        amplitude = 1.0 if site_index == 0 else ab_intensity_ratio
        amplitudes.append(np.full(len(origins), amplitude, dtype=np.float64))
        species.extend([site.species] * len(origins))
    return np.vstack(positions), np.concatenate(amplitudes), species


def _render_layer(
    material: Material,
    *,
    image_size: int,
    pixel_size: float,
    angle_deg: float,
    translation: tuple[float, float],
    sigma_angstrom: float,
    ab_intensity_ratio: float,
    ab_intensity_ratio_source: str,
    gain: float,
    vacancy_fraction: float,
    vacancy_seed: int,
    vacancy_support_sigma: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    field = image_size * pixel_size
    padding = 4.0 * sigma_angstrom
    positions, amplitudes, species = _lattice_points(
        material,
        field,
        padding,
        ab_intensity_ratio,
    )
    column_ids = np.arange(len(positions), dtype=np.int64)
    deleted_before_rotation = np.zeros(len(positions), dtype=bool)
    if vacancy_fraction > 0.0:
        vacancy_rng = np.random.default_rng(vacancy_seed)
        deleted_count = min(
            len(positions),
            max(1, int(round(vacancy_fraction * len(positions)))),
        )
        deleted_before_rotation[vacancy_rng.permutation(len(positions))[:deleted_count]] = True
    positions = positions @ _rotation(angle_deg).T + np.asarray(translation)
    half = field / 2.0
    keep = (
        (positions[:, 0] >= -half - padding)
        & (positions[:, 0] < half + padding)
        & (positions[:, 1] >= -half - padding)
        & (positions[:, 1] < half + padding)
    )
    positions = positions[keep]
    amplitudes = amplitudes[keep] * gain
    kept_species = np.asarray(species, dtype=object)[keep]
    kept_column_ids = column_ids[keep]
    kept_deleted = deleted_before_rotation[keep]
    pixel_xy = (positions + half) / pixel_size
    col = np.rint(pixel_xy[:, 0]).astype(int)
    row = np.rint(pixel_xy[:, 1]).astype(int)
    inside = (row >= 0) & (row < image_size) & (col >= 0) & (col < image_size)
    perfect_impulses = np.zeros((image_size, image_size), dtype=np.float32)
    np.add.at(
        perfect_impulses,
        (row[inside], col[inside]),
        amplitudes[inside].astype(np.float32),
    )
    defect_impulses = np.zeros((image_size, image_size), dtype=np.float32)
    retained_inside = inside & ~kept_deleted
    np.add.at(
        defect_impulses,
        (row[retained_inside], col[retained_inside]),
        amplitudes[retained_inside].astype(np.float32),
    )
    sigma_pixels = sigma_angstrom / pixel_size
    perfect_image = gaussian_filter(perfect_impulses, sigma=sigma_pixels, mode="wrap")
    image = gaussian_filter(defect_impulses, sigma=sigma_pixels, mode="wrap")
    vacancy_markers = np.zeros((image_size, image_size), dtype=bool)
    deleted_inside = inside & kept_deleted
    vacancy_markers[row[deleted_inside], col[deleted_inside]] = True
    if vacancy_markers.any():
        vacancy_mask = distance_transform_edt(~vacancy_markers) <= (
            vacancy_support_sigma * sigma_pixels
        )
    else:
        vacancy_mask = np.zeros((image_size, image_size), dtype=bool)
    counts = {str(name): int(np.sum(kept_species[inside] == name)) for name in set(kept_species[inside])}
    return perfect_image, image, vacancy_mask, {
        "material": material.name,
        "lattice_constant_angstrom": material.lattice_constant_angstrom,
        "angle_deg": angle_deg,
        "translation_angstrom": list(translation),
        "sigma_angstrom": sigma_angstrom,
        "sigma_pixels": sigma_pixels,
        "ab_intensity_ratio": ab_intensity_ratio,
        "ab_intensity_ratio_definition": "B_column_intensity / A_column_intensity",
        "ab_intensity_ratio_source": ab_intensity_ratio_source,
        "gain": gain,
        "column_counts": counts,
        "vacancy": {
            "selection_stage": "unrotated_column_list_before_rotation",
            "requested_fraction": float(vacancy_fraction),
            "seed": int(vacancy_seed),
            "deleted_column_ids_before_rotation": column_ids[deleted_before_rotation].astype(int).tolist(),
            "visible_deleted_column_ids": kept_column_ids[deleted_inside].astype(int).tolist(),
            "visible_deleted_count": int(np.sum(deleted_inside)),
            "support_sigma": float(vacancy_support_sigma),
        },
        "contrast_policy": (
            "graphene_equal_sublattices"
            if material.name == "graphene"
            else "seeded_non_graphene_ab_ratio"
        ),
    }


def generate_twisted_bilayer_pair(
    config: TwistedBilayerConfig,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], dict]:
    """Generate aligned perfect/defect bilayers with pre-rotation layer vacancies."""

    ImageContract(image_size=config.image_size).validate()
    _validate_range("lattice_pixels_range", config.lattice_pixels_range)
    _validate_range("sigma_fraction_range", config.sigma_fraction_range)
    _validate_range("ab_intensity_ratio_range", config.ab_intensity_ratio_range)
    if not 0.0 <= config.twist_angle_deg <= 60.0:
        raise ValueError("twist_angle_deg must be within [0, 60]")
    if config.min_sigma_pixels <= 0:
        raise ValueError("min_sigma_pixels must be positive")
    if config.pixel_size_angstrom is not None and config.pixel_size_angstrom <= 0:
        raise ValueError("pixel_size_angstrom must be positive")
    for name in ("bottom_vacancy_fraction", "top_vacancy_fraction"):
        value = getattr(config, name)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must lie within [0, 1]")
    if config.vacancy_support_sigma <= 0:
        raise ValueError("vacancy_support_sigma must be positive")
    try:
        bottom = MATERIALS[config.bottom_material]
        top = MATERIALS[config.top_material]
    except KeyError as error:
        raise ValueError(f"unknown material {error.args[0]!r}; choose from {sorted(MATERIALS)}") from error
    rng = np.random.default_rng(config.seed)
    geometric_mean_lattice = math.sqrt(
        bottom.lattice_constant_angstrom * top.lattice_constant_angstrom
    )
    if config.pixel_size_angstrom is None:
        reference_lattice_pixels = float(rng.uniform(*config.lattice_pixels_range))
        pixel_size_angstrom = geometric_mean_lattice / reference_lattice_pixels
        pixel_scale_mode = "seeded_symmlearn_like_pixel_range"
    else:
        pixel_size_angstrom = config.pixel_size_angstrom
        reference_lattice_pixels = geometric_mean_lattice / pixel_size_angstrom
        pixel_scale_mode = "explicit_pixel_size_angstrom"

    bottom_nearest_spacing = _nearest_column_spacing(bottom)
    top_nearest_spacing = _nearest_column_spacing(top)
    bottom_sigma_fraction = float(rng.uniform(*config.sigma_fraction_range))
    top_sigma_fraction = float(rng.uniform(*config.sigma_fraction_range))
    bottom_sigma = (
        config.bottom_sigma_angstrom
        if config.bottom_sigma_angstrom is not None
        else max(
            config.min_sigma_pixels * pixel_size_angstrom,
            bottom_nearest_spacing * bottom_sigma_fraction,
        )
    )
    top_sigma = (
        config.top_sigma_angstrom
        if config.top_sigma_angstrom is not None
        else max(
            config.min_sigma_pixels * pixel_size_angstrom,
            top_nearest_spacing * top_sigma_fraction,
        )
    )

    sampled_ratios: dict[str, float] = {}

    def resolve_ab_ratio(material: Material, explicit: float | None) -> tuple[float, str]:
        if material.name == "graphene":
            if explicit is not None and not np.isclose(explicit, 1.0):
                raise ValueError("graphene requires equivalent A/B columns with ratio 1.0")
            return 1.0, "graphene_equal_sublattices"
        if explicit is not None:
            if explicit <= 0:
                raise ValueError("explicit A/B intensity ratio must be positive")
            return float(explicit), "explicit"
        if material.name not in sampled_ratios:
            sampled_ratios[material.name] = float(rng.uniform(*config.ab_intensity_ratio_range))
        return sampled_ratios[material.name], "seeded_uniform_by_material"

    bottom_ab_ratio, bottom_ab_source = resolve_ab_ratio(
        bottom,
        config.bottom_ab_intensity_ratio,
    )
    top_ab_ratio, top_ab_source = resolve_ab_ratio(
        top,
        config.top_ab_intensity_ratio,
    )
    jitter = rng.uniform(-0.02, 0.02, size=2)
    vacancy_seed = config.seed if config.vacancy_seed is None else config.vacancy_seed
    vacancy_seed_sequence = np.random.SeedSequence(vacancy_seed).spawn(2)
    bottom_vacancy_seed = int(vacancy_seed_sequence[0].generate_state(1)[0])
    top_vacancy_seed = int(vacancy_seed_sequence[1].generate_state(1)[0])
    bottom_perfect, bottom_image, bottom_vacancy_mask, bottom_meta = _render_layer(
        bottom,
        image_size=config.image_size,
        pixel_size=pixel_size_angstrom,
        angle_deg=-config.twist_angle_deg / 2.0,
        translation=(float(jitter[0]), float(jitter[1])),
        sigma_angstrom=bottom_sigma,
        ab_intensity_ratio=bottom_ab_ratio,
        ab_intensity_ratio_source=bottom_ab_source,
        gain=config.bottom_gain,
        vacancy_fraction=config.bottom_vacancy_fraction,
        vacancy_seed=bottom_vacancy_seed,
        vacancy_support_sigma=config.vacancy_support_sigma,
    )
    top_translation = (
        config.translation_angstrom[0] - float(jitter[0]),
        config.translation_angstrom[1] - float(jitter[1]),
    )
    top_perfect, top_image, top_vacancy_mask, top_meta = _render_layer(
        top,
        image_size=config.image_size,
        pixel_size=pixel_size_angstrom,
        angle_deg=config.twist_angle_deg / 2.0,
        translation=top_translation,
        sigma_angstrom=top_sigma,
        ab_intensity_ratio=top_ab_ratio,
        ab_intensity_ratio_source=top_ab_source,
        gain=config.top_gain,
        vacancy_fraction=config.top_vacancy_fraction,
        vacancy_seed=top_vacancy_seed,
        vacancy_support_sigma=config.vacancy_support_sigma,
    )
    perfect_image = validate_2d_image(bottom_perfect + top_perfect + config.background)
    image = validate_2d_image(bottom_image + top_image + config.background)
    bottom_meta.update(
        {
            "lattice_constant_pixels": bottom.lattice_constant_angstrom / pixel_size_angstrom,
            "nearest_column_spacing_pixels": bottom_nearest_spacing / pixel_size_angstrom,
            "sigma_fraction_of_nearest_spacing": (
                None if config.bottom_sigma_angstrom is not None else bottom_sigma_fraction
            ),
        }
    )
    top_meta.update(
        {
            "lattice_constant_pixels": top.lattice_constant_angstrom / pixel_size_angstrom,
            "nearest_column_spacing_pixels": top_nearest_spacing / pixel_size_angstrom,
            "sigma_fraction_of_nearest_spacing": (
                None if config.top_sigma_angstrom is not None else top_sigma_fraction
            ),
        }
    )
    expected_moire_period_pixels = estimate_moire_period_pixels(
        bottom_meta["lattice_constant_pixels"],
        top_meta["lattice_constant_pixels"],
        config.twist_angle_deg,
    )
    metadata = {
        "source_id": "twisted_bilayer",
        "generator": "projected-column-stem-like-v2",
        "physics_scope": "synthetic projected columns; not multislice",
        "config": asdict(config),
        "angle_sampling": {
            "domain_deg": [0.0, 60.0],
            "source": config.angle_source,
            "special_case_id": config.special_case_id,
            "resolved_twist_angle_deg": config.twist_angle_deg,
        },
        "expected_moire_period_pixels": expected_moire_period_pixels,
        "moire_period_note": "reciprocal-vector estimate; infinity means aligned equal lattices",
        "pixel_scale": {
            "mode": pixel_scale_mode,
            "pixel_size_angstrom": pixel_size_angstrom,
            "reference_lattice_pixels": reference_lattice_pixels,
            "reference_definition": "geometric mean of bottom/top lattice-vector lengths",
            "symmlearn_a_range_reference": list(config.lattice_pixels_range),
            "symmlearn_sigma_fraction_reference": list(config.sigma_fraction_range),
        },
        "bottom_layer": bottom_meta,
        "top_layer": top_meta,
    }
    masks = {
        "vacancy": (bottom_vacancy_mask | top_vacancy_mask).astype(np.uint8),
        "bottom_layer_vacancy": bottom_vacancy_mask.astype(np.uint8),
        "top_layer_vacancy": top_vacancy_mask.astype(np.uint8),
    }
    metadata["vacancy_policy"] = {
        "selection_stage": "independent_layer_column_lists_before_rotation_and_overlap",
        "bottom_requested_fraction": float(config.bottom_vacancy_fraction),
        "top_requested_fraction": float(config.top_vacancy_fraction),
        "vacancy_seed": int(vacancy_seed),
        "other_layer_signal_may_remain_in_single_layer_vacancy_roi": True,
    }
    return perfect_image, image, masks, metadata


def generate_twisted_bilayer(config: TwistedBilayerConfig) -> tuple[np.ndarray, dict]:
    """Generate a deterministic two-layer projected-column image."""

    _, image, _, metadata = generate_twisted_bilayer_pair(config)
    return image, metadata


def build_twisted_bilayer_configs(
    count: int,
    *,
    seed: int,
    image_size: int = DEFAULT_IMAGE_SIZE,
    special_cases: Sequence[TwistedSpecialCase] = DEFAULT_SPECIAL_CASES,
    material_pairs: Sequence[tuple[str, str]] = DEFAULT_MATERIAL_PAIRS,
    generated_angle_range: tuple[float, float] = (0.0, 60.0),
) -> list[TwistedBilayerConfig]:
    """Build a reproducible batch with required angle cases plus random angles."""

    if count < len(special_cases):
        raise ValueError(
            f"count={count} cannot cover all {len(special_cases)} required special cases"
        )
    if not material_pairs:
        raise ValueError("material_pairs must not be empty")
    low, high = generated_angle_range
    if low < 0 or high > 60 or high < low:
        raise ValueError("generated_angle_range must satisfy 0 <= low <= high <= 60")
    for bottom_material, top_material in material_pairs:
        if bottom_material not in MATERIALS or top_material not in MATERIALS:
            raise ValueError("material_pairs contains an unknown material")

    rng = np.random.default_rng(seed)
    configs: list[TwistedBilayerConfig] = []
    for index, case in enumerate(special_cases):
        configs.append(
            TwistedBilayerConfig(
                bottom_material=case.bottom_material,
                top_material=case.top_material,
                twist_angle_deg=case.twist_angle_deg,
                image_size=image_size,
                seed=int(rng.integers(0, 2**32, dtype=np.uint32)),
                angle_source="fixed_special_case",
                special_case_id=case.case_id,
                sampling_seed=seed,
                sample_index=index,
            )
        )

    for index in range(len(special_cases), count):
        bottom_material, top_material = material_pairs[
            int(rng.integers(0, len(material_pairs)))
        ]
        configs.append(
            TwistedBilayerConfig(
                bottom_material=bottom_material,
                top_material=top_material,
                twist_angle_deg=float(rng.uniform(low, high)),
                image_size=image_size,
                seed=int(rng.integers(0, 2**32, dtype=np.uint32)),
                angle_source="seeded_uniform",
                sampling_seed=seed,
                sample_index=index,
            )
        )
    return configs
