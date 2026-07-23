"""Offline clean-image generators for the Atomic S/TEM denoise project."""

from .contracts import DEFAULT_IMAGE_SIZE, INT16_SCALE, normalize_clean_image
from .clean_master import (
    CleanMasterRecord,
    SplitConfig,
    SplitPlan,
    build_structure_split,
    sha256_file,
    validate_no_split_leakage,
    write_clean_master_bundle,
)
from .defects import (
    ColumnPartition,
    DEFAULT_VARIANT_MIX,
    DefectConfig,
    DefectResult,
    DefectVariantMix,
    apply_defects,
    apply_twisted_layer_vacancies,
    build_defect_variant_modes,
    build_column_partition,
    detect_atomic_columns,
)
from .h5io import read_clean_h5, read_defect_h5, write_clean_h5, write_defect_h5
from .multislice_tiles import (
    TileabilityReport,
    generate_from_tiff,
    inspect_tileability,
    load_tiff_2d,
    self_tile_image,
)
from .symmlearn import (
    SymmLearnSeed,
    generate_image_only,
    load_seed_registry,
    select_stratified_records,
)
from .twisted_bilayer import (
    MATERIALS,
    Material,
    TwistedBilayerConfig,
    build_twisted_bilayer_configs,
    generate_twisted_bilayer,
)

__all__ = [
    "DEFAULT_IMAGE_SIZE",
    "INT16_SCALE",
    "CleanMasterRecord",
    "SplitConfig",
    "SplitPlan",
    "DefectConfig",
    "DefectResult",
    "ColumnPartition",
    "DEFAULT_VARIANT_MIX",
    "DefectVariantMix",
    "apply_defects",
    "apply_twisted_layer_vacancies",
    "build_defect_variant_modes",
    "build_column_partition",
    "detect_atomic_columns",
    "normalize_clean_image",
    "build_structure_split",
    "read_clean_h5",
    "read_defect_h5",
    "write_clean_h5",
    "write_defect_h5",
    "sha256_file",
    "validate_no_split_leakage",
    "write_clean_master_bundle",
    "TileabilityReport",
    "generate_from_tiff",
    "inspect_tileability",
    "load_tiff_2d",
    "self_tile_image",
    "SymmLearnSeed",
    "generate_image_only",
    "load_seed_registry",
    "select_stratified_records",
    "MATERIALS",
    "Material",
    "TwistedBilayerConfig",
    "build_twisted_bilayer_configs",
    "generate_twisted_bilayer",
]
