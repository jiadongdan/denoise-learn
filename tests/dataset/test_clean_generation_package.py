from __future__ import annotations

import inspect
from pathlib import Path

from denoiselearn import dataset


def test_clean_generation_public_api_is_importable():
    expected = (
        "apply_defects",
        "build_structure_split",
        "generate_image_only",
        "generate_twisted_bilayer",
        "inspect_tileability",
        "read_clean_h5",
        "write_clean_master_bundle",
    )
    for name in expected:
        assert callable(getattr(dataset, name))


def test_symmlearn_external_inputs_must_be_supplied_explicitly():
    repository_parameter = inspect.signature(
        dataset.generate_image_only
    ).parameters[
        "repo_path"
    ]
    registry_parameter = inspect.signature(
        dataset.load_seed_registry
    ).parameters["path"]
    assert repository_parameter.default is inspect.Parameter.empty
    assert registry_parameter.default is inspect.Parameter.empty


def test_clean_generation_source_has_no_machine_specific_absolute_paths():
    package_root = Path(dataset.__file__).resolve().parent
    forbidden = ("C:\\Users\\", "D:\\work\\", "Denoise benchmark")
    for path in package_root.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert not any(value in text for value in forbidden), path
