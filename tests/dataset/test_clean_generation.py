"""Tests for the offline clean-data generators."""

from __future__ import annotations

import json
import sys
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np
from scipy.ndimage import rotate

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from denoiselearn.dataset.contracts import (
    ImageContract,
    dequantize_int16,
    quantize_int16,
)
from denoiselearn.dataset.clean_master import (
    CleanMasterRecord,
    SplitConfig,
    SplitPlan,
    build_structure_split,
    sha256_file,
    validate_no_split_leakage,
    write_clean_master_bundle,
)
from denoiselearn.dataset.defects import (
    DefectConfig,
    apply_defects,
    apply_twisted_layer_vacancies,
    build_defect_variant_modes,
)
from denoiselearn.dataset.h5io import (
    read_clean_h5,
    read_defect_h5,
    write_clean_h5,
    write_defect_h5,
)
from denoiselearn.dataset.multislice_tiles import (
    inspect_tileability,
    self_tile_image,
)
from denoiselearn.dataset.twisted_bilayer import (
    DEFAULT_SPECIAL_CASES,
    TwistedBilayerConfig,
    build_twisted_bilayer_configs,
    estimate_moire_period_pixels,
    generate_twisted_bilayer,
)


class CleanDataGenerationTests(unittest.TestCase):
    @staticmethod
    def _synthetic_column_image(size: int = 128) -> np.ndarray:
        yy, xx = np.mgrid[:size, :size]
        image = np.full((size, size), 0.03, dtype=np.float32)
        for row, y in enumerate(range(8, size - 7, 16)):
            for column, x in enumerate(range(8, size - 7, 16)):
                amplitude = 0.65 if (row + column) % 2 else 0.95
                image += amplitude * np.exp(-0.5 * ((yy - y) ** 2 + (xx - x) ** 2) / 2.2**2)
        return image

    @staticmethod
    def _clean_master_records(include_masks: bool = False) -> list[CleanMasterRecord]:
        records: list[CleanMasterRecord] = []
        for group_index in range(30):
            source = ("symmlearn", "multislice", "quasi")[group_index % 3]
            base = np.full((16, 16), 100 + group_index, dtype=np.int16)
            group_id = f"group_{group_index:03d}"
            records.append(
                CleanMasterRecord(
                    image_id=f"perfect_{group_index:03d}",
                    base_structure_id=group_id,
                    split_group_id=group_id,
                    source_id=source,
                    variant_type="perfect",
                    image=base,
                )
            )
            if group_index < 10:
                variant_type = (
                    ["defect_only"] * 6
                    + ["vacuum_only"] * 2
                    + ["defect_and_vacuum"] * 2
                )[group_index]
                masks = None
                if include_masks:
                    mask = np.zeros((16, 16), dtype=np.uint8)
                    mask[2:5, 2:5] = 1
                    masks = {"defect": mask}
                records.append(
                    CleanMasterRecord(
                        image_id=f"variant_{group_index:03d}",
                        base_structure_id=group_id,
                        split_group_id=group_id,
                        source_id=source,
                        variant_type=variant_type,
                        image=base + 10,
                        masks=masks,
                    )
                )
        return records

    def test_rotation_safe_contract_rejects_724(self) -> None:
        with self.assertRaises(ValueError):
            ImageContract(image_size=724).validate()
        ImageContract(image_size=768).validate()

    def test_768_center_crop_contains_no_rotation_padding(self) -> None:
        mask = np.ones((768, 768), dtype=np.float32)
        for angle in (1.0, 15.0, 30.0, 45.0, 73.0, 89.0):
            rotated = rotate(mask, angle, reshape=False, order=0, mode="constant", cval=0.0)
            crop = rotated[128:640, 128:640]
            self.assertEqual(crop.shape, (512, 512))
            self.assertTrue(np.all(crop == 1.0), msg=f"padding entered crop at {angle} degrees")

    def test_int16_quantization_round_trip(self) -> None:
        image = np.linspace(0, 1, 4096, dtype=np.float32).reshape(64, 64)
        restored = dequantize_int16(quantize_int16(image))
        self.assertEqual(restored.dtype, np.float32)
        self.assertLessEqual(float(np.max(np.abs(restored - image))), 0.5 / 32767 + 1e-7)

    def test_h5_contract_has_only_image_and_metadata(self) -> None:
        images = [np.eye(32, dtype=np.float32), np.fliplr(np.eye(32, dtype=np.float32))]
        with tempfile.TemporaryDirectory() as directory:
            path = write_clean_h5(
                Path(directory) / "clean.h5",
                images,
                [{"id": "a"}, {"id": "b"}],
                generator="test",
            )
            with h5py.File(path, "r") as handle:
                self.assertEqual(set(handle.keys()), {"images", "metadata"})
                self.assertEqual(handle["images"].shape, (2, 1, 32, 32))
                self.assertEqual(handle["images"].dtype, np.int16)
                self.assertEqual(handle["images"].attrs["semantic_channel"], "clean_image_only")
            restored, metadata, attrs = read_clean_h5(path)
            self.assertEqual(restored.shape, (2, 32, 32))
            self.assertEqual([item["id"] for item in metadata], ["a", "b"])
            self.assertEqual(attrs["stored_dtype"], "int16")

    def test_self_tiling_never_mixes_sources(self) -> None:
        source = np.arange(15, dtype=np.float32).reshape(3, 5)
        tiled, metadata = self_tile_image(source, output_size=768, offset_yx=(1, 2))
        expected = np.tile(source, (257, 154))[1:769, 2:770]
        np.testing.assert_array_equal(tiled, expected)
        self.assertEqual(metadata["tiling_policy"], "single_source_self_tile_only")
        self.assertEqual(set(np.unique(tiled)), set(np.unique(source)))

    def test_tileability_qc_accepts_periodic_array(self) -> None:
        y, x = np.mgrid[:64, :80]
        periodic = np.sin(2 * np.pi * x / 80) + np.cos(2 * np.pi * y / 64)
        report = inspect_tileability(periodic)
        self.assertEqual((report.height, report.width), (64, 80))
        self.assertEqual(report.status, "PASS")

    def test_twisted_generator_is_deterministic_and_columns_are_wide(self) -> None:
        config = TwistedBilayerConfig(
            bottom_material="MoS2",
            top_material="WS2",
            twist_angle_deg=3.2,
            image_size=768,
            seed=11,
        )
        first, metadata = generate_twisted_bilayer(config)
        second, _ = generate_twisted_bilayer(config)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(first.shape, (768, 768))
        self.assertGreater(float(first.std()), 0)
        self.assertGreaterEqual(metadata["bottom_layer"]["sigma_pixels"], 2.0)
        self.assertGreaterEqual(metadata["top_layer"]["sigma_pixels"], 2.0)
        self.assertGreaterEqual(metadata["pixel_scale"]["reference_lattice_pixels"], 19.0)
        self.assertLessEqual(metadata["pixel_scale"]["reference_lattice_pixels"], 30.0)
        self.assertEqual(
            metadata["pixel_scale"]["mode"],
            "seeded_symmlearn_like_pixel_range",
        )
        for layer_name in ("bottom_layer", "top_layer"):
            layer = metadata[layer_name]
            self.assertGreaterEqual(layer["sigma_fraction_of_nearest_spacing"], 0.16)
            self.assertLessEqual(layer["sigma_fraction_of_nearest_spacing"], 0.357)

    def test_twisted_seed_controls_pixel_scale_and_sigma(self) -> None:
        first_config = TwistedBilayerConfig(seed=11)
        second_config = TwistedBilayerConfig(seed=12)
        _, first = generate_twisted_bilayer(first_config)
        _, second = generate_twisted_bilayer(second_config)
        self.assertNotEqual(
            first["pixel_scale"]["reference_lattice_pixels"],
            second["pixel_scale"]["reference_lattice_pixels"],
        )
        self.assertNotEqual(
            first["bottom_layer"]["sigma_pixels"],
            second["bottom_layer"]["sigma_pixels"],
        )

    def test_twisted_material_lattice_ratio_is_preserved(self) -> None:
        _, metadata = generate_twisted_bilayer(
            TwistedBilayerConfig(bottom_material="graphene", top_material="MoS2", seed=19)
        )
        observed = (
            metadata["top_layer"]["lattice_constant_pixels"]
            / metadata["bottom_layer"]["lattice_constant_pixels"]
        )
        self.assertAlmostEqual(observed, 3.16 / 2.46)

    def test_twisted_ab_ratio_is_seeded_and_shared_by_material(self) -> None:
        config = TwistedBilayerConfig(
            bottom_material="MoS2",
            top_material="MoS2",
            seed=23,
        )
        _, first = generate_twisted_bilayer(config)
        _, second = generate_twisted_bilayer(config)
        bottom_ratio = first["bottom_layer"]["ab_intensity_ratio"]
        top_ratio = first["top_layer"]["ab_intensity_ratio"]
        self.assertGreaterEqual(bottom_ratio, 0.2)
        self.assertLessEqual(bottom_ratio, 0.8)
        self.assertEqual(bottom_ratio, top_ratio)
        self.assertEqual(
            bottom_ratio,
            second["bottom_layer"]["ab_intensity_ratio"],
        )

    def test_graphene_ab_columns_remain_equal(self) -> None:
        _, metadata = generate_twisted_bilayer(TwistedBilayerConfig(seed=31))
        self.assertEqual(metadata["bottom_layer"]["ab_intensity_ratio"], 1.0)
        self.assertEqual(metadata["top_layer"]["ab_intensity_ratio"], 1.0)

    def test_angle_batch_covers_special_cases_then_seeded_uniform(self) -> None:
        configs = build_twisted_bilayer_configs(10, seed=20260722)
        repeated = build_twisted_bilayer_configs(10, seed=20260722)
        self.assertEqual(configs, repeated)
        self.assertEqual(
            [item.special_case_id for item in configs[: len(DEFAULT_SPECIAL_CASES)]],
            [item.case_id for item in DEFAULT_SPECIAL_CASES],
        )
        graphene_30 = next(
            item for item in configs if item.special_case_id == "graphene_quasicrystal_30"
        )
        self.assertEqual(graphene_30.twist_angle_deg, 30.0)
        self.assertEqual(graphene_30.bottom_material, "graphene")
        self.assertEqual(graphene_30.top_material, "graphene")
        for item in configs[len(DEFAULT_SPECIAL_CASES) :]:
            self.assertEqual(item.angle_source, "seeded_uniform")
            self.assertGreaterEqual(item.twist_angle_deg, 0.0)
            self.assertLessEqual(item.twist_angle_deg, 60.0)

    def test_angle_batch_rejects_count_below_required_coverage(self) -> None:
        with self.assertRaises(ValueError):
            build_twisted_bilayer_configs(len(DEFAULT_SPECIAL_CASES) - 1, seed=1)

    def test_moire_period_estimate_matches_equal_lattice_formula(self) -> None:
        lattice_pixels = 24.0
        angle_deg = 30.0
        expected = lattice_pixels / (2.0 * np.sin(np.deg2rad(angle_deg) / 2.0))
        observed = estimate_moire_period_pixels(
            lattice_pixels,
            lattice_pixels,
            angle_deg,
        )
        self.assertAlmostEqual(observed, expected)
        self.assertTrue(
            np.isinf(estimate_moire_period_pixels(lattice_pixels, lattice_pixels, 0.0))
        )

    def test_twisted_metadata_records_angle_and_moire_contract(self) -> None:
        config = build_twisted_bilayer_configs(10, seed=42)[0]
        _, metadata = generate_twisted_bilayer(config)
        self.assertEqual(metadata["angle_sampling"]["domain_deg"], [0.0, 60.0])
        self.assertEqual(metadata["angle_sampling"]["source"], "fixed_special_case")
        self.assertGreater(metadata["expected_moire_period_pixels"], 0.0)

    def test_defect_generation_is_deterministic_and_uses_one_factor_per_category(self) -> None:
        image = self._synthetic_column_image()
        config = DefectConfig(
            seed=17,
            enable_vacuum=False,
            point_defect_fraction_range=(0.05, 0.05),
        )
        first = apply_defects(image, config)
        second = apply_defects(image, config)
        np.testing.assert_array_equal(first.image, second.image)
        for name in first.masks:
            np.testing.assert_array_equal(first.masks[name], second.masks[name])
        weak = {item["factor"] for item in first.metadata["columns"]["substitution_weak"]}
        strong = {item["factor"] for item in first.metadata["columns"]["substitution_strong"]}
        self.assertEqual(len(weak), 1)
        self.assertEqual(len(strong), 1)
        self.assertGreaterEqual(next(iter(weak)), config.weak_factor_range[0])
        self.assertLessEqual(next(iter(weak)), config.weak_factor_range[1])
        self.assertGreaterEqual(next(iter(strong)), config.strong_factor_range[0])
        self.assertLessEqual(next(iter(strong)), config.strong_factor_range[1])

    def test_defect_categories_change_selected_column_peaks_in_expected_direction(self) -> None:
        image = self._synthetic_column_image()
        result = apply_defects(
            image,
            DefectConfig(
                seed=23,
                enable_vacuum=False,
                point_defect_fraction_range=(0.05, 0.05),
            ),
        )
        for category, comparison in (
            ("vacancy", lambda after, before: after < before),
            ("substitution_weak", lambda after, before: after < before),
            ("substitution_strong", lambda after, before: after > before),
        ):
            record = result.metadata["columns"][category][0]
            y, x = record["yx"]
            self.assertTrue(comparison(float(result.image[y, x]), float(image[y, x])))

    def test_column_masks_apply_exact_zero_or_one_factor_and_do_not_overlap(self) -> None:
        image = self._synthetic_column_image()
        result = apply_defects(
            image,
            DefectConfig(
                seed=23,
                enable_vacuum=False,
                point_defect_fraction_range=(0.05, 0.05),
            ),
        )
        background = result.metadata["detector"]["column_mask_background"]
        vacancy = result.masks["vacancy"].astype(bool)
        weak = result.masks["substitution_weak"].astype(bool)
        strong = result.masks["substitution_strong"].astype(bool)
        self.assertFalse((vacancy & weak).any())
        self.assertFalse((vacancy & strong).any())
        self.assertFalse((weak & strong).any())
        np.testing.assert_array_equal(result.image[vacancy], background)
        weak_factor = result.metadata["resolved"]["weak_factor"]
        strong_factor = result.metadata["resolved"]["strong_factor"]
        np.testing.assert_allclose(
            result.image[weak] - background,
            (image[weak] - background) * weak_factor,
            rtol=1e-6,
            atol=1e-7,
        )
        np.testing.assert_allclose(
            result.image[strong] - background,
            (image[strong] - background) * strong_factor,
            rtol=1e-6,
            atol=1e-7,
        )
        outside = ~result.masks["defect"].astype(bool)
        np.testing.assert_array_equal(result.image[outside], image[outside])

    def test_vacuum_removes_columns_without_painting_the_whole_shape(self) -> None:
        image = self._synthetic_column_image()
        config = DefectConfig(
            seed=31,
            vacuum_shape="double_ellipse",
            vacuum_area_fraction_range=(0.18, 0.18),
            vacuum_region_count_range=(2, 2),
            boundary_connected_probability=1.0,
            enable_weak=False,
            enable_strong=False,
        )
        result = apply_defects(image, config)
        self.assertTrue(result.metadata["vacuum"]["boundary_connected"])
        self.assertEqual(result.metadata["vacuum"]["shape_family"], "double_ellipse")
        self.assertEqual(len(result.metadata["vacuum"]["regions"]), 2)
        self.assertEqual(
            result.metadata["vacuum"]["target_pixel_count"],
            result.metadata["vacuum"]["realized_pixel_count"],
        )
        self.assertAlmostEqual(
            result.metadata["vacuum"]["realized_area_fraction"],
            result.metadata["vacuum"]["target_pixel_count"] / image.size,
        )
        self.assertGreater(result.metadata["resolved"]["counts"]["vacuum_columns"], 0)
        untouched_background = (result.masks["vacuum"] == 1) & (result.masks["defect"] == 0)
        self.assertGreater(int(untouched_background.sum()), 0)
        np.testing.assert_array_equal(result.image[untouched_background], image[untouched_background])
        vacuum_columns = result.masks["vacuum_columns"].astype(bool)
        background = result.metadata["detector"]["column_mask_background"]
        np.testing.assert_array_equal(result.image[vacuum_columns], background)

    def test_single_ellipse_vacuum_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported vacuum_shape"):
            DefectConfig(seed=32, vacuum_shape="ellipse").validate()  # type: ignore[arg-type]

    def test_polygon_and_double_ellipse_never_exceed_target_pixel_area(self) -> None:
        image = self._synthetic_column_image()
        for shape in ("polygon", "double_ellipse"):
            for seed in range(33, 38):
                result = apply_defects(
                    image,
                    DefectConfig(
                        seed=seed,
                        vacuum_shape=shape,
                        vacuum_area_fraction_range=(0.20, 0.20),
                        enable_vacancy=False,
                        enable_weak=False,
                        enable_strong=False,
                    ),
                )
                vacuum = result.metadata["vacuum"]
                self.assertEqual(
                    vacuum["target_pixel_count"],
                    vacuum["realized_pixel_count"],
                )
                self.assertLessEqual(vacuum["realized_area_fraction"], 0.20)
                if shape == "polygon":
                    self.assertTrue(
                        all(region["kind"] == "polygon" for region in vacuum["regions"])
                    )
                    self.assertTrue(
                        all(6 <= len(region["vertices_yx"]) <= 9 for region in vacuum["regions"])
                    )
                else:
                    self.assertEqual(len(vacuum["regions"]), 2)

    def test_quasi_rejects_post_overlap_raster_deletion(self) -> None:
        with self.assertRaisesRegex(ValueError, "apply_twisted_layer_vacancies"):
            apply_defects(
                self._synthetic_column_image(),
                DefectConfig(seed=37, source_family="quasi"),
            )

    def test_quasi_vacancies_are_selected_per_layer_before_rotation(self) -> None:
        twisted_config = TwistedBilayerConfig(
            bottom_material="graphene",
            top_material="MoS2",
            twist_angle_deg=30.0,
            image_size=768,
            seed=47,
        )
        base, result = apply_twisted_layer_vacancies(
            twisted_config,
            DefectConfig(
                seed=37,
                source_family="quasi",
                variant_mode="defect_only",
                point_defect_fraction_range=(0.01, 0.01),
            ),
        )
        repeated_base, repeated = apply_twisted_layer_vacancies(
            twisted_config,
            DefectConfig(
                seed=37,
                source_family="quasi",
                variant_mode="defect_only",
                point_defect_fraction_range=(0.01, 0.01),
            ),
        )
        np.testing.assert_array_equal(base, repeated_base)
        np.testing.assert_array_equal(result.image, repeated.image)
        self.assertTrue(result.masks["bottom_layer_vacancy"].any())
        self.assertTrue(result.masks["top_layer_vacancy"].any())
        self.assertGreater(result.metadata["resolved"]["bottom_visible_deleted_count"], 0)
        self.assertGreater(result.metadata["resolved"]["top_visible_deleted_count"], 0)
        self.assertTrue(result.metadata["policy"]["selection_before_rotation_and_overlap"])
        self.assertGreater(float(np.max(base - result.image)), 0.0)
        self.assertTrue(np.all(result.image <= base + 1e-7))

    def test_point_defect_range_is_combined_across_three_categories(self) -> None:
        image = self._synthetic_column_image()
        result = apply_defects(
            image,
            DefectConfig(
                seed=51,
                variant_mode="defect_only",
                point_defect_fraction_range=(0.05, 0.05),
            ),
        )
        counts = result.metadata["resolved"]["counts"]
        combined = counts["vacancy"] + counts["substitution_weak"] + counts["substitution_strong"]
        self.assertEqual(combined, result.metadata["resolved"]["point_defect_count"])
        self.assertEqual(combined, round(result.metadata["detector"]["detected_column_count"] * 0.05))
        self.assertEqual(result.metadata["config"]["point_defect_fraction_range"], (0.05, 0.05))

    def test_variant_mix_is_30_percent_of_base_with_20_5_5_split(self) -> None:
        modes = build_defect_variant_modes(26_500, seed=20260722)
        repeated = build_defect_variant_modes(26_500, seed=20260722)
        self.assertEqual(modes, repeated)
        self.assertEqual(len(modes), 7_950)
        self.assertEqual(modes.count("defect_only"), 5_300)
        self.assertEqual(modes.count("vacuum_only"), 1_325)
        self.assertEqual(modes.count("defect_and_vacuum"), 1_325)

    def test_defect_h5_round_trip_preserves_masks_and_shared_pair_scale(self) -> None:
        image = self._synthetic_column_image()
        result = apply_defects(image, DefectConfig(seed=41))
        with tempfile.TemporaryDirectory() as directory:
            path = write_defect_h5(
                Path(directory) / "defects.h5",
                [image],
                [result.image],
                [result.masks],
                [result.metadata],
            )
            bases, defects, masks, metadata, attrs = read_defect_h5(path)
            self.assertEqual(bases.shape, (1, 128, 128))
            self.assertEqual(defects.shape, (1, 128, 128))
            self.assertEqual(attrs["schema_version"], "defect-image-h5-v1")
            self.assertEqual(set(masks), set(result.masks))
            for name in masks:
                np.testing.assert_array_equal(masks[name][0], result.masks[name])
            tolerance = metadata[0]["pair_dequantize_scale"] / 2.0 + 1e-7
            self.assertLessEqual(float(np.max(np.abs(bases[0] - image))), tolerance)
            self.assertLessEqual(float(np.max(np.abs(defects[0] - result.image))), tolerance)

    def test_defect_config_rejects_non_physical_factor_ranges(self) -> None:
        with self.assertRaises(ValueError):
            DefectConfig(seed=1, weak_factor_range=(0.8, 1.1)).validate()
        with self.assertRaises(ValueError):
            DefectConfig(seed=1, strong_factor_range=(0.9, 1.2)).validate()

    def test_structure_split_is_deterministic_balanced_and_leak_free(self) -> None:
        records = self._clean_master_records()
        config = SplitConfig(seed=73)
        plan = build_structure_split(records, config)
        repeated = build_structure_split(records, config)
        self.assertEqual(plan.assignments, repeated.assignments)
        validate_no_split_leakage(records, plan)
        counts = {split: plan.statistics[split]["count"] for split in ("train", "valid", "test")}
        self.assertEqual(sum(counts.values()), 40)
        self.assertLessEqual(abs(counts["train"] - 32), 2)
        self.assertLessEqual(abs(counts["valid"] - 4), 2)
        self.assertLessEqual(abs(counts["test"] - 4), 2)
        for split in ("train", "valid", "test"):
            variants = plan.statistics[split]["variant_counts"]
            self.assertGreater(variants.get("perfect", 0), 0)
            self.assertGreater(sum(value for key, value in variants.items() if key != "perfect"), 0)

    def test_structure_split_does_not_depend_on_masks(self) -> None:
        without_masks = self._clean_master_records(include_masks=False)
        with_masks = self._clean_master_records(include_masks=True)
        config = SplitConfig(seed=79)
        self.assertEqual(
            build_structure_split(without_masks, config).assignments,
            build_structure_split(with_masks, config).assignments,
        )

    def test_leakage_validator_rejects_sibling_split(self) -> None:
        records = self._clean_master_records()
        plan = build_structure_split(records, SplitConfig(seed=83))
        assignments = dict(plan.assignments)
        assignments["variant_000"] = "test" if assignments["perfect_000"] != "test" else "train"
        broken = SplitPlan(assignments, plan.ordered_image_ids, plan.config, plan.statistics)
        with self.assertRaisesRegex(ValueError, "leakage"):
            validate_no_split_leakage(records, broken)

    def test_clean_master_writer_omits_optional_masks(self) -> None:
        records = self._clean_master_records(include_masks=True)
        plan = build_structure_split(records, SplitConfig(seed=89))
        with tempfile.TemporaryDirectory() as directory:
            outputs = write_clean_master_bundle(
                Path(directory) / "clean_v1.h5", records, plan, include_masks=False
            )
            with h5py.File(outputs["h5"], "r") as handle:
                self.assertEqual(set(handle), {"train", "valid", "test"})
                self.assertEqual(
                    handle.attrs["schema_version"],
                    "atomic-stem-denoise-h5-v1",
                )
                self.assertAlmostEqual(
                    handle.attrs["dequantize_scale"],
                    1.0 / 32767.0,
                )
                self.assertEqual(handle.attrs["dequantize_offset"], 0.0)
                for split in ("train", "valid", "test"):
                    self.assertNotIn("masks", handle[split])
                    self.assertEqual(handle[f"{split}/images"].dtype, np.int16)
            self.assertEqual(sha256_file(outputs["h5"]), outputs["h5_sha256"])
            self.assertEqual(sha256_file(outputs["manifest"]), outputs["manifest_sha256"])

    def test_clean_master_writer_supports_masks_without_changing_split(self) -> None:
        records = self._clean_master_records(include_masks=True)
        plan = build_structure_split(records, SplitConfig(seed=97))
        with tempfile.TemporaryDirectory() as directory:
            outputs = write_clean_master_bundle(
                Path(directory) / "clean_v1.h5", records, plan, include_masks=True
            )
            with h5py.File(outputs["h5"], "r") as handle:
                for split in ("train", "valid", "test"):
                    self.assertIn("masks", handle[split])
                    self.assertEqual(handle[f"{split}/masks/defect"].dtype, np.uint8)
                    metadata = [json.loads(value) for value in handle[f"{split}/metadata/json"].asstr()[:]]
                    for index, item in enumerate(metadata):
                        if item["variant_type"] == "perfect":
                            self.assertFalse(handle[f"{split}/masks/defect"][index].any())


if __name__ == "__main__":
    unittest.main()
