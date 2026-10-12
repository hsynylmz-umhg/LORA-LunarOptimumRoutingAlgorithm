from pathlib import Path
import tempfile
import unittest

import numpy as np

from src.data.pds_reader import extract_lunar_patch, read_elevation_patch


class PdsReaderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def raster(self, *, name="ELEVATION", dtype="<f4", sample_type="PC_REAL",
               units="KM", scale="2", offset="1", extra="", values=None,
               projection="LOCAL_CARTESIAN"):
        values = np.arange(12).reshape(3, 4) if values is None else values
        image = self.root / "terrain.IMG"
        np.asarray(values, dtype=dtype).tofile(image)
        image.with_suffix(".LBL").write_text(
            f'PDS_VERSION_ID = PDS3\n^IMAGE = "terrain.IMG"\n'
            f'OBJECT = IMAGE\nNAME = {name}\nLINES = 3\nLINE_SAMPLES = 4\n'
            f'SAMPLE_TYPE = {sample_type}\nSAMPLE_BITS = {np.dtype(dtype).itemsize * 8}\n'
            f'UNIT = {units}\nSCALING_FACTOR = {scale}\nOFFSET = {offset}\n{extra}\n'
            'END_OBJECT = IMAGE\nMAP_SCALE = 0.118451 <KM/PIXEL>\n'
            f'MAP_PROJECTION_TYPE = {projection}\n'
            'A_AXIS_RADIUS = 1737.4 <KM>\nEND\n', encoding="utf-8")
        return image

    def test_label_dimensions_scaling_offset_units_and_spacing(self):
        image = self.raster()
        patch, metadata = read_elevation_patch(image, (3, 4), np.random.default_rng(1))
        np.testing.assert_array_equal(patch, (np.arange(12).reshape(3, 4) * 2 + 1) * 1000)
        self.assertAlmostEqual(metadata["sample_spacing_m"], 118.451)
        self.assertEqual((metadata["lines"], metadata["samples"]), (3, 4))
        self.assertEqual(metadata["units"], "m")

    def test_big_endian_integer_dem(self):
        image = self.raster(dtype=">i2", sample_type="MSB_INTEGER", units="M", scale=".5", offset="0")
        patch, _ = read_elevation_patch(image, (3, 4))
        np.testing.assert_array_equal(patch, np.arange(12).reshape(3, 4) * .5)

    def test_random_crop_is_reproducible(self):
        image = self.raster()
        a, ma = read_elevation_patch(image, 2, np.random.default_rng(3))
        b, mb = read_elevation_patch(image, 2, np.random.default_rng(3))
        np.testing.assert_array_equal(a, b)
        self.assertEqual((ma["row"], ma["col"]), (mb["row"], mb["col"]))

    def test_slope_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "SLOPE"):
            read_elevation_patch(self.raster(name="SLOPE", units="DEGREE"), 2)

    def test_nonlocal_or_unspecified_projection_requires_reprojection(self):
        for projection in ("SIMPLE CYLINDRICAL", "EQUIRECTANGULAR", "POLAR_STEREOGRAPHIC", "UNKNOWN"):
            with self.subTest(projection=projection), self.assertRaisesRegex(ValueError, "reprojected"):
                read_elevation_patch(self.raster(projection=projection), 2)
        image = self.raster()
        label = image.with_suffix(".LBL")
        label.write_text(label.read_text().replace("MAP_PROJECTION_TYPE = LOCAL_CARTESIAN\n", ""))
        with self.assertRaisesRegex(ValueError, "reprojected"):
            read_elevation_patch(image, 2)

    def test_float_missing_sentinel_matches_source_precision(self):
        values = np.zeros((3, 4))
        values[0, 0] = np.float32(-3.4028226550889045e38)
        with self.assertRaisesRegex(ValueError, "missing"):
            read_elevation_patch(self.raster(values=values, extra="MISSING_CONSTANT = -3.402822655e38"), (3, 4))

    def test_hex_missing_sentinel_matches_big_endian_source(self):
        values = np.zeros((3, 4))
        values[0, 0] = -9999
        sentinel = int.from_bytes(np.asarray(-9999, dtype=">f4").tobytes(), "big")
        with self.assertRaisesRegex(ValueError, "missing"):
            read_elevation_patch(self.raster(values=values, dtype=">f4", sample_type="IEEE_REAL",
                                            extra=f"MISSING_CONSTANT = 16#{sentinel:X}#"), (3, 4))

    def test_missing_label_unknown_units_and_bad_file_length(self):
        image = self.raster()
        image.with_suffix(".LBL").unlink()
        with self.assertRaisesRegex(ValueError, "label"):
            read_elevation_patch(image, 2)
        image = self.raster(units="UNKNOWN")
        with self.assertRaisesRegex(ValueError, "length unit"):
            read_elevation_patch(image, 2)
        image = self.raster()
        image.write_bytes(image.read_bytes()[:-4])
        with self.assertRaisesRegex(ValueError, "size mismatch"):
            read_elevation_patch(image, 2)

    def test_missing_and_nonfinite_samples_are_rejected(self):
        values = np.arange(12, dtype=float).reshape(3, 4)
        values[0, 0] = -9999
        with self.assertRaisesRegex(ValueError, "missing"):
            read_elevation_patch(self.raster(values=values, extra="MISSING_CONSTANT = -9999"), (3, 4))
        values[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            read_elevation_patch(self.raster(values=values), (3, 4))

    def test_radius_is_converted_to_elevation(self):
        values = np.full((3, 4), 1737.5)
        patch, _ = read_elevation_patch(self.raster(name="RADIUS", dtype="<f8", values=values,
                                                   scale="1", offset="0"), (3, 4))
        np.testing.assert_allclose(patch, 100.0)

    def test_export_bounds_and_no_partial_output_on_error(self):
        image = self.raster()
        output = self.root / "output.csv"
        with self.assertRaisesRegex(ValueError, "beyond"):
            extract_lunar_patch(image, output, start_row=2, start_col=3, patch_size=2)
        self.assertFalse(output.exists())
        metadata = extract_lunar_patch(image, output, start_row=1, start_col=2, patch_size=2)
        self.assertEqual(np.loadtxt(output).shape, (2, 2))
        self.assertEqual((metadata["row"], metadata["col"]), (1, 2))


if __name__ == "__main__":
    unittest.main()
