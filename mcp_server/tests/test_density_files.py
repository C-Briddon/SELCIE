#!/usr/bin/env python3
"""Tests for file-based density profiles."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest

from utils.density import create_density_function, extract_density_value

# Path to test data files
TEST_DATA_DIR = Path(__file__).parent / "test_data"


class TestRadialProfile2Column:
    """Tests for 2-column radial density profiles (r, rho)."""

    def test_load_radial_profile(self):
        """Test loading a basic 2-column radial profile."""
        file_path = TEST_DATA_DIR / "radial_profile_2col.txt"
        func = create_density_function(
            {"file": str(file_path)},
            symmetry="axial",
            dimension=2,
            geometry="sphere_in_vacuum"
        )

        # Test at r=0 (center): should be 1e6
        # For spherical geometry, r = sqrt(r_cyl^2 + z^2)
        result = func([0.0, 0.0])
        assert result == pytest.approx(1e6, rel=0.01)

    def test_radial_profile_spherical_geometry(self):
        """Test that r is spherical radius for spherical geometries."""
        file_path = TEST_DATA_DIR / "radial_profile_2col.txt"
        func = create_density_function(
            {"file": str(file_path)},
            symmetry="axial",
            dimension=2,
            geometry="sphere_in_vacuum"
        )

        # Point at r_cyl=0.3, z=0.4 -> r_spherical = 0.5
        # At r=0.5, the profile gives 10.0
        result = func([0.3, 0.4])
        assert result == pytest.approx(10.0, rel=0.01)

    def test_radial_profile_cylindrical_geometry(self):
        """Test that r is cylindrical radius for non-spherical geometries."""
        file_path = TEST_DATA_DIR / "radial_profile_2col.txt"
        func = create_density_function(
            {"file": str(file_path)},
            symmetry="axial",
            dimension=2,
            geometry="ellipse_in_vacuum"  # Non-spherical
        )

        # Point at r_cyl=0.5, z=0.4 -> r = 0.5 (cylindrical)
        # At r=0.5, the profile gives 10.0
        result = func([0.5, 0.4])
        assert result == pytest.approx(10.0, rel=0.01)

        # Compare: for spherical geometry same point would give different result
        # r_spherical = sqrt(0.5^2 + 0.4^2) = 0.64, which interpolates differently

    def test_radial_profile_interpolation(self):
        """Test linear interpolation between data points."""
        file_path = TEST_DATA_DIR / "linear_profile.txt"
        func = create_density_function(
            {"file": str(file_path)},
            symmetry="axial",
            dimension=2,
            geometry="sphere_in_vacuum"
        )

        # Linear profile: rho = 100 - 100*r
        # At r=0.25, should be 75.0
        result = func([0.25, 0.0])  # r_spherical = 0.25
        assert result == pytest.approx(75.0, rel=0.01)

        # At r=0.75, should be 25.0
        result = func([0.75, 0.0])
        assert result == pytest.approx(25.0, rel=0.01)

    def test_radial_profile_extrapolation(self):
        """Test extrapolation uses boundary values."""
        file_path = TEST_DATA_DIR / "linear_profile.txt"
        func = create_density_function(
            {"file": str(file_path)},
            symmetry="axial",
            dimension=2,
            geometry="sphere_in_vacuum"
        )

        # Beyond r=1.0, should use last value (0.0)
        result = func([1.5, 0.0])
        assert result == pytest.approx(0.0, abs=0.01)

        # Before r=0.0 (can't really happen, but test boundary)
        # Should use first value (100.0)

    def test_skip_header(self):
        """Test skip_header parameter."""
        file_path = TEST_DATA_DIR / "radial_profile_with_header.txt"
        func = create_density_function(
            {"file": str(file_path), "skip_header": 2},
            symmetry="axial",
            dimension=2,
            geometry="sphere_in_vacuum"
        )

        # First data row after header: r=0.0, rho=1000.0
        result = func([0.0, 0.0])
        assert result == pytest.approx(1000.0, rel=0.01)

    def test_tab_separated_file(self):
        """Test loading tab-separated file."""
        file_path = TEST_DATA_DIR / "tab_separated.txt"
        func = create_density_function(
            {"file": str(file_path)},
            symmetry="axial",
            dimension=2,
            geometry="sphere_in_vacuum"
        )

        # At r=0, should be 1e6
        result = func([0.0, 0.0])
        assert result == pytest.approx(1e6, rel=0.01)


class TestAxisymmetricProfile3Column:
    """Tests for 3-column axisymmetric profiles (r, z, rho)."""

    def test_load_axisymmetric_profile(self):
        """Test loading a 3-column axisymmetric profile."""
        file_path = TEST_DATA_DIR / "axisymmetric_profile_3col.txt"
        func = create_density_function(
            {"file": str(file_path)},
            symmetry="axial",
            dimension=2,
            geometry="ellipse_in_vacuum"
        )

        # At r=0, z=0: should be 1e6
        result = func([0.0, 0.0])
        assert result == pytest.approx(1e6, rel=0.01)

    def test_axisymmetric_interpolation(self):
        """Test 2D interpolation in axisymmetric profile."""
        file_path = TEST_DATA_DIR / "axisymmetric_profile_3col.txt"
        func = create_density_function(
            {"file": str(file_path)},
            symmetry="axial",
            dimension=2,
            geometry="ellipse_in_vacuum"
        )

        # Test at grid points
        assert func([0.1, 0.1]) == pytest.approx(1e5, rel=0.01)
        assert func([0.2, 0.2]) == pytest.approx(1e3, rel=0.01)

        # Test interpolation between points
        # At r=0.05, z=0.05 (midpoint of four corners)
        result = func([0.05, 0.05])
        # Should be somewhere between 1e5 and 1e6
        assert 1e5 <= result <= 1e6

    def test_axisymmetric_nearest_fallback(self):
        """Test NearestNDInterpolator fallback outside convex hull."""
        file_path = TEST_DATA_DIR / "axisymmetric_profile_3col.txt"
        func = create_density_function(
            {"file": str(file_path)},
            symmetry="axial",
            dimension=2,
            geometry="ellipse_in_vacuum"
        )

        # Outside the data range - should use nearest neighbor
        result = func([1.0, 1.0])
        # Nearest point is (0.5, 0.2) with rho=1.0
        assert result == pytest.approx(1.0, rel=0.1)


class TestCartesian2DProfile:
    """Tests for 3-column Cartesian 2D profiles (x, y, rho)."""

    def test_load_cartesian_2d_profile(self):
        """Test loading a 2D Cartesian profile."""
        file_path = TEST_DATA_DIR / "cartesian_2d_profile.txt"
        func = create_density_function(
            {"file": str(file_path)},
            symmetry="none",
            dimension=2,
            geometry=None
        )

        # At origin: should be 1e6
        result = func([0.0, 0.0])
        assert result == pytest.approx(1e6, rel=0.01)

        # At corners: should be 1.0
        result = func([0.5, 0.5])
        assert result == pytest.approx(1.0, rel=0.01)

    def test_cartesian_2d_interpolation(self):
        """Test 2D interpolation in Cartesian profile."""
        file_path = TEST_DATA_DIR / "cartesian_2d_profile.txt"
        func = create_density_function(
            {"file": str(file_path)},
            symmetry="none",
            dimension=2,
            geometry=None
        )

        # Interpolation between center (1e6) and edge (1.0)
        result = func([0.25, 0.0])
        # Should be between 1 and 1e6
        assert 1.0 < result < 1e6


class TestCartesian3DProfile:
    """Tests for 4-column Cartesian 3D profiles (x, y, z, rho)."""

    def test_load_cartesian_3d_profile(self):
        """Test loading a 3D Cartesian profile."""
        file_path = TEST_DATA_DIR / "cartesian_3d_profile.txt"
        func = create_density_function(
            {"file": str(file_path)},
            symmetry="none",
            dimension=3,
            geometry=None
        )

        # At origin: should be 1e6
        result = func([0.0, 0.0, 0.0])
        assert result == pytest.approx(1e6, rel=0.01)

        # At far corner: should be 1.0
        result = func([0.5, 0.5, 0.5])
        assert result == pytest.approx(1.0, rel=0.01)

    def test_cartesian_3d_interpolation(self):
        """Test 3D interpolation."""
        file_path = TEST_DATA_DIR / "cartesian_3d_profile.txt"
        func = create_density_function(
            {"file": str(file_path)},
            symmetry="none",
            dimension=3,
            geometry=None
        )

        # Midpoint along x-axis: between 1e6 and 1e3
        result = func([0.25, 0.0, 0.0])
        assert 1e3 < result < 1e6


class TestExtractDensityValue:
    """Tests for extract_density_value with file-based profiles."""

    def test_extract_from_radial_file(self):
        """Test extracting max density from radial profile file."""
        file_path = TEST_DATA_DIR / "radial_profile_2col.txt"
        result = extract_density_value(
            {"file": str(file_path)},
            symmetry="axial",
            geometry="sphere_in_vacuum"
        )

        # Max density in file is 1e6
        assert result == pytest.approx(1e6, rel=0.01)

    def test_extract_from_linear_file(self):
        """Test extracting max density from linear profile."""
        file_path = TEST_DATA_DIR / "linear_profile.txt"
        result = extract_density_value(
            {"file": str(file_path)},
            symmetry="axial",
            geometry="sphere_in_vacuum"
        )

        # Max density in file is 100.0
        assert result == pytest.approx(100.0, rel=0.01)


class TestErrorHandling:
    """Tests for error handling with file-based profiles."""

    def test_file_not_found(self):
        """Test error when file doesn't exist."""
        with pytest.raises(Exception):  # FileNotFoundError or OSError
            create_density_function(
                {"file": "/nonexistent/path/file.txt"},
                symmetry="axial",
                dimension=2,
                geometry="sphere_in_vacuum"
            )

    def test_invalid_column_count(self):
        """Test error with wrong number of columns for symmetry."""
        # Create a temporary file with wrong columns (need 2+ rows for 2D array)
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("0.0 1.0 2.0 3.0 4.0\n")  # 5 columns - invalid
            f.write("0.1 1.1 2.1 3.1 4.1\n")  # Need 2 rows for numpy to make 2D array
            temp_path = f.name

        try:
            with pytest.raises(ValueError, match="Unsupported column count"):
                create_density_function(
                    {"file": temp_path},
                    symmetry="axial",
                    dimension=2,
                    geometry="sphere_in_vacuum"
                )
        finally:
            Path(temp_path).unlink()

    def test_skip_header_too_large(self):
        """Test error when skip_header skips all data."""
        file_path = TEST_DATA_DIR / "linear_profile.txt"
        with pytest.raises(Exception):  # Will fail to load or create interpolator
            func = create_density_function(
                {"file": str(file_path), "skip_header": 100},
                symmetry="axial",
                dimension=2,
                geometry="sphere_in_vacuum"
            )
            # Try to use it - should fail
            func([0.0, 0.0])


class TestGeometryConsistency:
    """Tests ensuring geometry affects coordinate interpretation correctly."""

    def test_same_file_different_geometries(self):
        """Test that same file gives different results for different geometries."""
        file_path = TEST_DATA_DIR / "radial_profile_2col.txt"

        # Spherical geometry: r = sqrt(r_cyl^2 + z^2)
        func_spherical = create_density_function(
            {"file": str(file_path)},
            symmetry="axial",
            dimension=2,
            geometry="sphere_in_vacuum"
        )

        # Non-spherical geometry: r = r_cyl
        func_cylindrical = create_density_function(
            {"file": str(file_path)},
            symmetry="axial",
            dimension=2,
            geometry="cylinder_in_vacuum"
        )

        # At point r_cyl=0.3, z=0.4:
        # - Spherical: r = 0.5
        # - Cylindrical: r = 0.3
        point = [0.3, 0.4]

        result_spherical = func_spherical(point)
        result_cylindrical = func_cylindrical(point)

        # Results should be different
        assert result_spherical != pytest.approx(result_cylindrical, rel=0.1)

        # Spherical r=0.5 gives 10.0
        assert result_spherical == pytest.approx(10.0, rel=0.01)

        # Cylindrical r=0.3 interpolates between r=0.2 (1e3) and r=0.5 (10)
        assert 10 < result_cylindrical < 1e3


class TestCSVFormat:
    """Tests for CSV file format."""

    def test_load_csv_radial_profile(self):
        """Test loading a CSV radial profile."""
        file_path = TEST_DATA_DIR / "radial_profile.csv"
        func = create_density_function(
            {"file": str(file_path), "skip_header": 1},  # Skip comment line
            symmetry="axial",
            dimension=2,
            geometry="sphere_in_vacuum"
        )

        # At r=0: should be 1e6
        result = func([0.0, 0.0])
        assert result == pytest.approx(1e6, rel=0.01)

        # At r=0.5: should be 10.0
        result = func([0.5, 0.0])
        assert result == pytest.approx(10.0, rel=0.01)

    def test_csv_interpolation(self):
        """Test interpolation with CSV data."""
        file_path = TEST_DATA_DIR / "radial_profile.csv"
        func = create_density_function(
            {"file": str(file_path), "skip_header": 1},
            symmetry="axial",
            dimension=2,
            geometry="sphere_in_vacuum"
        )

        # Interpolation between r=0.5 (10.0) and r=0.8 (1.0)
        result = func([0.65, 0.0])
        assert 1.0 < result < 10.0


class TestNumpyBinaryFormat:
    """Tests for NumPy binary (.npy) file format."""

    def test_load_npy_radial_profile(self):
        """Test loading a .npy radial profile."""
        file_path = TEST_DATA_DIR / "radial_profile.npy"
        func = create_density_function(
            {"file": str(file_path)},
            symmetry="axial",
            dimension=2,
            geometry="sphere_in_vacuum"
        )

        # At r=0: should be 1e6
        result = func([0.0, 0.0])
        assert result == pytest.approx(1e6, rel=0.01)

        # At r=0.5: should be 10.0
        result = func([0.5, 0.0])
        assert result == pytest.approx(10.0, rel=0.01)

    def test_npy_axisymmetric_profile(self):
        """Test loading a 3-column .npy axisymmetric profile."""
        file_path = TEST_DATA_DIR / "axisymmetric_profile.npy"
        func = create_density_function(
            {"file": str(file_path)},
            symmetry="axial",
            dimension=2,
            geometry="ellipse_in_vacuum"
        )

        # At r=0, z=0: should be 1e6
        result = func([0.0, 0.0])
        assert result == pytest.approx(1e6, rel=0.01)

        # At r=0.1, z=0.1: should be 1e5
        result = func([0.1, 0.1])
        assert result == pytest.approx(1e5, rel=0.01)

    def test_npy_with_skip_rows(self):
        """Test skip_header with .npy files."""
        file_path = TEST_DATA_DIR / "radial_profile.npy"
        func = create_density_function(
            {"file": str(file_path), "skip_header": 2},  # Skip first 2 rows
            symmetry="axial",
            dimension=2,
            geometry="sphere_in_vacuum"
        )

        # After skipping 2 rows, first data point is r=0.10, rho=1e5
        result = func([0.10, 0.0])
        assert result == pytest.approx(1e5, rel=0.01)


class TestNumpyCompressedFormat:
    """Tests for NumPy compressed (.npz) file format."""

    def test_load_npz_default_key(self):
        """Test loading .npz with default key 'data'."""
        file_path = TEST_DATA_DIR / "radial_profile.npz"
        func = create_density_function(
            {"file": str(file_path)},  # Uses default key "data"
            symmetry="axial",
            dimension=2,
            geometry="sphere_in_vacuum"
        )

        # At r=0: should be 1e6
        result = func([0.0, 0.0])
        assert result == pytest.approx(1e6, rel=0.01)

    def test_load_npz_custom_key(self):
        """Test loading .npz with custom key."""
        file_path = TEST_DATA_DIR / "radial_profile_custom_key.npz"
        func = create_density_function(
            {"file": str(file_path), "npz_key": "density_profile"},
            symmetry="axial",
            dimension=2,
            geometry="sphere_in_vacuum"
        )

        # At r=0: should be 1e6
        result = func([0.0, 0.0])
        assert result == pytest.approx(1e6, rel=0.01)

    def test_npz_wrong_key_error(self):
        """Test error when .npz key doesn't exist."""
        file_path = TEST_DATA_DIR / "radial_profile.npz"
        with pytest.raises(ValueError, match="Key.*not found"):
            create_density_function(
                {"file": str(file_path), "npz_key": "wrong_key"},
                symmetry="axial",
                dimension=2,
                geometry="sphere_in_vacuum"
            )


class TestDatFormat:
    """Tests for .dat file format (whitespace-separated like .txt)."""

    def test_load_dat_radial_profile(self):
        """Test loading a .dat radial profile."""
        file_path = TEST_DATA_DIR / "radial_profile.dat"
        func = create_density_function(
            {"file": str(file_path)},
            symmetry="axial",
            dimension=2,
            geometry="sphere_in_vacuum"
        )

        # At r=0: should be 1e6
        result = func([0.0, 0.0])
        assert result == pytest.approx(1e6, rel=0.01)


class TestUnsupportedFormat:
    """Tests for unsupported file formats."""

    def test_unsupported_extension_error(self):
        """Test error for unsupported file extension."""
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xlsx', delete=False) as f:
            f.write("dummy")
            temp_path = f.name

        try:
            with pytest.raises(ValueError, match="Unsupported file format"):
                create_density_function(
                    {"file": temp_path},
                    symmetry="axial",
                    dimension=2,
                    geometry="sphere_in_vacuum"
                )
        finally:
            Path(temp_path).unlink()

    def test_error_message_lists_supported_formats(self):
        """Test that error message lists supported formats."""
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("{}")
            temp_path = f.name

        try:
            with pytest.raises(ValueError) as exc_info:
                create_density_function(
                    {"file": temp_path},
                    symmetry="axial",
                    dimension=2,
                    geometry="sphere_in_vacuum"
                )
            error_msg = str(exc_info.value)
            assert ".csv" in error_msg
            assert ".npy" in error_msg
            assert ".npz" in error_msg
            assert ".txt" in error_msg
        finally:
            Path(temp_path).unlink()


class TestFormatConsistency:
    """Tests ensuring all formats produce consistent results."""

    def test_txt_csv_npy_npz_consistency(self):
        """Test that all formats give the same results for the same data."""
        # Load from each format
        func_txt = create_density_function(
            {"file": str(TEST_DATA_DIR / "radial_profile_2col.txt")},
            symmetry="axial",
            dimension=2,
            geometry="sphere_in_vacuum"
        )
        func_csv = create_density_function(
            {"file": str(TEST_DATA_DIR / "radial_profile.csv"), "skip_header": 1},
            symmetry="axial",
            dimension=2,
            geometry="sphere_in_vacuum"
        )
        func_npy = create_density_function(
            {"file": str(TEST_DATA_DIR / "radial_profile.npy")},
            symmetry="axial",
            dimension=2,
            geometry="sphere_in_vacuum"
        )
        func_npz = create_density_function(
            {"file": str(TEST_DATA_DIR / "radial_profile.npz")},
            symmetry="axial",
            dimension=2,
            geometry="sphere_in_vacuum"
        )
        func_dat = create_density_function(
            {"file": str(TEST_DATA_DIR / "radial_profile.dat")},
            symmetry="axial",
            dimension=2,
            geometry="sphere_in_vacuum"
        )

        # Test at multiple points
        test_points = [[0.0, 0.0], [0.1, 0.0], [0.3, 0.4], [0.5, 0.0]]

        for point in test_points:
            result_txt = func_txt(point)
            result_csv = func_csv(point)
            result_npy = func_npy(point)
            result_npz = func_npz(point)
            result_dat = func_dat(point)

            # All should be approximately equal
            assert result_csv == pytest.approx(result_txt, rel=0.01), f"CSV mismatch at {point}"
            assert result_npy == pytest.approx(result_txt, rel=0.01), f"NPY mismatch at {point}"
            assert result_npz == pytest.approx(result_txt, rel=0.01), f"NPZ mismatch at {point}"
            assert result_dat == pytest.approx(result_txt, rel=0.01), f"DAT mismatch at {point}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
