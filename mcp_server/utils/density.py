"""
Shared utilities for density expression handling.

Used by both solve.py and create_mesh.py for consistent
expression parsing, validation, and coordinate handling.
"""

import numpy as np
from typing import Any, Callable


# Safe namespace for expression evaluation
SAFE_NAMESPACE = {
    "np": np,
    "sqrt": np.sqrt,
    "exp": np.exp,
    "log": np.log,
    "log10": np.log10,
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "abs": np.abs,
    "pow": pow,
    "pi": np.pi,
}

# Geometries where r should mean spherical radius (distance from origin)
# rather than cylindrical radius (distance from z-axis)
SPHERICAL_GEOMETRIES = {
    "sphere_in_vacuum",
    "sphere_domain",
    "sphere_near_wall",
    "two_spheres",
    "shell_in_vacuum",
}

# Supported file extensions for density profiles
SUPPORTED_EXTENSIONS = {".txt", ".dat", ".csv", ".npy", ".npz"}


def _load_density_file(file_path: str, skip_header: int = 0, npz_key: str = "data") -> np.ndarray:
    """
    Load density profile data from various file formats.

    Supported formats:
        - .txt, .dat: Whitespace-separated text (numpy.loadtxt)
        - .csv: Comma-separated values
        - .npy: NumPy binary format
        - .npz: NumPy compressed format (uses npz_key to select array)

    Args:
        file_path: Path to the data file
        skip_header: Number of header rows to skip (text formats only)
        npz_key: Key to use for .npz files (default: "data")

    Returns:
        2D numpy array with shape (n_points, n_columns)

    Raises:
        ValueError: If file format is unsupported or data is invalid
        FileNotFoundError: If file doesn't exist
    """
    import os
    ext = os.path.splitext(file_path)[1].lower()

    if ext in {".txt", ".dat"}:
        # Whitespace-separated text
        data = np.loadtxt(file_path, skiprows=skip_header)

    elif ext == ".csv":
        # Comma-separated values
        data = np.loadtxt(file_path, delimiter=",", skiprows=skip_header)

    elif ext == ".npy":
        # NumPy binary format
        data = np.load(file_path)
        if skip_header > 0:
            data = data[skip_header:]

    elif ext == ".npz":
        # NumPy compressed format
        with np.load(file_path) as npz:
            if npz_key not in npz:
                available = list(npz.keys())
                raise ValueError(
                    f"Key '{npz_key}' not found in .npz file. "
                    f"Available keys: {available}. "
                    f"Use 'npz_key' parameter to specify the correct key."
                )
            data = npz[npz_key]
        if skip_header > 0:
            data = data[skip_header:]

    else:
        raise ValueError(
            f"Unsupported file format '{ext}'. "
            f"Supported formats: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    return data


def get_coordinate_info(symmetry: str, geometry: str = None, dimension: int = 2) -> dict:
    """
    Get coordinate variable information for a given symmetry/geometry.

    Returns dict with:
        - 'variables': dict of variable names to descriptions
        - 'test_values': dict of variable names to test values for validation
        - 'is_spherical': whether r means spherical radius
    """
    is_spherical = geometry in SPHERICAL_GEOMETRIES

    if symmetry == "axial":
        if is_spherical:
            variables = {
                "r": "spherical radius (distance from origin)",
                "r_cyl": "cylindrical radius (distance from z-axis)",
                "z": "height along symmetry axis",
            }
            test_values = {"r": 0.5, "r_cyl": 0.3, "z": 0.4}
        else:
            variables = {
                "r": "cylindrical radius (distance from z-axis)",
                "R": "spherical radius (distance from origin)",
                "z": "height along symmetry axis",
            }
            test_values = {"r": 0.3, "R": 0.5, "z": 0.4}
    else:
        if dimension == 3:
            variables = {
                "x": "x coordinate",
                "y": "y coordinate",
                "z": "z coordinate",
                "r": "cylindrical radius sqrt(x^2 + y^2)",
            }
            test_values = {"x": 0.3, "y": 0.4, "z": 0.2, "r": 0.5}
        else:
            variables = {
                "x": "x coordinate",
                "y": "y coordinate",
                "r": "cylindrical radius sqrt(x^2 + y^2)",
            }
            test_values = {"x": 0.3, "y": 0.4, "r": 0.5}

    return {
        "variables": variables,
        "test_values": test_values,
        "is_spherical": is_spherical,
    }


def validate_expression(
    expr: str,
    symmetry: str,
    geometry: str = None,
    dimension: int = 2,
) -> None:
    """
    Validate a density expression by evaluating at a test point.

    Raises ValueError with helpful message if expression is invalid.
    """
    coord_info = get_coordinate_info(symmetry, geometry, dimension)
    test_vars = {**SAFE_NAMESPACE, **coord_info["test_values"]}

    try:
        eval(expr, {"__builtins__": {}}, test_vars)
    except NameError as e:
        var_descriptions = [f"{k} ({v})" for k, v in coord_info["variables"].items()]
        raise ValueError(
            f"Invalid expression '{expr}': {e}. "
            f"Available variables: {', '.join(var_descriptions)}. "
            f"Available functions: sqrt, exp, log, log10, sin, cos, tan, abs, pow, pi."
        )
    except Exception as e:
        raise ValueError(f"Invalid expression '{expr}': {e}")


def build_local_vars(
    x: np.ndarray,
    symmetry: str,
    geometry: str = None,
) -> dict:
    """
    Build local variable dict for expression evaluation at point x.

    Args:
        x: Coordinate array [x0, x1, ...] from FEniCS
        symmetry: Mesh symmetry ('axial' or 'none')
        geometry: Geometry type (affects coordinate interpretation)

    Returns:
        Dict of variable names to values for eval()
    """
    local_vars = SAFE_NAMESPACE.copy()
    is_spherical = geometry in SPHERICAL_GEOMETRIES

    if symmetry == "axial":
        r_cyl = x[0]  # cylindrical radius
        z = x[1]
        r_sph = np.sqrt(r_cyl**2 + z**2)  # spherical radius

        if is_spherical:
            local_vars["r"] = r_sph
            local_vars["r_cyl"] = r_cyl
        else:
            local_vars["r"] = r_cyl
            local_vars["R"] = r_sph
        local_vars["z"] = z
    else:
        local_vars["x"] = x[0]
        local_vars["y"] = x[1]
        if len(x) > 2:
            local_vars["z"] = x[2]
        local_vars["r"] = np.sqrt(x[0]**2 + x[1]**2)

    return local_vars


def create_density_function(
    spec: Any,
    symmetry: str,
    dimension: int,
    geometry: str = None,
) -> Callable:
    """
    Create a density function from a specification.

    Args:
        spec: Density specification - number, dict with 'expression', or dict with 'file'
        symmetry: Mesh symmetry ('axial' or 'none')
        dimension: Mesh dimension (2 or 3)
        geometry: Geometry type - affects coordinate interpretation for expressions

    Returns:
        Function that takes (x) and returns density
    """
    # Constant density
    if isinstance(spec, (int, float)):
        rho = float(spec)
        return lambda x: rho

    # Expression-based density
    if isinstance(spec, dict) and "expression" in spec:
        expr_str = spec["expression"]

        # Validate expression upfront
        validate_expression(expr_str, symmetry, geometry, dimension)

        def expr_func(x):
            local_vars = build_local_vars(x, symmetry, geometry)
            return eval(expr_str, {"__builtins__": {}}, local_vars)

        return expr_func

    # File-based (tabulated) density
    if isinstance(spec, dict) and "file" in spec:
        file_path = spec["file"]
        skip_header = spec.get("skip_header", 0)
        npz_key = spec.get("npz_key", "data")  # Key for .npz files

        # Load data based on file extension
        data = _load_density_file(file_path, skip_header, npz_key)

        # Ensure 2D array
        if data.ndim == 1:
            raise ValueError(
                f"File '{file_path}' contains only 1D data. "
                "Need at least 2 columns (position, density)."
            )
        n_cols = data.shape[1]

        is_spherical = geometry in SPHERICAL_GEOMETRIES

        # Auto-detect column format based on symmetry and columns
        if symmetry == "axial":
            if n_cols == 2:
                # (r, rho) - 1D radial profile
                from scipy.interpolate import interp1d
                r_data = data[:, 0]
                rho_data = data[:, 1]
                interp = interp1d(r_data, rho_data, bounds_error=False,
                                  fill_value=(rho_data[0], rho_data[-1]))

                def tabulated_func(x):
                    if is_spherical:
                        r = np.sqrt(x[0]**2 + x[1]**2)  # spherical
                    else:
                        r = x[0]  # cylindrical
                    return float(interp(r))

                return tabulated_func

            elif n_cols == 3:
                # (r, z, rho) - 2D axisymmetric profile
                from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
                r_data = data[:, 0]
                z_data = data[:, 1]
                rho_data = data[:, 2]
                points = np.column_stack([r_data, z_data])
                interp = LinearNDInterpolator(points, rho_data)
                nearest = NearestNDInterpolator(points, rho_data)

                def tabulated_func(x):
                    r, z = x[0], x[1]
                    val = interp(r, z)
                    if np.isnan(val):
                        val = nearest(r, z)
                    return float(val)

                return tabulated_func

        else:  # none symmetry
            if n_cols == 3 and dimension == 2:
                # (x, y, rho) - 2D Cartesian
                from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
                points = data[:, :2]
                rho_data = data[:, 2]
                interp = LinearNDInterpolator(points, rho_data)
                nearest = NearestNDInterpolator(points, rho_data)

                def tabulated_func(x):
                    val = interp(x[0], x[1])
                    if np.isnan(val):
                        val = nearest(x[0], x[1])
                    return float(val)

                return tabulated_func

            elif n_cols == 4 and dimension == 3:
                # (x, y, z, rho) - 3D Cartesian
                from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
                points = data[:, :3]
                rho_data = data[:, 3]
                interp = LinearNDInterpolator(points, rho_data)
                nearest = NearestNDInterpolator(points, rho_data)

                def tabulated_func(x):
                    val = interp(x[0], x[1], x[2])
                    if np.isnan(val):
                        val = nearest(x[0], x[1], x[2])
                    return float(val)

                return tabulated_func

        raise ValueError(
            f"Unsupported column count {n_cols} for symmetry '{symmetry}' and dimension {dimension}"
        )

    raise ValueError(f"Invalid density specification: {spec}")


def extract_density_value(
    spec: Any,
    symmetry: str = "axial",
    geometry: str = None,
) -> float | None:
    """
    Extract a representative density value from various formats.

    For mesh refinement, we need a single number to compute lambda.
    For non-numeric values, we extract/estimate the maximum density.

    Args:
        spec: Density specification - number, dict with 'expression', or dict with 'file'
        symmetry: Mesh symmetry ('axial' or 'none')
        geometry: Geometry type - affects coordinate interpretation

    Returns:
        Representative density value, or None if cannot extract
    """
    if isinstance(spec, (int, float)):
        return float(spec)

    if isinstance(spec, dict):
        if "expression" in spec:
            expr = spec["expression"]

            # Validate expression first
            try:
                validate_expression(expr, symmetry, geometry)
            except ValueError:
                return None

            # Sample at grid of points and take max
            coord_info = get_coordinate_info(symmetry, geometry)
            is_spherical = coord_info["is_spherical"]

            max_rho = 0.0
            r_vals = np.linspace(0.001, 1.0, 20)
            z_vals = np.linspace(-1.0, 1.0, 20)

            for r_cyl in r_vals:
                for z in z_vals:
                    local_vars = SAFE_NAMESPACE.copy()
                    r_sph = np.sqrt(r_cyl**2 + z**2)

                    if symmetry == "axial":
                        if is_spherical:
                            local_vars["r"] = r_sph
                            local_vars["r_cyl"] = r_cyl
                        else:
                            local_vars["r"] = r_cyl
                            local_vars["R"] = r_sph
                        local_vars["z"] = z
                    else:
                        local_vars["x"] = r_cyl
                        local_vars["y"] = z
                        local_vars["r"] = np.sqrt(r_cyl**2 + z**2)

                    try:
                        val = eval(expr, {"__builtins__": {}}, local_vars)
                        if isinstance(val, (int, float)) and val > max_rho:
                            max_rho = val
                    except Exception:
                        pass

            return max_rho if max_rho > 0 else None

        elif "file" in spec:
            # Load profile from file and take max
            try:
                skip_header = spec.get("skip_header", 0)
                data = np.loadtxt(spec["file"], skiprows=skip_header)
                # Assume density is in second column (first is position)
                if data.ndim == 1:
                    return float(np.max(data))
                else:
                    return float(np.max(data[:, 1]))
            except Exception:
                return None

    return None
