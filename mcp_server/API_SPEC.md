# SELCIE MCP Server API Specification

## Overview

This document specifies the MCP (Model Context Protocol) server interface for SELCIE, enabling LLMs to perform chameleon field calculations and physics exploration.

**Version:** 1.0.0
**Protocol:** MCP over stdio
**Backend:** SELCIE (FEniCS-based chameleon field solver)

---

## Design Principles

1. **Physics-first**: Tools provide physical insight, not just numerical output
2. **Regime awareness**: Automatically detect adiabatic/transition/screened regimes
3. **Reusability**: Meshes and profiles can be reused across multiple solves
4. **Interpretability**: Include natural language explanations in responses
5. **Validation**: Compare to analytic solutions where available

---

## Session State

The server maintains session state with the following objects:

```
Session
├── meshes: dict[mesh_id -> Mesh]
├── solutions: dict[solution_id -> Solution]
└── metadata: dict
```

Object IDs are auto-generated (e.g., `mesh_001`, `solution_001`) unless user specifies a custom ID.

---

## Tools

### 1. `calculate_physical_parameters`

Convert physical parameters to SELCIE's dimensionless form and assess the screening regime.

This is the recommended entry point before running any simulation. It determines whether SELCIE
is needed or if analytic solutions suffice.

#### Parameters

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `beta` | number | Yes | Matter coupling strength β (M = M_pl/β) |
| `Lambda_eV` | number | No | Energy scale Λ in eV. Default: 2.4e-3 (dark energy scale) |
| `n` | integer | No | Potential power. Default: 1 |
| `rho_0` | number | Yes | Characteristic density scale (e.g., central density). Enters α as α ∝ 1/ρ₀ |
| `rho_0_units` | string | Yes | Units: `"g/cm^3"`, `"kg/m^3"`, `"eV^4"`, `"M_sun/kpc^3"` |
| `L` | number | Yes | Characteristic length scale (e.g., object radius). Enters α as α ∝ 1/L² |
| `L_units` | string | Yes | Units: `"m"`, `"cm"`, `"km"`, `"R_sun"`, `"kpc"`, `"Mpc"`, `"AU"` |
| `rho_max` | number | No | Maximum density in system (for regime estimation), in same units as rho_0 |
| `rho_min` | number | No | Minimum density in system (for regime estimation), in same units as rho_0 |

#### Physics Background

SELCIE solves the dimensionless chameleon field equation:
```
α ∇²φ + φ^{-(n+1)} = ρ
```

The dimensionless α parameter encapsulates all physical parameters:
```
α = (M·Λ)/(ρ₀·L²) × [(n·M·Λ³)/ρ₀]^{1/(n+1)}
```

where M = M_pl/β.

The **dimensionless Compton wavelength** determines field behavior (from SELCIE paper arXiv:2110.11917):
```
λ̂²(ρ̂) = α/(n+1) × ρ̂^{-(n+2)/(n+1)}
```

- **λ̂ << 1**: Field tightly tracks local minimum → φ ≈ ρ̂^{-1/(n+1)} (adiabatic)
- **λ̂ >> 1**: Field cannot respond to local density → boundary conditions dominate
- **λ̂ ~ 1**: Transition region where screening dynamics occur

#### Returns

**If rho_max and rho_min are NOT provided:**

```json
{
  "alpha": 3.5,
  "n": 1,

  "formulas": {
    "phi_min": "φ_min(ρ̂) = ρ̂^{-1/(n+1)} = ρ̂^{-0.5}",
    "lambda": "λ̂(ρ̂) = √(α/(n+1)) × ρ̂^{-(n+2)/(2(n+1))} = 1.32 × ρ̂^{-0.75}"
  },

  "regime": null,
  "regime_note": "Provide rho_max and rho_min for regime estimation"
}
```

**If rho_max and rho_min ARE provided:**

```json
{
  "alpha": 3.5,
  "n": 1,

  "at_rho_max": {
    "rho": 1e6,
    "phi_min": 0.001,
    "lambda": 2.1e-5
  },
  "at_rho_min": {
    "rho": 1.0,
    "phi_min": 1.0,
    "lambda": 1.32
  },

  "regime": "thin_shell",
  "regime_explanation": "λ̂(ρ_max) = 2.1×10⁻⁵ << 1: field tracks local minimum at high density. λ̂(ρ_min) = 1.32 ~ 1: field transitions at low density. Expect thin-shell screening.",

  "selcie_needed": true,
  "recommended_action": "Use SELCIE solver with α=3.5 to resolve the transition region"
}
```

#### Regime Classification

The regime is determined by the Compton wavelength at the density extremes:

| λ̂(ρ_max) | λ̂(ρ_min) | Regime | SELCIE needed? | Behavior |
|-----------|-----------|--------|----------------|----------|
| << 0.1 | << 0.1 | `adiabatic` | No | φ = ρ^{-1/(n+1)} everywhere |
| << 0.1 | ~ 1 or > 1 | `thin_shell` | Yes | Screened core, transition shell |
| ~ 1 | ~ 1 | `transition` | Yes | Intermediate, complex behavior |
| >> 1 | >> 1 | `constant_field` | No | φ ≈ constant everywhere |

#### Examples

**Lab experiment (α ~ 1, interesting regime):**
```
Input: beta=1e8, rho_0=2.7 g/cm³, L=0.01 m, rho_max=2.7, rho_min=1e-10
Output: alpha=3.5, λ̂(ρ_max)=2×10⁻⁵, λ̂(ρ_min)=0.7, regime="thin_shell"
```

**Solar chameleon (α ~ 10⁻²⁴, adiabatic):**
```
Input: beta=1, rho_0=150 g/cm³, L=R_sun, rho_max=150, rho_min=1e-6
Output: alpha=1e-24, λ̂(ρ_max)=10⁻¹⁴, λ̂(ρ_min)=10⁻⁵, regime="adiabatic"
```

---

### 2. `create_mesh`

Generate a finite element mesh for chameleon field simulations. Uses geometry templates that
automatically handle subdomain creation, symmetry, and mesh refinement.

#### Parameters

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `geometry` | string | Yes | Geometry template (see below) |
| `params` | object | Yes | Geometry-specific parameters |
| `mesh_quality` | string | No | `"very_coarse"`, `"coarse"`, `"medium"`, `"fine"`, `"very_fine"`. Default: `"medium"` |
| `symmetry` | string | No | `"axial"`, `"none"`. Default: inferred from geometry |
| `physics_params` | object | No | Physics-aware refinement parameters (see below) |
| `custom_id` | string | No | Custom mesh ID. Default: auto-generated |

#### Geometry Templates

Geometries are organized into three categories:

**1. Object-in-vacuum templates** (automatic subdomain creation):

| Geometry | Default Symmetry | Description |
|----------|------------------|-------------|
| `sphere_in_vacuum` | `axial` | Spherical source in vacuum chamber |
| `ellipse_in_vacuum` | `axial` | Oblate/prolate spheroid in vacuum |
| `ellipsoid_in_vacuum` | `none` (3D) | Triaxial ellipsoid in vacuum |
| `cylinder_in_vacuum` | `axial` | Cylindrical source in vacuum |
| `shell_in_vacuum` | `axial` | Hollow sphere (shell) in vacuum |
| `two_spheres` | `axial` | Two spheres (source + test mass) |
| `sphere_near_wall` | `axial` | Sphere near planar boundary |

**2. Plain domain templates** (for custom/imported density fields):

| Geometry | Default Symmetry | Description |
|----------|------------------|-------------|
| `box_2d` | `none` | 2D rectangular domain |
| `box_3d` | `none` | 3D rectangular domain |
| `disk` | `axial` | 2D circular domain |
| `sphere_domain` | `axial` | Spherical domain (no interior object) |

**3. Custom templates** (from file):

| Geometry | Default Symmetry | Description |
|----------|------------------|-------------|
| `custom_2d` | `axial` | 2D shape from points file |
| `custom_3d` | `none` | 3D shape from contours file |

#### Geometry-Specific Parameters

##### `sphere_in_vacuum`
```json
{
  "geometry": "sphere_in_vacuum",
  "params": {
    "object_radius": 0.1,
    "vacuum_radius": 1.0,
    "wall_thickness": 0.01
  }
}
```
- `object_radius` (required): Radius of the spherical source
- `vacuum_radius` (required): Outer radius of vacuum region
- `wall_thickness` (optional): Thickness of chamber wall

##### `ellipse_in_vacuum`
```json
{
  "geometry": "ellipse_in_vacuum",
  "params": {
    "rx": 0.1,
    "ry": 0.05,
    "vacuum_radius": 1.0
  }
}
```
- `rx` (required): Semi-axis in r direction
- `ry` (required): Semi-axis in z direction
- `vacuum_radius` (required): Outer radius of vacuum region

##### `ellipsoid_in_vacuum`
```json
{
  "geometry": "ellipsoid_in_vacuum",
  "params": {
    "rx": 0.1,
    "ry": 0.08,
    "rz": 0.05,
    "vacuum_radius": 1.0
  }
}
```
- `rx`, `ry`, `rz` (required): Semi-axes in x, y, z directions
- `vacuum_radius` (required): Outer radius of vacuum region

##### `cylinder_in_vacuum`
```json
{
  "geometry": "cylinder_in_vacuum",
  "params": {
    "radius": 0.1,
    "height": 0.5,
    "vacuum_radius": 1.0
  }
}
```
- `radius` (required): Cylinder radius
- `height` (required): Cylinder height
- `vacuum_radius` (required): Outer radius of vacuum region

##### `shell_in_vacuum`
```json
{
  "geometry": "shell_in_vacuum",
  "params": {
    "inner_radius": 0.05,
    "outer_radius": 0.1,
    "vacuum_radius": 1.0
  }
}
```
- `inner_radius` (required): Inner radius of shell
- `outer_radius` (required): Outer radius of shell
- `vacuum_radius` (required): Outer radius of vacuum region

##### `two_spheres`
```json
{
  "geometry": "two_spheres",
  "params": {
    "radius_1": 0.1,
    "radius_2": 0.05,
    "separation": 0.5,
    "vacuum_radius": 2.0
  }
}
```
- `radius_1` (required): Radius of first sphere (source)
- `radius_2` (required): Radius of second sphere (test mass)
- `separation` (required): Center-to-center distance
- `vacuum_radius` (required): Outer radius of vacuum region

##### `sphere_near_wall`
```json
{
  "geometry": "sphere_near_wall",
  "params": {
    "object_radius": 0.1,
    "wall_distance": 0.15,
    "wall_thickness": 0.1,
    "vacuum_radius": 1.0
  }
}
```
- `object_radius` (required): Radius of sphere
- `wall_distance` (required): Distance from sphere center to top of wall (z=0)
- `wall_thickness` (required): Thickness of the physical wall subdomain
- `vacuum_radius` (required): Outer radius of vacuum region

Creates three regions: `sphere`, `wall`, and `vacuum`. The wall extends from z=-wall_thickness to z=0.

##### `box_2d`
```json
{
  "geometry": "box_2d",
  "params": {
    "width": 10.0,
    "height": 10.0
  }
}
```
- `width` (required): Domain width (x direction)
- `height` (required): Domain height (y direction)

##### `box_3d`
```json
{
  "geometry": "box_3d",
  "params": {
    "width": 10.0,
    "height": 10.0,
    "depth": 10.0
  }
}
```
- `width`, `height`, `depth` (required): Domain dimensions

##### `disk`
```json
{
  "geometry": "disk",
  "params": {
    "radius": 1.0
  }
}
```
- `radius` (required): Disk radius

##### `sphere_domain`
```json
{
  "geometry": "sphere_domain",
  "params": {
    "radius": 1.0
  }
}
```
- `radius` (required): Sphere radius

##### `custom_2d`

Custom 2D shapes can be defined either with inline points or from a file:

**Option 1: Inline points**
```json
{
  "geometry": "custom_2d",
  "params": {
    "points": [
      [0.2, 0.0], [0.1, 0.173], [-0.1, 0.173],
      [-0.2, 0.0], [-0.1, -0.173], [0.1, -0.173]
    ],
    "vacuum_radius": 1.0
  }
}
```
- `points` (required if no `shape_file`): Array of [r, z] coordinate pairs defining the shape
- `vacuum_radius` (required): Outer radius of vacuum region

**Option 2: From file**
```json
{
  "geometry": "custom_2d",
  "params": {
    "shape_file": "/path/to/shape.txt",
    "vacuum_radius": 2.0
  }
}
```
- `shape_file` (required if no `points`): Path to file with (r, z) points defining the shape
- `vacuum_radius` (required): Outer radius of vacuum region

File format (whitespace-separated):
```
# r z coordinates defining closed shape
0.25 0.0
0.08 0.06
-0.1 0.0
```

##### `custom_3d`
```json
{
  "geometry": "custom_3d",
  "params": {
    "contours_file": "/path/to/contours.json",
    "vacuum_radius": 2.0
  }
}
```
- `contours_file` (required): Path to JSON file with 3D contour definitions
- `vacuum_radius` (required): Outer radius of vacuum region

#### Mesh Quality Settings

| Quality | Approx. cells (2D) | Use case |
|---------|-------------------|----------|
| `very_coarse` | ~200 | Rapid iteration, quick checks |
| `coarse` | ~500 | Quick tests, prototyping |
| `medium` | ~2000 | Standard runs |
| `fine` | ~8000 | Publication quality |
| `very_fine` | ~32000 | High precision, convergence studies |

Mesh refinement is automatic: finer cells near object boundaries, coarser in vacuum.

#### Physics-Aware Refinement

The `physics_params` parameter enables automatic mesh refinement based on expected physics behavior:

```json
{
  "physics_params": {
    "lambda_subdomain": 0.01
  }
}
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `lambda_subdomain` | number | Dimensionless Compton wavelength inside the object subdomain |

When `lambda_subdomain` is provided, the mesh generator estimates the thin-shell thickness (δ ≈ λ_subdomain) and refines the mesh at object boundaries to resolve this scale. This is essential for accurate solutions when:
- The thin shell is much smaller than the object radius
- High density contrast leads to rapid field transitions at boundaries

**Refinement limits** prevent excessive cell counts:
- Minimum cell size is bounded by 0.2% of object size
- Maximum refinement ratio (largest/smallest cells) capped at 50
- Target of ~5 cells across the thin-shell region

Example with thin-shell refinement:
```json
{
  "geometry": "sphere_in_vacuum",
  "params": {"object_radius": 0.15, "vacuum_radius": 1.0},
  "mesh_quality": "coarse",
  "physics_params": {"lambda_subdomain": 0.01}
}
```

This produces more cells than the same geometry without `physics_params`, with refinement concentrated at the sphere boundary.

#### Symmetry Override

The `symmetry` parameter can override the default:
- `"axial"`: 2D mesh in (r, z) plane, axisymmetric around z-axis
- `"none"`: True 2D (no symmetry) or full 3D

Example: A 2D circle problem (not axisymmetric):
```json
{
  "geometry": "disk",
  "params": {"radius": 1.0},
  "symmetry": "none"
}
```

#### Returns

```json
{
  "mesh_id": "mesh_001",
  "geometry": "sphere_in_vacuum",
  "symmetry": "axial",
  "dimension": 2,
  "n_cells": 2048,
  "n_vertices": 1089,
  "regions": {"object": 1, "vacuum": 0},
  "domain_bounds": {
    "r_min": 0.0,
    "r_max": 1.0,
    "z_min": -1.0,
    "z_max": 1.0
  },
  "mesh_path": "/path/to/mesh/files",
  "quality": "medium"
}
```

Note: If `wall_thickness` is provided, the `regions` will also include `"wall": 2`.

---

### 3. `plot_mesh`

Generate a visualization of a mesh showing the geometry and subdomain structure.

#### Parameters

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `mesh_id` | string | Yes | ID of the mesh to plot |
| `show_edges` | boolean | No | Show cell edges. Default: `true` |
| `title` | string | No | Custom plot title. Default: auto-generated |
| `output_path` | string | No | Save to file path. If not provided, returns base64 image |
| `figsize` | array | No | Figure size `[width, height]` in inches. Default: `[8, 8]` |
| `dpi` | integer | No | Resolution in dots per inch. Default: `150` |

#### Returns

If `output_path` is provided:
```json
{
  "mesh_id": "mesh_001",
  "plot_saved": "/path/to/plot.png",
  "geometry": "sphere_in_vacuum",
  "n_cells": 2048
}
```

If `output_path` is not provided, returns an image content block (base64-encoded PNG) along with mesh metadata.

#### Subdomain Colors

Regions are colored to distinguish different subdomains:
- Light blue: vacuum/domain (marker 0)
- Medium blue: object/inner region (marker 1)
- Dark blue: shell/second region (marker 2)
- Green: third region (marker 3)

---

### 4. `solve`

Solve the chameleon field equation on a mesh with specified density profile.

#### Parameters

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `mesh_id` | string | Yes | Mesh ID |
| `alpha` | number | Yes | Dimensionless α parameter |
| `density` | object | Yes | Density profile specification (see below) |
| `n` | integer | No | Potential power. Default: 1 |
| `method` | string | No | `"picard"`, `"auto"`. Default: `"auto"` |
| `tol` | number | No | Convergence tolerance. Default: 1e-8 |
| `max_iter` | integer | No | Maximum iterations. Default: 100 |
| `relaxation` | number | No | Relaxation factor (0-1]. Default: 1.0 |
| `initial_guess` | string | No | `"constant"`, `"adiabatic"`, `"previous"`. Default: `"constant"` (uses lowest density across subdomains) |
| `custom_id` | string | No | Custom solution ID |

#### Density Specification

The `density` parameter is a dictionary mapping region names to density values. Each region can specify density as:

- **Number**: Constant density value
- **Object with `expression`**: Custom formula
- **Object with `file`**: Tabulated data from file

##### Constant density
```json
{
  "density": {
    "object": 1e6,
    "vacuum": 1.0
  }
}
```

##### Custom expression
```json
{
  "density": {
    "object": {"expression": "1e6 / (1 + (r/0.05)**2)"},
    "vacuum": 1.0
  }
}
```

Available variables: `r` (radial), `z` (axial), `x`, `y` (Cartesian).

Common profile examples:
```python
# NFW profile
{"expression": "rho_s / ((r/r_s) * (1 + r/r_s)**2)"}

# Isothermal sphere
{"expression": "rho_0 / (1 + (r/r_c)**2)"}

# Power law
{"expression": "rho_0 * (r/r_0)**gamma"}
```

##### Tabulated data
```json
{
  "density": {
    "object": {"file": "/path/to/profile.dat", "skip_header": 1},
    "vacuum": 1e-10
  }
}
```

Column format is auto-detected from mesh symmetry and number of columns:

| Symmetry | Columns | Format |
|----------|---------|--------|
| `axial` | 2 | (r, ρ) |
| `axial` | 3 | (r, z, ρ) |
| `none` (2D) | 3 | (x, y, ρ) |
| `none` (3D) | 4 | (x, y, z, ρ) |

##### Mixed example
```json
{
  "density": {
    "halo": {"expression": "rho_s / ((r/r_s) * (1 + r/r_s)**2)"},
    "core": {"file": "/data/core_profile.dat"},
    "vacuum": 1e-10
  }
}
```

#### Solver Selection (auto mode)

| Condition | Relaxation | Reason |
|-----------|------------|--------|
| α < 1 | 1.0 | Stable for weakly nonlinear |
| 1 ≤ α < 1000 | 0.5 - 1.0 | Moderate nonlinearity needs damping |
| α ≥ 1000 | 0.5 | Strong relaxation for stiff problems |

#### Returns

```json
{
  "solution_id": "solution_001",
  "mesh_id": "mesh_001",
  "alpha": 3.5,
  "n": 1,

  "density_stats": {
    "rho_min": 1.0,
    "rho_max": 1e6
  },

  "status": "converged",
  "iterations": 23,
  "final_residual": 4.2e-9,

  "method_used": "picard",
  "relaxation_used": 0.8,
  "initial_guess": "constant",

  "field_stats": {
    "min": 0.001,
    "max": 1.0,
    "mean": 0.45,
    "at_origin": 0.001
  },

  "runtime_seconds": 2.3
}
```

#### Error Response

```json
{
  "solution_id": null,
  "status": "failed",
  "error": "diverged",
  "iterations": 100,
  "final_residual": 1.2e3,
  "suggestion": "Try reducing relaxation factor to 0.5, or use Newton method for high alpha"
}
```

---

### 5. `evaluate`

Evaluate field values and derived quantities at specified locations.

#### Parameters

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `solution_id` | string | Yes | Solution ID |
| `mode` | string | Yes | Evaluation mode (see below) |
| `params` | object | No | Mode-specific parameters |
| `quantities` | array | No | What to compute. Default: all |

#### Evaluation Modes

##### `radial`
Along radial direction:
```json
{
  "mode": "radial",
  "params": {
    "n_points": 200,
    "r_min": 0.01,
    "r_max": null,
    "log_spacing": true,
    "direction": [1, 0]
  }
}
```

##### `line`
Along arbitrary line:
```json
{
  "mode": "line",
  "params": {
    "start": [0, 0],
    "end": [1, 1],
    "n_points": 100
  }
}
```

##### `points`
At specific points:
```json
{
  "mode": "points",
  "params": {
    "coordinates": [[0, 0], [0.5, 0], [1.0, 0]]
  }
}
```

##### `grid`
On regular grid:
```json
{
  "mode": "grid",
  "params": {
    "r_range": [0, 2],
    "z_range": [-1, 1],
    "n_r": 50,
    "n_z": 50
  }
}
```

#### Available Quantities

| Quantity | Description |
|----------|-------------|
| `field` | Chameleon field φ |
| `gradient` | ∇φ (vector) |
| `gradient_magnitude` | |∇φ| |
| `fifth_force` | -∇φ/M (acceleration) |
| `fifth_force_g` | Fifth force in units of g |
| `density` | ρ at evaluation points |
| `adiabatic_field` | ρ^{-1/(n+1)} for comparison |
| `field_deviation` | (φ - φ_adiabatic) / φ_adiabatic |

#### Returns

```json
{
  "solution_id": "solution_001",
  "mode": "radial",
  "n_points": 200,

  "data": {
    "r": [0.01, 0.012, 0.015, ...],
    "field": [0.001, 0.0012, 0.0015, ...],
    "gradient_magnitude": [0.12, 0.11, 0.10, ...],
    "fifth_force_g": [3.2e-4, 3.1e-4, 2.9e-4, ...],
    "density": [1e6, 9.8e5, 9.5e5, ...],
    "adiabatic_field": [0.001, 0.00101, 0.00103, ...],
    "field_deviation": [0.0, 0.02, 0.05, ...]
  },

  "units": {
    "r": "dimensionless (L)",
    "field": "dimensionless (φ₀)",
    "fifth_force_g": "g (9.81 m/s²)"
  }
}
```

---

### 6. `analyze`

Physical interpretation of the solution.

#### Parameters

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `solution_id` | string | Yes | Solution ID |
| `reference` | string | No | Compare to: `"adiabatic"`, `"thin_shell"`, `"none"` |

#### Returns

```json
{
  "solution_id": "solution_001",
  "alpha": 3.5,
  "n": 1,

  "regime": {
    "classification": "thin_shell",
    "confidence": 0.92,
    "description": "Object exhibits thin-shell screening"
  },

  "screening": {
    "is_screened": true,
    "screening_radius": 0.82,
    "screening_radius_fraction": 0.82,
    "thin_shell_thickness": 0.18,
    "thin_shell_fraction": 0.18,
    "volume_fraction_screened": 0.55
  },

  "field_values": {
    "at_center": 0.001,
    "at_screening_radius": 0.45,
    "at_surface": 0.92,
    "at_boundary": 0.98,
    "background_analytic": 1.0
  },

  "fifth_force": {
    "max_value_g": 3.2e-4,
    "max_location_r": 1.0,
    "mean_in_shell_g": 1.8e-4,
    "surface_value_g": 3.2e-4
  },

  "comparison_to_analytic": {
    "reference": "adiabatic",
    "rms_deviation": 0.34,
    "max_deviation": 0.92,
    "max_deviation_location": 1.0,
    "agreement_region": "r < 0.5"
  },

  "interpretation": "The object has a well-developed thin shell. The inner 82% by radius (55% by volume) is fully screened, with the field matching the adiabatic solution φ = ρ^{-1/2}. In the outer shell (r > 0.82), the field deviates significantly and approaches the background value. The maximum fifth force of 3.2×10⁻⁴ g occurs at the surface."
}
```

---

### 7. `compare`

Compare multiple solutions or parameter sweeps.

#### Parameters

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `solution_ids` | array | Yes | List of solution IDs to compare |
| `metric` | string | No | Comparison metric. Default: `"all"` |

#### Returns

```json
{
  "solutions": [
    {
      "solution_id": "solution_001",
      "alpha": 0.1,
      "regime": "adiabatic",
      "screening_radius": null,
      "max_fifth_force_g": 1.2e-6
    },
    {
      "solution_id": "solution_002",
      "alpha": 1.0,
      "regime": "transition",
      "screening_radius": 0.65,
      "max_fifth_force_g": 8.5e-5
    },
    {
      "solution_id": "solution_003",
      "alpha": 10.0,
      "regime": "thin_shell",
      "screening_radius": 0.92,
      "max_fifth_force_g": 4.1e-4
    }
  ],

  "trends": {
    "screening_radius_vs_alpha": {
      "fit": "r_s = 1 - C * alpha^{-1/(n+1)}",
      "C": 0.35
    },
    "max_force_vs_alpha": {
      "scaling": "F_max ∝ alpha^{0.45}"
    }
  },

  "regime_boundaries": {
    "adiabatic_to_transition": 0.3,
    "transition_to_thin_shell": 3.0
  },

  "interpretation": "As α increases from 0.1 to 10, the system transitions from adiabatic (field tracks density) through partial screening to thin-shell regime. Screening radius increases from 0 to 92% of object radius. Fifth force increases by factor ~300."
}
```

---

### 8. `plot`

Generate visualizations.

#### Parameters

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `solution_id` | string or array | Yes | Solution(s) to plot |
| `plot_type` | string | Yes | Type of plot (see below) |
| `options` | object | No | Plot customization |
| `output_path` | string | No | Save path. If null, returns base64 |
| `format` | string | No | `"png"`, `"pdf"`, `"svg"`. Default: `"png"` |

#### Plot Types

##### `field_1d`
Radial field profile:
```json
{
  "plot_type": "field_1d",
  "options": {
    "log_r": true,
    "show_density": true,
    "show_adiabatic": true,
    "show_screening_radius": true
  }
}
```

##### `field_2d`
2D field visualization:
```json
{
  "plot_type": "field_2d",
  "options": {
    "colormap": "viridis",
    "show_mesh": false,
    "log_scale": false
  }
}
```

##### `force_1d`
Radial force profile:
```json
{
  "plot_type": "force_1d",
  "options": {
    "units": "g",
    "log_scale": true
  }
}
```

##### `force_2d`
2D force field (vectors or magnitude):
```json
{
  "plot_type": "force_2d",
  "options": {
    "style": "magnitude",
    "colormap": "hot"
  }
}
```

##### `comparison`
Compare multiple solutions:
```json
{
  "solution_id": ["sol_001", "sol_002", "sol_003"],
  "plot_type": "comparison",
  "options": {
    "quantity": "field",
    "legend_by": "alpha"
  }
}
```

##### `convergence`
Solver convergence history:
```json
{
  "plot_type": "convergence",
  "options": {
    "log_scale": true
  }
}
```

##### `mesh`
Visualize mesh:
```json
{
  "plot_type": "mesh",
  "options": {
    "show_regions": true,
    "show_quality": false
  }
}
```

#### Returns

```json
{
  "plot_type": "field_1d",
  "output_path": "/path/to/plot.png",
  "image_base64": null,
  "metadata": {
    "solution_ids": ["solution_001"],
    "figure_size": [8, 6],
    "dpi": 150
  }
}
```

---

### 9. `sweep`

Run parameter sweep (batch solving).

#### Parameters

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `mesh_id` | string | Yes | Mesh ID |
| `profile_id` | string | Yes | Profile ID |
| `parameter` | string | Yes | Parameter to sweep: `"alpha"`, `"n"` |
| `values` | array | Yes | Parameter values |
| `solver_options` | object | No | Solver configuration |
| `analyze` | boolean | No | Run analysis on each. Default: true |

#### Returns

```json
{
  "sweep_id": "sweep_001",
  "parameter": "alpha",
  "n_runs": 10,

  "results": [
    {"alpha": 0.01, "solution_id": "sol_001", "converged": true, "regime": "adiabatic"},
    {"alpha": 0.1, "solution_id": "sol_002", "converged": true, "regime": "adiabatic"},
    {"alpha": 1.0, "solution_id": "sol_003", "converged": true, "regime": "transition"},
    ...
  ],

  "summary": {
    "all_converged": true,
    "regime_transitions": [
      {"from": "adiabatic", "to": "transition", "at_alpha": 0.3},
      {"from": "transition", "to": "thin_shell", "at_alpha": 5.0}
    ]
  },

  "total_runtime_seconds": 45.2
}
```

---

### 10. `get_state`

Query current session state.

#### Parameters

None.

#### Returns

```json
{
  "session_id": "sess_abc123",
  "created_at": "2024-01-15T10:30:00Z",

  "meshes": [
    {"mesh_id": "mesh_001", "geometry": "sphere_in_vacuum", "n_cells": 2048, "dimension": 2, "regions": {"object": 1, "vacuum": 0}}
  ],

  "solutions": [
    {"solution_id": "solution_001", "mesh_id": "mesh_001", "alpha": 3.5, "converged": true}
  ]
}
```

---

### 11. `clear`

Clear session objects.

#### Parameters

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `what` | string | No | `"all"`, `"solutions"`, `"meshes"`. Default: `"all"` |
| `ids` | array | No | Specific IDs to clear. If provided, clears those specific IDs regardless of `what` |

#### Returns

```json
{
  "cleared": ["mesh_001", "profile_001", "solution_001"],
  "remaining": []
}
```

---

### 12. `get_documentation`

Get documentation for profiles, geometries, or physics.

#### Parameters

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `topic` | string | Yes | Topic to document |

#### Topics

- `"profiles"` - List available density profiles
- `"geometries"` - List available mesh geometries
- `"regimes"` - Explain screening regimes
- `"parameters"` - Explain α, n, β, Λ
- `"units"` - Unit conversion reference
- `"solver"` - Solver options and troubleshooting

#### Returns

```json
{
  "topic": "regimes",
  "content": "# Chameleon Screening Regimes\n\n## Adiabatic (α×ρ << 1)\n..."
}
```

---

## Error Handling

All tools return errors in a consistent format:

```json
{
  "error": {
    "code": "MESH_NOT_FOUND",
    "message": "Mesh 'mesh_999' does not exist",
    "suggestion": "Use get_state() to list available meshes",
    "available_meshes": ["mesh_001", "mesh_002"]
  }
}
```

### Error Codes

| Code | Description |
|------|-------------|
| `MESH_NOT_FOUND` | Referenced mesh doesn't exist |
| `PROFILE_NOT_FOUND` | Referenced profile doesn't exist |
| `SOLUTION_NOT_FOUND` | Referenced solution doesn't exist |
| `SOLVER_DIVERGED` | Solver failed to converge |
| `INVALID_GEOMETRY` | Geometry specification invalid |
| `INVALID_PROFILE` | Profile specification invalid |
| `INVALID_PARAMETER` | Parameter out of valid range |
| `FILE_NOT_FOUND` | Tabulated data file not found |
| `MESH_PROFILE_MISMATCH` | Profile mesh doesn't match solve mesh |

---

## Physical Constants

The server uses these constants (from astropy):

| Constant | Value | Unit |
|----------|-------|------|
| M_pl | 2.435 × 10¹⁸ | GeV |
| c | 299792458 | m/s |
| ℏ | 1.055 × 10⁻³⁴ | J·s |
| g | 9.80665 | m/s² |

---

## Example Workflows

### 1. Lab Experiment Analysis

```python
# Calculate physical parameters
params = calculate_physical_parameters(
    beta=1e8, Lambda_eV=2.4e-3, n=1,
    rho_0=2.7, rho_0_units="g/cm^3",
    L=0.01, L_units="m",
    rho_max=2.7, rho_min=1e-10
)

# Create mesh
mesh = create_mesh(
    geometry="sphere_in_vacuum",
    params={"object_radius": 0.1, "vacuum_radius": 1.0},
    mesh_quality="fine"
)

# Solve (density specified inline)
solution = solve(
    mesh_id=mesh["mesh_id"],
    alpha=params["alpha"],
    density={
        "object": 2.7e6,  # Aluminum density (dimensionless)
        "vacuum": 1.0
    }
)

# Analyze
analysis = analyze(solution_id=solution["solution_id"])

# Plot
plot(solution_id=solution["solution_id"], plot_type="field_1d")
```

### 2. Parameter Space Exploration

```python
# Create mesh once
mesh = create_mesh(
    geometry="sphere_in_vacuum",
    params={"object_radius": 0.1, "vacuum_radius": 1.0}
)

# Solve for multiple alpha values
solutions = []
for alpha in [0.01, 0.1, 1.0, 10.0, 100.0]:
    sol = solve(
        mesh_id=mesh["mesh_id"],
        alpha=alpha,
        density={"object": 1e6, "vacuum": 1.0}
    )
    solutions.append(sol["solution_id"])

# Compare results
comparison = compare(solution_ids=solutions)

# Plot comparison
plot(solution_id=solutions, plot_type="comparison")
```

### 3. Astrophysical Profile

```python
# Calculate solar parameters
params = calculate_physical_parameters(
    beta=1, Lambda_eV=2.4e-3, n=1,
    rho_0=150, rho_0_units="g/cm^3",
    L=6.96e8, L_units="m",
    rho_max=150, rho_min=1e-6
)
# Returns: alpha ~ 1e-24, regime: "adiabatic", selcie_needed: false

# Since adiabatic, can use analytic solution directly
# No need to run SELCIE!
```

---

## Versioning

This API follows semantic versioning. Breaking changes will increment the major version.

Current version: **1.0.0**
