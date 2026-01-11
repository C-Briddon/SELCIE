# SELCIE MCP Server - Tool Reference

Auto-generated documentation for all available tools.

_Generated: 2026-01-11 23:12_

---

## Table of Contents

- [calculate_physical_parameters](#calculate_physical_parameters)
- [clear](#clear)
- [create_mesh](#create_mesh)
- [evaluate](#evaluate)
- [get_state](#get_state)
- [plot](#plot)
- [plot_mesh](#plot_mesh)
- [solve](#solve)

---

## calculate_physical_parameters

Convert physical chameleon parameters to SELCIE's dimensionless α and assess the screening regime using the Compton wavelength criterion. Use this first to determine if SELCIE is needed or if analytic solutions suffice. Returns the dimensionless Compton wavelength λ̂(ρ̂) = √(α/(n+1)) × ρ̂^{-(n+2)/(2(n+1))} at the density extremes. Since λ̂ is in units of L, compare to 1: λ̂ << 1 means adiabatic (field tracks ρ̂^{-1/(n+1)}), λ̂ >> 1 means field is set by boundaries, λ̂ ~ 1 is the transition region where SELCIE is needed. Also returns a conversion factor to translate dimensionless grad(φ) from the solver to physical fifth force in units of g.

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `beta` | number | Yes | Matter coupling strength β. M = M_pl/β where M_pl is Planck mass. Typical values: β=1 (gravitational strength), β=10^6-10^8 (lab experiments). |
| `Lambda_eV` | number | No | Energy scale Λ in eV. Default 2.4e-3 (dark energy scale). Default: `0.0024` |
| `n` | integer | No | Potential power in V(φ) = Λ⁴(1 + Λⁿ/φⁿ). Default 1. Default: `1` |
| `rho_0` | number | Yes | Characteristic density scale of the system (e.g., central density, object density). Enters the α calculation as α ∝ 1/ρ₀. |
| `rho_0_units` | `g/cm^3` | `kg/m^3` | `eV^4` | `M_sun/kpc^3` | `GeV^4` | Yes | Units of rho_0. |
| `L` | number | Yes | Characteristic length scale of the system (e.g., object radius, domain size). Enters the α calculation as α ∝ 1/L². |
| `L_units` | `m` | `cm` | `km` | `R_sun` | `kpc` | `Mpc` | `AU` | Yes | Units of L. |
| `rho_max` | number | No | Maximum density in system, in same units as rho_0. Used for regime estimation via Compton wavelength. |
| `rho_min` | number | No | Minimum density in system, in same units as rho_0. Used for regime estimation via Compton wavelength. |

---

## clear

Clear session objects to free memory.

Can clear all objects, just meshes, just solutions, or specific IDs.

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `what` | `all` | `solutions` | `meshes` | No | What to clear. Default: all Default: `"all"` |
| `ids` | array[string] | No | Specific IDs to clear. If provided, clears those specific IDs regardless of 'what' |

---

## create_mesh

Generate a finite element mesh for chameleon field simulations. Uses geometry templates that automatically handle subdomain creation, symmetry, and mesh refinement. Templates include object-in-vacuum (sphere_in_vacuum, ellipse_in_vacuum, etc.), plain domains (box_2d, disk, etc.), and custom shapes from file.

IMPORTANT: For thin-shell problems (high α, high density contrast), provide physics_params with alpha and density_contrast to enable automatic mesh refinement near object boundaries. This ensures the thin shell region is properly resolved.

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `geometry` | `sphere_in_vacuum` | `ellipse_in_vacuum` | `ellipsoid_in_vacuum` | `cylinder_in_vacuum` | `shell_in_vacuum` | `two_spheres` | `sphere_near_wall` | `box_2d` | `box_3d` | `disk` | `sphere_domain` | `custom_2d` | `custom_3d` | Yes | Geometry template. |
| `params` | object | Yes | Geometry-specific parameters. |
| `mesh_quality` | `very_coarse` | `coarse` | `medium` | `fine` | `very_fine` | No | Mesh resolution. Default: `"medium"` |
| `symmetry` | `axial` | `none` | No | Override default symmetry. |
| `custom_id` | string | No | Custom mesh ID. |
| `allow_large_mesh` | boolean | No | Allow meshes exceeding 200,000 cells. Default: false. Default: `False` |
| `physics_params` | object | No | Physics parameters for automatic thin-shell mesh refinement. Option 1: Provide 'lambda' dict mapping region names to Compton wavelengths. Option 2: Provide 'alpha', 'density' dict, and 'n' - lambdas will be computed per region. The mesh will be refined near boundaries of dense regions to resolve thin shells. |

#### `params` options

- **`object_radius`** (number): Radius of spherical source
- **`vacuum_radius`** (number): Outer radius of vacuum region
- **`wall_thickness`** (number): Wall thickness (sphere_in_vacuum, sphere_near_wall)
- **`rx`** (number): Semi-axis in r/x direction
- **`ry`** (number): Semi-axis in z/y direction
- **`rz`** (number): Semi-axis in z direction (3D)
- **`radius`** (number): Radius for cylinder, disk, sphere_domain
- **`height`** (number): Height for cylinder, box
- **`width`** (number): Width for box
- **`depth`** (number): Depth for box_3d
- **`inner_radius`** (number): Inner radius for shell
- **`outer_radius`** (number): Outer radius for shell
- **`radius_1`** (number): First sphere radius (two_spheres)
- **`radius_2`** (number): Second sphere radius (two_spheres)
- **`separation`** (number): Center-to-center separation (two_spheres)
- **`wall_distance`** (number): Distance from sphere center to wall (sphere_near_wall)
- **`points`** (array[any]): Array of [r,z] points for custom_2d
- **`shape_file`** (string): Path to file with shape points (custom_2d)
- **`contours_file`** (string): Path to 3D contours file (custom_3d)

#### `physics_params` options

- **`lambda`** (object): Direct specification of Compton wavelength per region. Example: {"object": 0.001, "wall": 0.002}. If provided, alpha/density are ignored.
- **`alpha`** (number): Dimensionless coupling constant α. Used with 'density' to compute λ = √(α/n(n+1)) × ρ^(-(n+2)/(2(n+1))) for each region.
- **`density`** (object): Dimensionless density ρ̂ = ρ/ρ₀ per region. Value is: number, {expression: str}, or {file: str, format?: 'tabulated'|'grid', columns?: int[], bounds?: number[], skip_header?: int, npz_key?: str}. For tabulated: columns selects columns (1-based). For grid: bounds maps grid to spatial coordinates. For non-numeric values, max density is used to compute λ.
- **`n`** (integer): Potential power index (default: 1)

---

## evaluate

Evaluate field values and derived quantities at specified locations.

Modes:
- radial: Sample along radial direction from origin
- line: Sample along arbitrary line between two points
- points: Evaluate at specific coordinates
- grid: Sample on regular 2D grid
- max_in_region: Find max/min values within a region, with optional minimum distance from other region(s). Use min_distance_from='all' to exclude points near any other domain boundary.

Quantities:
- field: Chameleon field φ
- gradient_magnitude: |∇φ| (dimensionless). Multiply by force_conversion_to_g from calculate_physical_parameters to get fifth force acceleration in units of g.
- fifth_force_g: Alias for gradient_magnitude (returns gradient_magnitude)
- density: ρ̂ at evaluation points (if available)
- adiabatic_field: ρ̂^{-1/(n+1)} for comparison
- field_deviation: (φ - φ_adiabatic) / φ_adiabatic

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `solution_id` | string | Yes | ID of the solution to evaluate |
| `mode` | `radial` | `line` | `points` | `grid` | `max_in_region` | Yes | Evaluation mode |
| `params` | object | No | Mode-specific parameters |
| `quantities` | array[string] | No | Quantities to compute. Default: ['field', 'gradient_magnitude'] |

#### `params` options

- **`n_points`** (integer): Number of sample points (radial, line)
- **`r_min`** (number): Minimum radius (radial)
- **`r_max`** (number): Maximum radius (radial)
- **`log_spacing`** (boolean): Use log spacing (radial)
- **`direction`** (array[any]): Direction vector [r, z] (radial)
- **`start`** (array[any]): Start point (line)
- **`end`** (array[any]): End point (line)
- **`coordinates`** (array[any]): List of [r, z] points (points)
- **`r_range`** (array[any]): [r_min, r_max] (grid)
- **`z_range`** (array[any]): [z_min, z_max] (grid)
- **`n_r`** (integer): Number of r points (grid)
- **`n_z`** (integer): Number of z points (grid)
- **`region`** (string): Region to sample (max_in_region). Default: vacuum
- **`min_distance_from`** (string | array[string]): Region(s) to keep distance from (max_in_region). Can be: a region name, 'all' for all other regions, or a list of region names
- **`min_distance`** (number): Minimum distance from boundary of exclusion region(s) (max_in_region)
- **`n_samples`** (integer): Number of random samples (max_in_region). Default: 1000

---

## get_state

Query current session state: list all meshes and solutions.

Returns information about all meshes and solutions in the current session,
including their IDs, parameters, and status.

### Parameters

_No parameters_

---

## plot

Generate visualizations of chameleon field solutions.

Plot types:
- field_1d: Radial field profile φ(r)
- field_2d: 2D colormap of field
- force_1d: Radial gradient magnitude |∇φ|(r)
- force_2d: 2D colormap of gradient magnitude
- comparison: Compare multiple solutions on same plot
- density or density_1d: Radial density profile ρ̂(r) with optional adiabatic field overlay
- density_2d: 2D colormap of density

Returns PNG image (base64 or saved to file).

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `solution_id` | string | array[string] | Yes | Solution ID(s) to plot |
| `plot_type` | `field_1d` | `field_2d` | `force_1d` | `force_2d` | `comparison` | `density` | `density_1d` | `density_2d` | Yes | Type of plot to generate |
| `options` | object | No | Plot customization options |
| `output_path` | string | No | Save to file path. If not provided, returns base64 image |
| `format` | `png` | `pdf` | `svg` | No | Output format. Default: png Default: `"png"` |

#### `options` options

- **`log_r`** (boolean): Use log scale for r-axis (1D plots)
- **`log_scale`** (boolean): Use log scale for values
- **`colormap`** (string): Matplotlib colormap name
- **`show_mesh`** (boolean): Overlay mesh on 2D plots
- **`n_points`** (integer): Number of sample points for 1D
- **`r_min`** (number): Minimum radius for 1D
- **`r_max`** (number): Maximum radius for 1D
- **`quantity`** (string): Quantity to plot (comparison mode)
- **`legend_by`** (string): Label legend by this field
- **`figsize`** (array[number]): 
- **`title`** (string): Custom title

---

## plot_mesh

Generate a visualization of a mesh showing the geometry and subdomain structure. Returns a PNG image showing the mesh with different colors for each subdomain region.

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `mesh_id` | string | Yes | ID of the mesh to plot. |
| `show_edges` | boolean | No | Show cell edges. Default: true. Default: `True` |
| `title` | string | No | Custom plot title. Default: auto-generated from mesh info. |
| `output_path` | string | No | Save plot to this path. If not provided, returns base64-encoded image. |
| `figsize` | array[number] | No | Figure size as [width, height] in inches. Default: `[8, 8]` |
| `dpi` | integer | No | Resolution in dots per inch. Default: `150` |

---

## solve

Solve the chameleon field equation on a mesh with specified density profile.

Uses SELCIE's Picard or Newton solver to compute the chameleon scalar field
throughout the domain. The dimensionless field equation is:
    α ∇²φ + φ^{-(n+1)} = ρ̂

where ρ̂ = ρ/ρ₀ is the dimensionless density (ρ₀ is the reference density used to compute α).

Parameters:
- mesh_id: Reference to a previously created mesh
- alpha: Dimensionless coupling constant (from calculate_physical_parameters)
- density: Dimensionless density ρ̂ = ρ/ρ₀ per region (e.g., if ρ₀ = vacuum density, then vacuum → 1.0)
- n: Potential power (default: 1)
- method: Solver method - "picard", "newton", or "auto"
- tol: Convergence tolerance (default: 1e-14)
- max_iter: Maximum iterations (default: 100)
- relaxation: Relaxation factor for Picard (default: 1.0)
- initial_guess: "constant" (default, recommended for SELCIE), "adiabatic", or "previous"

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `mesh_id` | string | Yes | ID of the mesh to solve on |
| `alpha` | number | Yes | Dimensionless α parameter (coupling constant) |
| `density` | object | Yes | Dimensionless density ρ̂ = ρ/ρ₀ per region. Value is: number, {expression: str}, or {file: str, format?: 'tabulated'\|'grid', columns?: int[], bounds?: number[], skip_header?: int, npz_key?: str}. For tabulated: columns selects columns (1-based). For grid: bounds maps grid to spatial coordinates. For expressions: spherical geometries use r=spherical radius. |
| `n` | integer | No | Potential power index. Default: 1 Default: `1` |
| `method` | `picard` | `auto` | No | Solver method. Default: auto (uses picard with relaxation based on alpha) Default: `"auto"` |
| `tol` | number | No | Convergence tolerance. Default: 1e-14 Default: `1e-14` |
| `max_iter` | integer | No | Maximum iterations. Default: 100 Default: `100` |
| `relaxation` | number | No | Relaxation factor (0-1]. Default: 1.0 Default: `1.0` |
| `initial_guess` | `constant` | `adiabatic` | `previous` | No | Initial guess strategy. Default: constant (recommended for SELCIE) Default: `"constant"` |
| `custom_id` | string | No | Custom solution ID. Default: auto-generated |
| `deg_V` | integer | No | Function space degree (1=CG1, 2=CG2). Default: 2 Default: `2` |

---
