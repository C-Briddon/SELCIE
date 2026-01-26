# SELCIE MCP Server - Tool Reference

Auto-generated documentation for all available tools.

_Generated: 2026-01-26 22:46_

---

## Table of Contents

- [calculate_physical_parameters](#calculate_physical_parameters)
- [clear](#clear)
- [create_mesh](#create_mesh)
- [evaluate](#evaluate)
- [get_state](#get_state)
- [plot](#plot)
- [plot_mesh](#plot_mesh)
- [plot_step](#plot_step)
- [solve](#solve)

---

## calculate_physical_parameters

Convert physical chameleon parameters to SELCIE's dimensionless α and assess the screening regime using the Compton wavelength criterion.

IMPORTANT: If working with physical units, call this tool FIRST before creating geometry (STEP files) or meshes. The returned coordinate_scaling factor must be applied to all physical dimensions when creating geometry.

Returns the dimensionless Compton wavelength λ̂(ρ̂) = √(α/(n+1)) × ρ̂^{-(n+2)/(2(n+1))} at the density extremes. Since λ̂ is in units of L, compare to 1: λ̂ << 1 means adiabatic (field tracks ρ̂^{-1/(n+1)}), λ̂ >> 1 means field is set by boundaries, λ̂ ~ 1 is the transition region where SELCIE is needed. Also returns conversion factors: grad_to_acceleration_g converts dimensionless ∇φ to acceleration in units of g; mass_scale_kg, force_scale_N, and torque_scale_Nm convert integrate mode outputs to physical mass (kg), force (N), and torque (N·m).

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `beta` | number | Yes | Matter coupling strength β. M = M_pl/β where M_pl is Planck mass. Typical values: β=1 (gravitational strength), β=10^6-10^8 (lab experiments). |
| `Lambda_eV` | number | No | Energy scale Λ in eV. Default 2.4e-3 (dark energy scale). Default: `0.0024` |
| `n` | integer | No | Potential power in V(φ) = Λ⁴(1 + Λⁿ/φⁿ). Default 1. Default: `1` |
| `rho_0` | number | No | Reference density scale for non-dimensionalization. Enters α as α ∝ 1/ρ₀. Optional: defaults to rho_min if provided. For best solver convergence, use the lowest density (e.g., vacuum) or an intermediate value. |
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

All distances (object_radius, domain_radius, etc.) are dimensionless. If using physical parameters from calculate_physical_parameters, distances should be in units of L: x̂ = x_physical / L.

. This includes all mesh distances and step file distances.IMPORTANT: For thin-shell problems (high α, high density contrast), provide physics_params with alpha and density to enable automatic mesh refinement near object boundaries. This ensures the thin shell region is properly resolved.

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `geometry` | `sphere_in_vacuum` | `ellipse_in_vacuum` | `cylinder_in_vacuum` | `shell_in_vacuum` | `two_spheres` | `sphere_near_wall` | `sphere_in_profile` | `box_2d` | `box_3d` | `disk` | `sphere_domain` | `parallel_plates` | `custom_2d_axial` | `custom_2d_translation` | `custom_3d` | `custom_step` | Yes | Geometry template. Choose based on physical setup:

OBJECT-IN-VACUUM (screening/force calculations):
- sphere_in_vacuum: Spherical source in vacuum. Regions: object, vacuum [+wall]. Fixed symmetry: axial (2D). 'r' = spherical radius.
- ellipse_in_vacuum: Oblate/prolate ellipsoid in vacuum. Regions: object, vacuum [+wall]. Fixed symmetry: axial (2D). 'r' = cylindrical radius.
- cylinder_in_vacuum: Cylindrical source in vacuum. Regions: cylinder, vacuum [+wall]. Fixed symmetry: axial (2D). 'r' = cylindrical radius.
- shell_in_vacuum: Hollow spherical shell in vacuum. Regions: shell, vacuum [+wall]. Fixed symmetry: axial (2D). 'r' = spherical radius.
- two_spheres: Two spheres for force calculations. Regions: sphere_1, sphere_2, vacuum [+wall]. Fixed symmetry: axial (2D). 'r' = cylindrical radius.
- sphere_near_wall: Sphere near planar wall. Regions: sphere, wall, vacuum. Fixed symmetry: axial (2D). 'r' = cylindrical radius.
- sphere_in_profile: Sphere in spatially-varying density profile (e.g., NFW, isothermal). Regions: sphere, background [+wall]. Fixed symmetry: axial (2D). 'r' = spherical radius. Use center_z to offset sphere along z-axis.

[+wall] = optional wall region added if wall_thickness parameter is set.

PLAIN DOMAINS (no interior object):
- sphere_domain: For spherically-symmetric profiles (e.g. NFW, isothermal). Regions: domain [+wall]. Fixed symmetry: axial (2D). 'r' = spherical radius.
- disk: For cylindrically-symmetric profiles. Regions: domain [+wall]. Fixed symmetry: axial (2D). 'r' = cylindrical radius.
- box_2d: 2D Cartesian rectangle. Regions: domain. Fixed symmetry: translation (2D).
- box_3d: 3D Cartesian box. Regions: domain. Fixed symmetry: none (true 3D).
- parallel_plates: Two parallel plates with vacuum gap. Regions: vacuum, plate. Fixed symmetry: translation (2D extended in y). 'x' = perpendicular to plates.

CUSTOM SHAPES:
- custom_2d_axial: Arbitrary 2D axisymmetric shape. Points are [r, z] with r >= 0, revolved around z-axis. Regions: object, vacuum [+wall]. Fixed symmetry: axial (2D).
- custom_2d_translation: Arbitrary 2D shape with translation symmetry. Points are [x, y], extruded in z. Regions: object, vacuum [+wall]. Fixed symmetry: translation (2D).
- custom_3d: Arbitrary 3D shape from contours. Regions: object. Fixed symmetry: none (true 3D).
- custom_step: Import 3D geometry from STEP/IGES/BREP file. Object is centered in spherical vacuum domain. Single solid: regions are 'object', 'vacuum'. Multiple solids: regions are 'object_0', 'object_1', ... (sorted by z-centroid, lowest first), plus 'vacuum'. Fixed symmetry: none (true 3D). |
| `params` | object | Yes | Geometry-specific parameters. |
| `mesh_quality` | `very_coarse` | `coarse` | `medium` | `fine` | `very_fine` | No | Mesh resolution. Default: `"medium"` |
| `custom_id` | string | No | Custom mesh ID. |
| `allow_large_mesh` | boolean | No | Allow meshes exceeding 2,000,000 cells. Default: false. Default: `False` |
| `physics_params` | object | No | Physics parameters for automatic thin-shell mesh refinement (recommended when available). Option 1: Provide 'lambda' dict mapping region names to Compton wavelengths. Option 2: Provide 'alpha', 'density' dict, and 'n' - lambdas will be computed per region. The mesh will be refined near boundaries of dense regions to resolve thin shells. |

#### `params` options

- **`object_radius`** (number): Radius of object inside domain. Used by: sphere_in_vacuum, sphere_in_profile, sphere_near_wall, cylinder_in_vacuum.
- **`domain_radius`** (number): Outer boundary radius of simulation domain. Used by: sphere_in_vacuum, ellipse_in_vacuum, cylinder_in_vacuum, shell_in_vacuum, two_spheres, sphere_near_wall, sphere_in_profile, sphere_domain, disk, custom_2d_axial, custom_2d_translation.
- **`center_z`** (number): Z-position of sphere center (sphere_in_profile, default 0)
- **`wall_thickness`** (number): Optional outer wall thickness. Adds 'wall' region if set. Supported by: sphere_in_vacuum, ellipse_in_vacuum, cylinder_in_vacuum, shell_in_vacuum, two_spheres, sphere_in_profile, sphere_domain, disk, custom_2d_axial, custom_2d_translation. For sphere_near_wall, wall is always present (defaults to 0.1 if not specified).
- **`rx`** (number): Semi-axis in r/x direction (ellipse_in_vacuum)
- **`ry`** (number): Semi-axis in z/y direction (ellipse_in_vacuum)
- **`object_height`** (number): Height of cylindrical object (cylinder_in_vacuum)
- **`domain_width`** (number): Width of domain in x direction (box_2d, box_3d)
- **`domain_height`** (number): Height of domain in y direction (box_2d, box_3d, parallel_plates)
- **`domain_depth`** (number): Depth of domain in z direction (box_3d)
- **`inner_radius`** (number): Inner radius for shell_in_vacuum
- **`outer_radius`** (number): Outer radius for shell_in_vacuum
- **`radius_1`** (number): First sphere radius (two_spheres)
- **`radius_2`** (number): Second sphere radius (two_spheres)
- **`separation`** (number): Center-to-center separation (two_spheres)
- **`wall_distance`** (number): Distance from sphere center to wall (sphere_near_wall)
- **`points`** (array[any]): Array of [r,z] points for custom_2d_axial or [x,y] points for custom_2d_translation
- **`shape_file`** (string): Path to file with shape points (custom_2d_axial, custom_2d_translation)
- **`contour_file`** (string): Path to 3D contour file (custom_3d)
- **`contours`** (array[any]): List of contour point lists for custom_3d
- **`step_file`** (string): Path to STEP/IGES/BREP file (custom_step)
- **`plate_separation`** (number): Gap between inner surfaces of plates (parallel_plates)
- **`plate_thickness`** (number): Thickness of each plate (parallel_plates)
- **`measuring_distance`** (number): Distance from object surface to create measuring boundary shell. Creates 'measuring_boundary' region for evaluation of quantities (e.g field gradient) along the boundary). Recommended when you need to evaluate field gradient at a specific distance from the source, as the mesh resolution is increased at this boundary - use with evaluate mode='boundary_max'. Used by: sphere_in_vacuum.

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
- integrate: Compute volume integrals over a region. Returns total force, mass, volume. Essential for torsion balance experiments, Casimir force measurements, and any extended object where thin-shell effects matter.
- boundary_max: Find max gradient magnitude along a region boundary (e.g., measuring_boundary). Useful for finding peak fifth force at a specific distance from source.

Quantities (for point-based modes):
- field: Chameleon field φ
- gradient_magnitude: |∇φ| (dimensionless). Multiply by grad_to_acceleration_g from calculate_physical_parameters to get acceleration in units of g.
- density: ρ̂ at evaluation points (if available)
- adiabatic_field: ρ̂^{-1/(n+1)} for comparison
- field_deviation: (φ - φ_adiabatic) / φ_adiabatic

Quantities (for integrate mode) - all in rescaled (dimensionless) units:
- force: Total force F = ∫ρ̂∇φ̂ dV̂ on region. Returns F_x, F_y (and F_z for 3D). Multiply by force_scale_N to get Newtons.
- torque: Torque τ = ∫(r-r₀)×(ρ̂∇φ̂) dV̂ around torque_origin. Returns τ_x, τ_y, τ_z. Multiply by torque_scale_Nm to get Newton-meters.
- mass: Total mass M = ∫ρ̂ dV̂. Multiply by mass_scale_kg to get kg.
- bounds: Optional axis cuts (x_min, x_max, y_min, y_max, z_min, z_max) to restrict integration to a subregion (e.g., upper disk only).

Note: For screened objects, force/torque contributions come from a thin shell of thickness ~λ near the surface. For accurate integrated quantities, cell_min must be smaller than λ. Check physics_refinement.cell_min vs physics_refinement.lambda_min from create_mesh output.

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `solution_id` | string | Yes | ID of the solution to evaluate |
| `mode` | `radial` | `line` | `points` | `grid` | `max_in_region` | `integrate` | `boundary_max` | Yes | Evaluation mode |
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
- **`region`** (string): Region name (max_in_region, integrate). Default: vacuum
- **`quantity`** (`force` | `torque` | `mass` | `all`): Quantity to integrate (integrate mode). 'force' returns F components, 'torque' returns τ components, 'mass' returns only mass/volume, 'all' returns force+torque. Default: all
- **`torque_origin`** (array[number]): Origin point for torque calculation [x, y, z] (integrate mode). Default: [0, 0, 0]
- **`bounds`** (object): Axis cuts to restrict integration volume (integrate mode). Only cells with midpoint within bounds are included.
  - **`x_min`** (number): 
  - **`x_max`** (number): 
  - **`y_min`** (number): 
  - **`y_max`** (number): 
  - **`z_min`** (number): 
  - **`z_max`** (number): 
- **`min_distance_from`** (string | array[string]): Region(s) to keep distance from (max_in_region). Can be: a region name, 'all' for all other regions, or a list of region names
- **`min_distance`** (number): Minimum distance from boundary of exclusion region(s) (max_in_region)
- **`n_samples`** (integer): Number of random samples (max_in_region). Default: 1000
- **`boundary`** (`outer` | `inner` | `all`): Which boundary to evaluate for shell regions (boundary_max). 'outer' = adjacent to vacuum, 'inner' = adjacent to object. Default: outer
- **`adjacent_to`** (string): Explicit region name that the boundary should be adjacent to (boundary_max). Overrides 'boundary' parameter. Use for non-spherical geometries or custom region names.

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
- field_2d: 2D colormap of field (2D meshes only)
- force_1d: Radial gradient magnitude |∇φ|(r)
- force_2d: 2D colormap of gradient magnitude (2D meshes only)
- comparison: Compare multiple solutions on same plot
- density or density_1d: Radial density profile ρ̂(r) with optional adiabatic field overlay
- density_2d: 2D colormap of density (2D meshes only)
- slice_xy: 2D slice through 3D field in xy-plane at given z
- slice_xz: 2D slice through 3D field in xz-plane at given y
- slice_yz: 2D slice through 3D field in yz-plane at given x

Returns PNG image (base64 or saved to file).

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `solution_id` | string | array[string] | Yes | Solution ID(s) to plot |
| `plot_type` | `field_1d` | `field_2d` | `force_1d` | `force_2d` | `comparison` | `density` | `density_1d` | `density_2d` | `slice_xy` | `slice_xz` | `slice_yz` | Yes | Type of plot to generate |
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
- **`quantity`** (`field` | `gradient_magnitude` | `density`): Quantity to plot in slice (default: field)
- **`legend_by`** (string): Label legend by this field
- **`figsize`** (array[number]): 
- **`title`** (string): Custom title
- **`slice_position`** (number): Position of slice plane (default: 0)
- **`n_grid`** (integer): Grid resolution for slice sampling (default: 100)

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
| `clip` | object | No | Clip plane for 3D meshes to show internal structure. Specify normal and origin. |
| `opacity` | number | No | Opacity for 3D mesh rendering (0-1). Use < 1 to see through outer surface. Default: `1.0` |
| `show_regions` | array[string] | No | Only show these regions (by name). E.g. ['object'] to hide vacuum. Default: show all. |

#### `clip` options

- **`normal`** (array[number]): Normal vector of clip plane, e.g. [1, 0, 0] for x-plane.
- **`origin`** (array[number]): Origin point of clip plane. Default: mesh center.

---

## plot_step

Preview a STEP/IGES/BREP file geometry.

Useful for verifying geometry before running a full simulation. Shows the CAD geometry with different colors for each volume.

Returns an image of the geometry along with metadata (bounding box, number of volumes, region names, etc.).

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `step_file` | string | Yes | Path to STEP/IGES/BREP file |
| `output_path` | string | No | Optional path to save image (PNG). If not provided, returns base64 image. |
| `title` | string | No | Optional plot title |

---

## solve

Solve the chameleon field equation on a mesh with specified density profile.

Uses SELCIE's Picard or Newton solver to compute the chameleon scalar field
throughout the domain. The dimensionless field equation is:
    α ∇²φ + φ^{-(n+1)} = ρ̂

All values are dimensionless. If using physical parameters from calculate_physical_parameters:
- Density: ρ̂ = ρ_physical / rho_0
- Coordinates in mesh: x̂ = x_physical / L

Parameters:
- mesh_id: Reference to a previously created mesh
- alpha: Dimensionless coupling constant (from calculate_physical_parameters)
- density: Dimensionless density ρ̂ = ρ/ρ₀ per region (e.g., if ρ₀ = vacuum density, then vacuum → 1.0)
- n: Potential power (default: 1)
- tol: Convergence tolerance (default: 1e-14)
- max_iter: Maximum iterations (default: 100)
- relaxation: Relaxation factor for Picard iteration (0-1]. In tests 1 performs well, and is faster, so is recommeneded. Default: 1.0
- initial_guess: "constant" (default, recommended for SELCIE), "adiabatic", or "previous"

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `mesh_id` | string | Yes | ID of the mesh to solve on |
| `alpha` | number | Yes | Dimensionless α parameter (coupling constant) |
| `density` | object | Yes | Dimensionless density ρ̂ = ρ/ρ₀ per region. Value is: number, {expression: str}, or {file: str, format?: 'tabulated'\|'grid', columns?: int[], bounds?: number[], skip_header?: int, npz_key?: str}. For tabulated: columns selects columns (1-based). For grid: bounds maps grid to spatial coordinates. For expressions: spherical geometries use r=spherical radius. |
| `n` | integer | No | Potential power index. Default: 1 Default: `1` |
| `tol` | number | No | Convergence tolerance. Default: 1e-14 Default: `1e-14` |
| `max_iter` | integer | No | Maximum iterations. Default: 100 Default: `100` |
| `relaxation` | number | No | Relaxation factor for Picard iteration (0-1]. In tests 1 performs well, and is faster, so is recommeneded. Default: 1.0 Default: `1.0` |
| `initial_guess` | `constant` | `adiabatic` | `previous` | No | Initial guess strategy. Default: constant (recommended for SELCIE) Default: `"constant"` |
| `custom_id` | string | No | Custom solution ID. Default: auto-generated |
| `deg_V` | integer | No | Function space degree (1=CG1, 2=CG2). Default: 2 Default: `2` |

---
