# SELCIE MCP Server - Example Prompts

## calculate_physical_parameters

### Lab experiments

> Calculate the chameleon parameters for a 1cm aluminum sphere (density 2.7 g/cm³) with β=10⁸. The vacuum density is about 10⁻¹⁰ g/cm³.

> What screening regime would a 5mm tungsten bead (19.3 g/cm³) be in with β=10⁶? Assume it's in a vacuum chamber at 10⁻⁸ g/cm³.

> For a tabletop experiment with β=10⁷, Λ=2.4meV, a 2cm steel sphere at 7.8 g/cm³ in vacuum, do I need SELCIE or can I use analytics?

### Astrophysical objects

> What's the screening regime for a solar chameleon with β=1? Use solar core density 150 g/cm³, surface density 10⁻⁶ g/cm³, and radius R_sun.

> Calculate α for a neutron star chameleon: β=1, central density 10¹⁵ g/cm³, radius 10 km.

> For an NFW galaxy halo with β=1, characteristic density 10⁶ M_sun/kpc³, scale radius 20 kpc, what is the screening regime?

### Parameter exploration

> Compare the regimes for β = 1, 10³, 10⁶, 10⁹ with a 1m object at 1 g/cm³.

> How does changing n from 1 to 2 to 4 affect the Compton wavelength for α=1?

> At what β does a 1cm sphere transition from adiabatic to thin-shell regime?

## create_mesh

### Basic geometries

> Create a coarse mesh for a sphere of radius 0.1 in vacuum of radius 1.0

> Create a mesh for an oblate ellipsoid with rx=0.2 and ry=0.1 in vacuum

> Create a shell geometry with inner radius 0.1, outer radius 0.2, and vacuum radius 1.0

> Create a cylinder mesh with radius 0.15 and height 0.4 in vacuum

### Multi-object geometries

> Create a mesh with two spheres - radii 0.12 and 0.08, separated by 0.5, in vacuum of radius 1.0

> Create a sphere near a wall: sphere radius 0.1, wall distance 0.15, wall thickness 0.1, vacuum radius 1.0

### Custom shapes

> Create a 2D hexagonal mesh with vertices at (0.2,0), (0.1,0.173), (-0.1,0.173), (-0.2,0), (-0.1,-0.173), (0.1,-0.173) in vacuum of radius 1.0

> Create a custom star shape from the points file test_data/star.txt in vacuum

### Physics-aware refinement

> Create a sphere mesh with thin shell refinement using physics_params with lambda 0.01 for the object

> Create a sphere mesh with physics refinement using α=1e18, n=1, object density 1e17, vacuum density 1.0

> Create a two-sphere mesh with physics refinement for lambda 0.005 in both spheres

> Compare the cell count of a sphere mesh with and without physics refinement (lambda=0.01 for object)

### Plain domains

> Create a 2D box mesh with width 2.0 and height 1.5

> Create a disk mesh with radius 1.0

> Create a 3D box with dimensions 2x1.5x1

## plot_mesh

### Basic visualization

> Plot the mesh mesh_001

> Create a sphere mesh and show me what it looks like

> Create an ellipse mesh and visualize it without cell edges

### Comparing meshes

> Create a sphere mesh with and without physics refinement and plot both so I can compare

> Create meshes for sphere, ellipse, and cylinder geometries and show me each one

### Custom plot options

> Plot mesh_001 with a custom title "My Experiment Setup"

> Plot the mesh at high resolution (dpi=300) and save to experiment_mesh.png

> Create a two-sphere mesh and plot it with figure size 10x10 inches

## Combined workflows

### Full setup

> I want to study chameleon screening for a 1cm aluminum sphere with β=10⁷. Calculate the parameters, create an appropriate mesh with physics refinement, and show me the geometry.

> Set up a simulation for two spheres (source: radius 0.1, test mass: radius 0.05, separation 0.3) with physics refinement (lambda=0.01 for both spheres). Create the mesh and visualize it.

### Mesh quality iteration

> Create a coarse sphere mesh and plot it. If it looks too coarse, create a medium quality one.

## solve

### Basic solving

> Solve for the chameleon field on mesh_001 with α=3.5 and densities: object=2.7e10, vacuum=1.0

> Solve with α=1.0 using the adiabatic initial guess

> Run a solve with relaxation=0.5 and max_iter=200 for a stiff problem with α=1000

### Expression-based density

> Solve with a radially varying density profile: object density = 1e6/(1 + (r/0.1)^2), vacuum = 1.0

## evaluate

### Radial profiles

> Evaluate the field along the radial direction with 200 points using log spacing

> Get the field and gradient magnitude from r=0.01 to r=0.9

### Specific points

> Evaluate the field at points (0.2, 0), (0.5, 0), and (0.8, 0)

### Grid evaluation

> Evaluate the field on a 50x50 grid from r=0 to 1 and z=-0.5 to 0.5

### Maximum force finding

> Find the maximum gradient magnitude in the vacuum region

> Find the max fifth force in vacuum at least 0.1 away from the object surface

> Find the max fifth force in vacuum at least 0.05 away from ALL other domains (object and wall)

> Find the max gradient in vacuum, keeping at least 0.1 distance from both the object and wall regions

## plot

### 1D profiles

> Plot the radial field profile φ(r) for solution_001

> Plot the gradient magnitude |∇φ|(r) with log scale

> Plot field_1d with log r-axis and 300 points

### 2D visualizations

> Plot the 2D field distribution with the plasma colormap

> Create a 2D plot of the gradient magnitude

### Comparison plots

> Compare solutions solution_001, solution_002, solution_003 on the same plot

## get_state and clear

> Show me what meshes and solutions are in the current session

> Clear all the solutions but keep the meshes

> Clear everything and start fresh

## Full workflow tests

### Sphere in Vacuum (Complete Test, Baseline result here is 3.4e-6)

> I want to simulate a chameleon field around a sphere in a vacuum chamber.
>
> Setup:
> - Sphere with radius 0.1337 (corresponds to volume 0.01)
> - Vacuum chamber radius 1.0, wall thickness 0.05
> - Use α = 1e18 and n = 1
> - Source density: 1e17
> - Vacuum density: 1.0
>
> Please:
> 1. Create a mesh for this geometry (sphere_in_vacuum with object_radius=0.1337)
> 2. Plot the mesh to show the geometry
> 3. Solve for the chameleon field
> 4. Plot the radial field profile φ(r) and gradient magnitude |∇φ|(r) w
> 5. Plot the 2D density, field distribution, and gradient magnitude
> 6. Find the max fifth gradient in vacuum at least 0.05 away from ALL other domains (object and wall)
> 7. Save the plots in this directory

### Sphere in Vacuum (Physical Parameters)

> I want to simulate the chameleon field around an aluminum sphere in a vacuum chamber, starting from physical parameters.
>
> Physical setup:
> - Aluminum sphere: radius 1 cm, density 2.7 g/cm³
> - Vacuum chamber: radius 10 cm, wall thickness 0.5 cm
> - Chamber wall: stainless steel, density 8.0 g/cm³
> - Vacuum: density 10⁻¹⁷ g/cm³ (good lab vacuum)
> - Chameleon coupling: β = 10⁶, Λ = 2.4 meV, n = 1
>
> Please:
> 1. Report the screening regime and Compton wavelengths
> 2. Create a mesh with physics-aware refinement using the computed parameters
> 3. Solve for the chameleon field
> 4. Plot the radial field profile and gradient magnitude
> 5. Find the maximum fifth force acceleration in the vacuum region (at least 0.5 cm from all surfaces)
> 6. Convert the result back to physical units (m/s²)

### Two-Sphere Interaction

> I want to study the chameleon-mediated force between two spheres.
>
> Setup:
> - Source sphere: radius 2 cm, tungsten (19.3 g/cm³)
> - Test mass: radius 0.5 cm, aluminum (2.7 g/cm³)
> - Separation (center to center): 5 cm
> - Vacuum chamber radius: 20 cm
> - Vacuum density: 10⁻¹⁰ g/cm³
> - Chameleon: β = 10⁸, n = 1
>
> Calculate the field, show me the 2D field distribution, and find the maximum force on the test mass.

### Solar Chameleon

> I want to model the chameleon field profile inside and around the Sun using the Standard Solar Model.
>
> Setup:
> - Use the AGSS09 solar model from `AGSS09_solar_model.dat`
> - Sun: radius R☉ = 7×10⁸ m
> - Solar atmosphere/corona: extend to 2 R☉, density ~10⁻¹² g/cm³
> - Chameleon: β = 1 (gravitational strength coupling), n = 1
>
> Please:
> 1. Report whether the Sun is in the thin-shell or thick-shell regime
> 2. Create a mesh for sphere_in_vacuum with the Sun as the object
> 3. Solve using the tabulated density profile
> 4. Plot the radial field profile from the center to 2 R☉
> 5. Find the maximum fifth force in the corona region

### Neutron Star

> Model the chameleon field around a neutron star to check screening.
>
> Setup:
> - Neutron star: radius 10 km, average density 5×10¹⁴ g/cm³
> - Surrounding medium: interstellar medium at 10⁻²⁴ g/cm³, extend to 100 km
> - Chameleon: β = 1, n = 1
>
> I expect extreme screening. Calculate the field profile and report the thin-shell thickness relative to the star radius.

### Galaxy Halo (NFW Profile)

> Model the chameleon field in a Milky Way-like dark matter halo with an NFW density profile.
>
> Setup:
> - NFW profile: ρ(r) = ρ_s / [(r/r_s)(1 + r/r_s)²]
> - Scale radius: r_s = 20 kpc
> - Characteristic density: ρ_s = 10⁻²⁵ g/cm³
> - Domain: extend to 200 kpc (10 × r_s)
> - Chameleon: β = 1, n = 1
>
> Use the NFW density as an expression. Plot the chameleon field profile and compare how it tracks the local density minimum.

### N-body Cosmological Simulation

> Model the chameleon field in a 2D slice from an N-body cosmological simulation.
>
> Setup:
> - Density grid from `nbody.npy` (units: M☉/Mpc³)
> - Box size: 25 Mpc × 25 Mpc
> - Chameleon: β = 10^{-4}, n = 1
>
> Please:
> 1. Calculate physical parameters using cosmic mean density as reference
> 2. Create a 2D Cartesian box mesh covering [0, 25] × [0, 25] Mpc
> 3. Plot the 2D field distribution overlaid on the density structure
> 4. Compare with the adiabatic solution 
> 4. Identify where the field is screened vs unscreened

### R(theta) optimisation

> I want to optimise the shape of an source with axial symmetry, parameterised by R(theta)
>
> Setup:
> - Fixed volume of 0.01
> - 20 equally spaced theta values 
> - Vacuum chamber radius 1.0, wall thickness 0.05
> - Use α = 1e18 and n = 1
> - Source density: 1e17
> - Vacuum density: 1.0
> 
> Find the optimal fifth force at a minimum distance of 0.05 from the source and wall
> 
> Explain your reasoning process. Save the results in an optimal directory. 