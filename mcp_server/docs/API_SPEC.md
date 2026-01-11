# SELCIE MCP Server API Specification

## Overview

This document specifies the MCP (Model Context Protocol) server interface for SELCIE, enabling LLMs to perform chameleon field calculations and physics exploration.

**Version:** 1.0.0
**Protocol:** MCP over stdio
**Backend:** SELCIE (FEniCS-based chameleon field solver)

For detailed tool parameters and options, see [TOOLS.md](TOOLS.md).

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

## Physics Background

SELCIE solves the dimensionless chameleon field equation:
```
α ∇²φ + φ^{-(n+1)} = ρ̂
```

where ρ̂ = ρ/ρ₀ is the dimensionless density (ρ₀ is the reference density used to compute α).

The dimensionless α parameter encapsulates all physical parameters:
```
α = (M·Λ)/(ρ₀·L²) × [(n·M·Λ³)/ρ₀]^{1/(n+1)}
```

where M = M_pl/β.

The **dimensionless Compton wavelength** determines field behavior (from SELCIE paper arXiv:2110.11917):
```
λ̂(ρ̂) = √(α/(n+1)) × ρ̂^{-(n+2)/(2(n+1))}
```

Since λ̂ is defined relative to the characteristic length scale L (typically the object radius), compare λ̂ to 1:

- **λ̂ << 1**: Compton wavelength << L → field tracks local density → φ ≈ ρ̂^{-1/(n+1)} (adiabatic)
- **λ̂ >> 1**: Compton wavelength >> L → field cannot adjust → set by boundaries/environment
- **λ̂ ~ 1**: Transition region, requires numerical solution (SELCIE)

The thin shell forms where λ̂ becomes comparable to the distance to the surface.

**Caveat**: This assumes objects are comparable in size to L. If an object is much smaller than L, compare λ̂ to the object size (in units of L) instead. The field can only relax to the adiabatic solution if the Compton wavelength is smaller than the object.

### Fifth Force on a Test Mass

The acceleration on a test particle due to the chameleon field is:
```
a_φ = -(β/M_pl) ∇φ
```

or equivalently the force:
```
F_φ = -m(β/M_pl) ∇φ
```

where m is the test mass, β is the matter coupling, and M_pl is the Planck mass. 

### Regime Classification

| λ̂(ρ_max) | λ̂(ρ_min) | Regime | SELCIE needed? | Behavior |
|-----------|-----------|--------|----------------|----------|
| << 1 | << 1 | `adiabatic` | No | φ = ρ^{-1/(n+1)} everywhere |
| << 1 | ~ 1 or > 1 | `thin_shell` | Yes | Screened core, transition shell |
| ~ 1 | ~ 1 | `transition` | Yes | Intermediate, complex behavior |
| >> 1 | >> 1 | `constant_field` | No | φ ≈ constant everywhere |

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
| `SOLUTION_NOT_FOUND` | Referenced solution doesn't exist |
| `SOLVER_DIVERGED` | Solver failed to converge |
| `INVALID_GEOMETRY` | Geometry specification invalid |
| `INVALID_PARAMETER` | Parameter out of valid range |
| `FILE_NOT_FOUND` | Tabulated data file not found |

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

# Solve
solution = solve(
    mesh_id=mesh["mesh_id"],
    alpha=params["alpha"],
    density={"object": 2.7e10, "vacuum": 1.0}
)

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
```
