#!/usr/bin/env python3
"""Benchmark local gradient evaluation vs projection.

This script demonstrates that local gradient evaluation (using basis function
derivatives) is much faster than global projection, especially on large meshes
where projection can fail due to memory constraints.
"""

import time
import numpy as np


def evaluate_gradient_locally(field, point, mesh, tree=None):
    """Evaluate gradient at a point using local basis function derivatives.

    This avoids global projection by computing the gradient directly from
    the element's shape function derivatives at the given point.

    Args:
        field: FEniCS Function
        point: array-like coordinates
        mesh: FEniCS Mesh
        tree: Optional pre-built BoundingBoxTree for efficiency

    Returns:
        numpy array of gradient components, or NaN if point outside mesh
    """
    import dolfin as d

    dim = mesh.geometry().dim()
    point = np.asarray(point, dtype=float)

    if tree is None:
        tree = mesh.bounding_box_tree()

    point_obj = d.Point(*point[:dim])
    cell_id = tree.compute_first_entity_collision(point_obj)

    if cell_id >= mesh.num_cells():
        return np.array([np.nan] * dim)

    cell = d.Cell(mesh, cell_id)
    V = field.function_space()
    element = V.element()
    dofmap = V.dofmap()

    cell_dofs = dofmap.cell_dofs(cell_id)
    dof_values = field.vector().get_local()[cell_dofs]
    coordinate_dofs = np.array(cell.get_vertex_coordinates(), dtype=float)
    n_basis = element.space_dimension()

    derivs = element.evaluate_basis_derivatives_all(1, point, coordinate_dofs, cell.orientation())
    basis_derivatives = derivs.reshape(n_basis, dim)
    gradient = np.dot(dof_values, basis_derivatives)

    return gradient


def validate_2d_gradient():
    """Validate local gradient evaluation on a 2D mesh with known function."""
    import dolfin as d

    print("=" * 60)
    print("Validation 1: 2D gradient")
    print("=" * 60)

    mesh = d.UnitSquareMesh(10, 10)
    V = d.FunctionSpace(mesh, "CG", 2)

    # f(x,y) = x^2 + 2*y  =>  grad(f) = [2x, 2]
    f = d.interpolate(d.Expression("x[0]*x[0] + 2*x[1]", degree=2), V)

    test_points = [[0.5, 0.5], [0.25, 0.75], [0.1, 0.1]]
    tree = mesh.bounding_box_tree()

    print("\nFunction: f(x,y) = x^2 + 2y")
    print("Expected gradient: [2x, 2]")
    print("-" * 60)

    all_pass = True
    for pt in test_points:
        grad_local = evaluate_gradient_locally(f, pt, mesh, tree)
        expected = np.array([2 * pt[0], 2.0])
        error = np.linalg.norm(grad_local - expected)

        status = "PASS" if error < 1e-10 else "FAIL"
        if error >= 1e-10:
            all_pass = False

        print(f"Point {pt}: grad={grad_local}, expected={expected}, error={error:.2e} [{status}]")

    return all_pass


def validate_3d_gradient():
    """Validate local gradient evaluation on a 3D mesh with known function."""
    import dolfin as d

    print("\n" + "=" * 60)
    print("Validation 2: 3D gradient")
    print("=" * 60)

    mesh = d.UnitCubeMesh(5, 5, 5)
    V = d.FunctionSpace(mesh, "CG", 2)

    # f(x,y,z) = x^2 + 2*y + 3*z  =>  grad(f) = [2x, 2, 3]
    f = d.interpolate(d.Expression("x[0]*x[0] + 2*x[1] + 3*x[2]", degree=2), V)

    test_points = [[0.5, 0.5, 0.5], [0.25, 0.75, 0.1], [0.1, 0.1, 0.9]]
    tree = mesh.bounding_box_tree()

    print("\nFunction: f(x,y,z) = x^2 + 2y + 3z")
    print("Expected gradient: [2x, 2, 3]")
    print("-" * 60)

    all_pass = True
    for pt in test_points:
        grad_local = evaluate_gradient_locally(f, pt, mesh, tree)
        expected = np.array([2 * pt[0], 2.0, 3.0])
        error = np.linalg.norm(grad_local - expected)

        status = "PASS" if error < 1e-10 else "FAIL"
        if error >= 1e-10:
            all_pass = False

        print(f"Point {pt}: error={error:.2e} [{status}]")

    return all_pass


def benchmark_3d(n_cells=20):
    """Benchmark local vs projection gradient on a 3D mesh.

    Args:
        n_cells: Number of cells per dimension (total cells ~ n_cells^3 * 6)
    """
    import dolfin as d

    print("\n" + "=" * 60)
    print(f"Benchmark: 3D mesh ({n_cells}x{n_cells}x{n_cells})")
    print("=" * 60)

    # Create mesh and function
    print("\nCreating mesh...")
    mesh = d.UnitCubeMesh(n_cells, n_cells, n_cells)
    print(f"  Cells: {mesh.num_cells():,}, Vertices: {mesh.num_vertices():,}")

    V = d.FunctionSpace(mesh, "CG", 2)
    print(f"  DOFs: {V.dim():,}")

    # Create a more complex test function
    f = d.interpolate(
        d.Expression("sin(pi*x[0])*cos(pi*x[1])*exp(x[2])", degree=3, pi=np.pi),
        V
    )

    # Generate random test points
    np.random.seed(42)
    n_points = 1000
    test_points = np.random.uniform(0.05, 0.95, (n_points, 3))

    print(f"\nEvaluating gradient at {n_points} random points...")

    # Method 1: Local gradient
    print("\n--- Method 1: Local gradient (no projection) ---")
    tree = mesh.bounding_box_tree()

    t0 = time.time()
    grad_local = []
    for pt in test_points:
        grad_local.append(evaluate_gradient_locally(f, pt, mesh, tree))
    t_local = time.time() - t0

    print(f"  Time: {t_local:.3f}s ({t_local/n_points*1000:.3f}ms per point)")

    # Method 2: Projection
    print("\n--- Method 2: Projection (global solve) ---")

    try:
        t0 = time.time()
        V_vec = d.VectorFunctionSpace(mesh, "CG", 2)
        grad_projected = d.project(d.grad(f), V_vec)
        t_project = time.time() - t0

        t0 = time.time()
        grad_proj = []
        for pt in test_points:
            try:
                grad_proj.append(np.array(grad_projected(*pt)))
            except RuntimeError:
                grad_proj.append(np.array([np.nan, np.nan, np.nan]))
        t_eval = time.time() - t0

        print(f"  Projection time: {t_project:.3f}s")
        print(f"  Evaluation time: {t_eval:.3f}s")
        print(f"  Total time: {t_project + t_eval:.3f}s")

        # Compare results
        diffs = []
        for g_l, g_p in zip(grad_local, grad_proj):
            if not np.any(np.isnan(g_l)) and not np.any(np.isnan(g_p)):
                diffs.append(np.linalg.norm(g_l - g_p))

        if diffs:
            print(f"\n  Agreement: mean diff={np.mean(diffs):.2e}, max diff={np.max(diffs):.2e}")

        speedup = (t_project + t_eval) / t_local
        print(f"\n  Speedup: {speedup:.1f}x faster with local method")

    except Exception as e:
        print(f"  Projection FAILED: {e}")
        print("  (This demonstrates why local evaluation is needed for large meshes)")

    return True


if __name__ == "__main__":
    import sys

    # Run validation tests
    validate_2d_gradient()
    validate_3d_gradient()

    # Run benchmark
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 15
    benchmark_3d(n_cells=n)
