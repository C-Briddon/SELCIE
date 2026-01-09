#!/usr/bin/env python3
"""Tests for distance filtering optimization in evaluate tool."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
import numpy as np
import pytest


def naive_distance_filter(target_centers, source_boundary_points, min_distance):
    """Original O(N × M) naive implementation."""
    valid_mask = np.ones(len(target_centers), dtype=bool)
    for i, pt in enumerate(target_centers):
        distances = np.linalg.norm(source_boundary_points - pt, axis=1)
        if distances.min() < min_distance:
            valid_mask[i] = False
    return valid_mask


def kdtree_distance_filter(target_centers, source_boundary_points, min_distance):
    """Optimized O(N log M) KD-tree implementation."""
    from scipy.spatial import cKDTree
    tree = cKDTree(source_boundary_points)
    distances, _ = tree.query(target_centers, k=1)
    valid_mask = distances >= min_distance
    return valid_mask


class TestDistanceFilteringAccuracy:
    """Test that KD-tree method produces identical results to naive method."""

    def test_small_dataset_identical_results(self):
        """Test accuracy on small dataset."""
        np.random.seed(42)
        target_centers = np.random.rand(100, 2)
        source_points = np.random.rand(50, 2)
        min_distance = 0.1

        naive_result = naive_distance_filter(target_centers, source_points, min_distance)
        kdtree_result = kdtree_distance_filter(target_centers, source_points, min_distance)

        assert np.array_equal(naive_result, kdtree_result), "Results should be identical"

    def test_medium_dataset_identical_results(self):
        """Test accuracy on medium dataset."""
        np.random.seed(123)
        target_centers = np.random.rand(1000, 2)
        source_points = np.random.rand(500, 2)
        min_distance = 0.05

        naive_result = naive_distance_filter(target_centers, source_points, min_distance)
        kdtree_result = kdtree_distance_filter(target_centers, source_points, min_distance)

        assert np.array_equal(naive_result, kdtree_result), "Results should be identical"

    def test_edge_case_no_exclusions(self):
        """Test when no points should be excluded."""
        np.random.seed(42)
        # Points far apart
        target_centers = np.array([[0.0, 0.0], [0.1, 0.1]])
        source_points = np.array([[1.0, 1.0], [1.1, 1.1]])
        min_distance = 0.1

        naive_result = naive_distance_filter(target_centers, source_points, min_distance)
        kdtree_result = kdtree_distance_filter(target_centers, source_points, min_distance)

        assert np.array_equal(naive_result, kdtree_result)
        assert all(naive_result), "All points should be valid (far from source)"

    def test_edge_case_all_excluded(self):
        """Test when all points should be excluded."""
        target_centers = np.array([[0.0, 0.0], [0.1, 0.1]])
        source_points = np.array([[0.01, 0.01], [0.11, 0.11]])
        min_distance = 0.5  # Large distance excludes all

        naive_result = naive_distance_filter(target_centers, source_points, min_distance)
        kdtree_result = kdtree_distance_filter(target_centers, source_points, min_distance)

        assert np.array_equal(naive_result, kdtree_result)
        assert not any(naive_result), "All points should be excluded"

    def test_boundary_distance(self):
        """Test points exactly at the boundary distance."""
        target_centers = np.array([[0.0, 0.0], [0.2, 0.0]])
        source_points = np.array([[0.1, 0.0]])  # Distance 0.1 from first, 0.1 from second
        min_distance = 0.1

        naive_result = naive_distance_filter(target_centers, source_points, min_distance)
        kdtree_result = kdtree_distance_filter(target_centers, source_points, min_distance)

        assert np.array_equal(naive_result, kdtree_result)
        # Points at exactly min_distance should be included (>= comparison)
        assert all(naive_result), "Points at exactly min_distance should be included"


class TestDistanceFilteringPerformance:
    """Test that KD-tree method is faster than naive method."""

    def test_speedup_medium_mesh(self):
        """Test speedup on medium-sized mesh (typical use case)."""
        np.random.seed(42)
        n_target = 5000
        n_source = 1000
        target_centers = np.random.rand(n_target, 2)
        source_points = np.random.rand(n_source, 2)
        min_distance = 0.05

        # Time naive method
        start = time.perf_counter()
        naive_result = naive_distance_filter(target_centers, source_points, min_distance)
        naive_time = time.perf_counter() - start

        # Time KD-tree method
        start = time.perf_counter()
        kdtree_result = kdtree_distance_filter(target_centers, source_points, min_distance)
        kdtree_time = time.perf_counter() - start

        # Verify results match
        assert np.array_equal(naive_result, kdtree_result), "Results must match"

        # Calculate speedup
        speedup = naive_time / kdtree_time

        print(f"\nMedium mesh ({n_target} targets, {n_source} sources):")
        print(f"  Naive:  {naive_time*1000:.2f} ms")
        print(f"  KDTree: {kdtree_time*1000:.2f} ms")
        print(f"  Speedup: {speedup:.1f}x")

        # KD-tree should be at least 2x faster for this size
        assert speedup > 2, f"Expected at least 2x speedup, got {speedup:.1f}x"

    def test_speedup_large_mesh(self):
        """Test speedup on large mesh."""
        np.random.seed(42)
        n_target = 20000
        n_source = 5000
        target_centers = np.random.rand(n_target, 2)
        source_points = np.random.rand(n_source, 2)
        min_distance = 0.02

        # Time naive method
        start = time.perf_counter()
        naive_result = naive_distance_filter(target_centers, source_points, min_distance)
        naive_time = time.perf_counter() - start

        # Time KD-tree method
        start = time.perf_counter()
        kdtree_result = kdtree_distance_filter(target_centers, source_points, min_distance)
        kdtree_time = time.perf_counter() - start

        # Verify results match
        assert np.array_equal(naive_result, kdtree_result), "Results must match"

        # Calculate speedup
        speedup = naive_time / kdtree_time

        print(f"\nLarge mesh ({n_target} targets, {n_source} sources):")
        print(f"  Naive:  {naive_time*1000:.2f} ms")
        print(f"  KDTree: {kdtree_time*1000:.2f} ms")
        print(f"  Speedup: {speedup:.1f}x")

        # KD-tree should be at least 10x faster for this size
        assert speedup > 10, f"Expected at least 10x speedup, got {speedup:.1f}x"

    def test_speedup_very_large_mesh(self):
        """Test speedup on very large mesh (stress test)."""
        np.random.seed(42)
        n_target = 50000
        n_source = 10000
        target_centers = np.random.rand(n_target, 2)
        source_points = np.random.rand(n_source, 2)
        min_distance = 0.01

        # Time naive method (this will be slow!)
        start = time.perf_counter()
        naive_result = naive_distance_filter(target_centers, source_points, min_distance)
        naive_time = time.perf_counter() - start

        # Time KD-tree method
        start = time.perf_counter()
        kdtree_result = kdtree_distance_filter(target_centers, source_points, min_distance)
        kdtree_time = time.perf_counter() - start

        # Verify results match
        assert np.array_equal(naive_result, kdtree_result), "Results must match"

        # Calculate speedup
        speedup = naive_time / kdtree_time

        print(f"\nVery large mesh ({n_target} targets, {n_source} sources):")
        print(f"  Naive:  {naive_time*1000:.2f} ms")
        print(f"  KDTree: {kdtree_time*1000:.2f} ms")
        print(f"  Speedup: {speedup:.1f}x")

        # KD-tree should be dramatically faster for this size
        assert speedup > 50, f"Expected at least 50x speedup, got {speedup:.1f}x"


class TestIntegrationWithEvaluate:
    """Integration tests with actual mesh and evaluate tool."""

    @pytest.fixture(autouse=True)
    def reset(self):
        """Reset session before each test."""
        from utils.session import reset_session
        reset_session()

    @pytest.mark.asyncio
    async def test_max_in_region_with_distance_filter(self):
        """Test max_in_region mode with distance filtering on real mesh."""
        from tools.create_mesh import handle as create_mesh
        from tools.solve import handle as solve
        from tools.evaluate import handle as evaluate
        import json

        # Create mesh
        mesh_result = await create_mesh({
            "geometry": "sphere_in_vacuum",
            "params": {
                "object_radius": 0.15,
                "vacuum_radius": 1.0,
            },
            "mesh_quality": "coarse",
        })
        mesh_data = json.loads(mesh_result[0].text)
        mesh_id = mesh_data["mesh_id"]

        # Solve
        solve_result = await solve({
            "mesh_id": mesh_id,
            "alpha": 1.0,
            "density": {
                "object": 1e6,
                "vacuum": 1.0,
            },
        })
        solve_data = json.loads(solve_result[0].text)
        solution_id = solve_data["solution_id"]

        # Evaluate with distance filter - should be fast now
        start = time.perf_counter()
        eval_result = await evaluate({
            "solution_id": solution_id,
            "mode": "max_in_region",
            "quantities": ["gradient_magnitude"],
            "params": {
                "region": "vacuum",
                "min_distance_from": "object",
                "min_distance": 0.05,
                "n_samples": 1000,
            }
        })
        eval_time = time.perf_counter() - start

        eval_data = json.loads(eval_result[0].text)

        # Check result is valid
        assert "error" not in eval_data, f"Evaluate failed: {eval_data}"
        assert "gradient_magnitude" in eval_data["data"]
        assert eval_data["data"]["gradient_magnitude"]["max"] > 0

        print(f"\nIntegration test:")
        print(f"  Evaluate time: {eval_time*1000:.2f} ms")
        print(f"  Max gradient: {eval_data['data']['gradient_magnitude']['max']:.6f}")
        print(f"  Valid samples: {eval_data['n_valid_samples']}")

        # Should complete in reasonable time (< 2 seconds)
        assert eval_time < 2.0, f"Evaluate took too long: {eval_time:.2f}s"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
