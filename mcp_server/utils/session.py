"""Session state management for SELCIE MCP server."""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class MeshInfo:
    """Information about a stored mesh."""
    mesh_id: str
    geometry: str
    dimension: int
    n_cells: int
    n_vertices: int
    symmetry: str
    mesh_path: str  # Path to the xdmf files
    params: dict
    regions: dict[str, int] = field(default_factory=dict)  # region name -> marker mapping
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class SolutionInfo:
    """Information about a stored solution."""
    solution_id: str
    mesh_id: str
    profile_id: str
    alpha: float
    n: int
    converged: bool
    iterations: int
    final_residual: float
    field_min: float
    field_max: float
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


class Session:
    """
    Manages session state for SELCIE MCP server.

    Stores meshes and solutions with auto-generated IDs.
    """

    def __init__(self):
        self.session_id = str(uuid.uuid4())[:8]
        self.created_at = datetime.now().isoformat()

        self.meshes: dict[str, MeshInfo] = {}
        self.solutions: dict[str, SolutionInfo] = {}

        # Counters for auto-generated IDs
        self._mesh_counter = 0
        self._solution_counter = 0

    def generate_mesh_id(self, custom_id: str | None = None) -> str:
        """Generate a unique mesh ID."""
        if custom_id:
            if custom_id in self.meshes:
                raise ValueError(f"Mesh ID '{custom_id}' already exists")
            return custom_id

        self._mesh_counter += 1
        return f"mesh_{self._mesh_counter:03d}"

    def generate_solution_id(self, custom_id: str | None = None) -> str:
        """Generate a unique solution ID."""
        if custom_id:
            if custom_id in self.solutions:
                raise ValueError(f"Solution ID '{custom_id}' already exists")
            return custom_id

        self._solution_counter += 1
        return f"solution_{self._solution_counter:03d}"

    def add_mesh(self, info: MeshInfo) -> None:
        """Add a mesh to the session."""
        self.meshes[info.mesh_id] = info

    def add_solution(self, info: SolutionInfo) -> None:
        """Add a solution to the session."""
        self.solutions[info.solution_id] = info

    def get_mesh(self, mesh_id: str) -> MeshInfo | None:
        """Get mesh info by ID."""
        return self.meshes.get(mesh_id)

    def get_solution(self, solution_id: str) -> SolutionInfo | None:
        """Get solution info by ID."""
        return self.solutions.get(solution_id)

    def clear(self, what: str = "all", ids: list[str] | None = None) -> list[str]:
        """
        Clear session objects.

        Parameters
        ----------
        what : str
            "all", "meshes", or "solutions"
        ids : list[str] | None
            Specific IDs to clear. If None, clears all of the specified type.

        Returns
        -------
        list[str]
            List of cleared IDs.
        """
        cleared = []

        if ids:
            # Clear specific IDs
            for obj_id in ids:
                if obj_id in self.meshes:
                    del self.meshes[obj_id]
                    cleared.append(obj_id)
                elif obj_id in self.solutions:
                    del self.solutions[obj_id]
                    cleared.append(obj_id)
        else:
            # Clear by type
            if what in ("all", "solutions"):
                cleared.extend(list(self.solutions.keys()))
                self.solutions.clear()

            if what in ("all", "meshes"):
                cleared.extend(list(self.meshes.keys()))
                self.meshes.clear()

        return cleared

    def to_dict(self) -> dict[str, Any]:
        """Convert session state to dictionary for JSON serialization."""
        return {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "meshes": [
                {
                    "mesh_id": m.mesh_id,
                    "geometry": m.geometry,
                    "n_cells": m.n_cells,
                    "dimension": m.dimension,
                    "regions": m.regions,
                }
                for m in self.meshes.values()
            ],
            "solutions": [
                {
                    "solution_id": s.solution_id,
                    "mesh_id": s.mesh_id,
                    "alpha": s.alpha,
                    "converged": s.converged,
                }
                for s in self.solutions.values()
            ],
        }


# Global session instance
_session: Session | None = None


def get_session() -> Session:
    """Get the global session instance, creating it if needed."""
    global _session
    if _session is None:
        _session = Session()
    return _session


def reset_session() -> Session:
    """Reset the global session (for testing)."""
    global _session
    _session = Session()
    return _session
