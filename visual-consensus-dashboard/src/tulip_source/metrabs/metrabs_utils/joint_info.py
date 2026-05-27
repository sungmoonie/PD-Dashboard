from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class JointInfo:
    """
    Local copy of posepile.joint_info.JointInfo.
    Keeps metrabs self-contained without external posepile dependency.
    """

    names: list[str]
    edges: np.ndarray
    stick_figure_edges: list[tuple[int, int]]
    mirror_mapping: list[int]
    n_joints: int

    def __init__(self, names: np.ndarray | list[str], edges: np.ndarray):
        self.names = [str(name) for name in names]
        self.edges = np.asarray(edges, dtype=np.int32)
        self.stick_figure_edges = [
            (int(edge[0]), int(edge[1])) for edge in self.edges.reshape(-1, 2)
        ]
        self.n_joints = len(self.names)
        self.mirror_mapping = self._build_mirror_mapping()

    def _build_mirror_mapping(self) -> list[int]:
        name_to_index = {name: idx for idx, name in enumerate(self.names)}
        mapping: list[int] = []
        for idx, name in enumerate(self.names):
            if name.startswith("l"):
                opposite = f"r{name[1:]}"
            elif name.startswith("r"):
                opposite = f"l{name[1:]}"
            else:
                opposite = name
            mapping.append(name_to_index.get(opposite, idx))
        return mapping
