from __future__ import annotations

from posepile.joint_info import JointInfo


def get_joint_info(_skeleton_name: str) -> JointInfo:
    raise NotImplementedError(
        "posepile.datasets3d.get_joint_info is not required for this inference pipeline."
    )

