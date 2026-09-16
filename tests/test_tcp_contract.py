from pathlib import Path
import inspect
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from atomic_latent_vla.annotation.lerobot import LeRobotEpisode
from atomic_latent_vla.annotation.motion import (
    CartesianPose,
    JointMotionTrace,
    LocalOffsetFK,
)
from atomic_latent_vla.tcp import (
    BIMANUAL_TCP_POSE_METADATA,
    BIMANUAL_TCP_POSE_SIDECAR,
    BIMANUAL_TCP_LOCAL_Z_OFFSET_M,
    TCP_LOCAL_Z_OFFSET_M,
    TCP_OFFSET_TAG,
    TCP_POSE_METADATA,
    TCP_POSE_SIDECAR,
)


def test_tcp_contract_uses_10cm_and_new_sidecar_identity() -> None:
    assert TCP_LOCAL_Z_OFFSET_M == pytest.approx(0.10)
    assert TCP_OFFSET_TAG == "tcp100"
    assert TCP_POSE_SIDECAR == "tcp_pose_right_base_tcp100.npy"
    assert TCP_POSE_METADATA == "tcp_pose_right_base_tcp100.json"
    assert BIMANUAL_TCP_LOCAL_Z_OFFSET_M == pytest.approx(0.20)
    assert BIMANUAL_TCP_POSE_SIDECAR == "tcp_pose_bimanual_base_tcp200.npy"
    assert BIMANUAL_TCP_POSE_METADATA == "tcp_pose_bimanual_base_tcp200.json"


def test_bundled_urdf_tcp_joint_matches_shared_contract() -> None:
    root = Path(__file__).resolve().parents[1]
    urdf = root / "assets/cr1_right_arm_tcp_red/urdf/CR1ARMR_TCP_RED.urdf"
    robot = ET.parse(urdf).getroot()
    joint = next(item for item in robot.findall("joint") if item.get("name") == "tcp_joint")
    origin = joint.find("origin")
    assert origin is not None
    xyz = [float(value) for value in origin.attrib["xyz"].split()]
    assert xyz == pytest.approx([0.0, 0.0, TCP_LOCAL_Z_OFFSET_M])


def test_default_trace_builders_use_shared_offset_contract() -> None:
    joint_default = inspect.signature(JointMotionTrace.load_hdf5).parameters[
        "tcp_frame"
    ].default
    lerobot_default = inspect.signature(LeRobotEpisode.make_trace).parameters[
        "tcp_frame"
    ].default
    assert joint_default is None
    assert lerobot_default is None

    class IdentityFK:
        def pose(self, q: np.ndarray) -> CartesianPose:
            del q
            return CartesianPose(np.asarray([1.0, 2.0, 3.0]), np.eye(3))

    fk = LocalOffsetFK(IdentityFK(), (0.0, 0.0, TCP_LOCAL_Z_OFFSET_M))
    pose = fk.pose(np.zeros(7))
    np.testing.assert_allclose(pose.translation, [1.0, 2.0, 3.10])
