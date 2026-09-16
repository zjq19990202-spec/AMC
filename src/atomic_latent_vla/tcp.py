from __future__ import annotations

from typing import Final


# One authoritative CR1 tool-center-point contract for FK annotation and Q1.
TCP_LOCAL_Z_OFFSET_M: Final[float] = 0.10
TCP_OFFSET_TAG: Final[str] = "tcp100"
TCP_POSE_SIDECAR: Final[str] = f"tcp_pose_right_base_{TCP_OFFSET_TAG}.npy"
TCP_POSE_METADATA: Final[str] = f"tcp_pose_right_base_{TCP_OFFSET_TAG}.json"
# The compact target corpus was annotated with a 0.20 m bimanual TCP. Keep it
# distinct from the legacy/right-only tcp100 contract so their labels and FK
# regression targets can never be mixed silently.
BIMANUAL_TCP_LOCAL_Z_OFFSET_M: Final[float] = 0.20
BIMANUAL_TCP_OFFSET_TAG: Final[str] = "tcp200"
BIMANUAL_TCP_POSE_SIDECAR: Final[str] = f"tcp_pose_bimanual_base_{BIMANUAL_TCP_OFFSET_TAG}.npy"
BIMANUAL_TCP_POSE_METADATA: Final[str] = f"tcp_pose_bimanual_base_{BIMANUAL_TCP_OFFSET_TAG}.json"
