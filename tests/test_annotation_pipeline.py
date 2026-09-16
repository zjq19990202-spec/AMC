from pathlib import Path

import cv2
import numpy as np

from atomic_latent_vla.annotation.batch_cli import load_manifest
from atomic_latent_vla.annotation.lerobot import load_lerobot_episode
from atomic_latent_vla.annotation.mirror import (
    CR1_LEFT_TO_RIGHT_STATE_SIGN,
    flip_frame_horizontal,
    mirror_left_arm_values,
    mirror_left_from_bimanual,
    write_mirrored_episode_bundle,
)
from atomic_latent_vla.annotation.motion import (
    base_frame_pose_delta,
    CartesianPose,
    JointMotionTrace,
    MotionTrace,
    tcp_twist_delta_sequence,
)
from atomic_latent_vla.annotation.pipeline import (
    AtomicSegmentationPipeline,
    PipelineConfig,
    _instruction_has_direction_grounding,
    _instruction_has_natural_direction,
    _instruction_has_signed_axis,
)
from atomic_latent_vla.annotation.prompts import proposal_user_text, system_prompt
from atomic_latent_vla.annotation.schema import CandidateAnnotation, CandidateSegment
from atomic_latent_vla.annotation.timeline import (
    FixedAtomicSegment,
    _classify_window,
    _local_ratio_blocks,
    _scaled_score_probabilities,
    _step_atomic_label,
    _top2_evidence,
    build_fk_atomic_timeline,
)
from atomic_latent_vla.annotation.video import SampledVideo, sample_synchronized_videos
from atomic_latent_vla.atomic import AtomicSkill
from atomic_latent_vla.data.gating import AtomicGateDecision, classify_atomic_targets


def _fixed_segment(
    probabilities: list[float],
    *,
    start_s: float = 0.0,
    end_s: float = 3.0,
    mode: str = "single",
    labels: tuple[int, ...] = (0,),
) -> FixedAtomicSegment:
    return FixedAtomicSegment(
        segment_id=0,
        start_s=start_s,
        end_s=end_s,
        atomic_probabilities=tuple(probabilities),
        activity_score=1.0,
        decision=AtomicGateDecision(mode, labels, (1.0,) * len(labels), "test FK gate"),
    )


def test_retained_atom_requires_explicit_signed_axis() -> None:
    assert _instruction_has_signed_axis(
        "Move toward the screw hole along base-frame +y.", "move_y_pos"
    )
    assert _instruction_has_signed_axis(
        "Rotate positively about the base-frame z axis.", "rotate_z_pos"
    )
    assert not _instruction_has_signed_axis(
        "Hold the screwdriver steady while aligning its tip.", "move_y_pos"
    )


def test_retained_atom_accepts_target_relative_direction() -> None:
    assert _instruction_has_direction_grounding(
        "Move the screwdriver toward the visible screw hole left of the current tip.",
        ["move_y_pos"],
    )
    assert not _instruction_has_direction_grounding(
        "Rotate the screwdriver clockwise until its tip aligns with the slot.",
        ["rotate_z_neg"],
    )
    assert _instruction_has_direction_grounding(
        "Rotate negatively about base-frame z until the tip aligns with the slot.",
        ["rotate_z_neg"],
    )
    assert _instruction_has_direction_grounding(
        "Move the tip into the hole ahead of it while rotating negatively about base-frame z.",
        ["move_y_pos", "rotate_z_neg"],
    )
    assert not _instruction_has_direction_grounding(
        "Move the screwdriver toward the visible screw hole.", ["move_y_pos"]
    )
    assert not _instruction_has_direction_grounding(
        "Adjust the screwdriver position carefully.", ["move_y_pos"]
    )


def test_natural_base_directions_match_fk_axes() -> None:
    examples = {
        "move_x_pos": "Move the tool forward.",
        "move_x_neg": "Move the tool backward.",
        "move_y_pos": "Shift the tool to the left.",
        "move_y_neg": "Shift the tool to the right.",
        "move_z_pos": "Lift the tool up.",
        "move_z_neg": "Lower the tool down.",
    }
    assert all(
        _instruction_has_natural_direction(instruction, atom_name)
        for atom_name, instruction in examples.items()
    )
    assert not _instruction_has_direction_grounding("Move the tool forward.", ["move_y_pos"])
    assert not _instruction_has_direction_grounding(
        "Move the tool along base-frame +x.", ["move_x_pos"]
    )
    assert not _instruction_has_direction_grounding(
        "Move the tool backward along +x.", ["move_x_neg"]
    )
    assert _instruction_has_direction_grounding(
        "Move backward while rotating positively about the base-frame z axis.",
        ["move_x_neg", "rotate_z_pos"],
    )


def test_annotation_direction_hints_use_natural_translation() -> None:
    move_payload = _fixed_segment([0.9] + [0.1 / 11] * 11).prompt_payload()
    assert move_payload["direction_hints"] == ["move forward"]
    assert "right_fk_gate_reason" in move_payload
    assert "canonical_instructions" not in move_payload

    rotate_payload = _fixed_segment([0.0] * 10 + [1.0, 0.0], labels=(10,)).prompt_payload()
    assert "base-frame z axis" in rotate_payload["direction_hints"][0]


def test_drop_instruction_does_not_require_axis() -> None:
    candidate = CandidateAnnotation(
        global_description="The robot holds the screwdriver near the fixture.",
        segments=[
            CandidateSegment(
                segment_id=0,
                start_s=0.0,
                end_s=3.0,
                low_level_instruction="Hold the screwdriver steady near the screw hole.",
                visual_evidence="The right TCP remains still.",
            )
        ],
    )
    pipeline = AtomicSegmentationPipeline(type("Client", (), {"model": "fake"})())
    fixed = _fixed_segment(
        [1 / 12] * 12,
        mode="drop",
        labels=(),
    )
    locked = pipeline._lock_candidate_to_fk_timeline(candidate, [fixed])
    assert locked.segments[0].low_level_instruction.startswith("Hold")


def _write_lerobot_v3_episode(
    tmp_path: Path, *, include_left: bool = False, include_right: bool = True
) -> Path:
    import json

    import pyarrow as pa
    import pyarrow.parquet as pq

    root = tmp_path / "lerobot"
    video_keys = ["observation.images.base_0_rgb"]
    if include_left:
        video_keys.append("observation.images.left_wrist_0_rgb")
    if include_right:
        video_keys.append("observation.images.right_wrist_0_rgb")
    features = {key: {"dtype": "video", "shape": [24, 32, 3]} for key in video_keys}
    features["observation.state"] = {"dtype": "float32", "shape": [16]}
    info = {
        "codebase_version": "v3.0",
        "chunks_size": 1000,
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": features,
    }
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(json.dumps(info), encoding="utf-8")

    episode_columns = {
        "episode_index": [7],
        "tasks": [["turn the right hand clockwise"]],
        "data/chunk_index": [0],
        "data/file_index": [0],
    }
    for key in video_keys:
        prefix = f"videos/{key}"
        episode_columns[f"{prefix}/chunk_index"] = [0]
        episode_columns[f"{prefix}/file_index"] = [7]
        episode_columns[f"{prefix}/from_timestamp"] = [0.0]
        episode_columns[f"{prefix}/to_timestamp"] = [0.1]
    episode_meta = root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    episode_meta.parent.mkdir(parents=True)
    pq.write_table(pa.table(episode_columns), episode_meta)

    states = np.arange(4 * 16, dtype=np.float32).reshape(4, 16)
    data_path = root / "data" / "chunk-000" / "file-000.parquet"
    data_path.parent.mkdir(parents=True)
    pq.write_table(
        pa.table(
            {
                "observation.state": pa.array(states.tolist(), type=pa.list_(pa.float32(), 16)),
                "action": pa.array((states + 0.5).tolist(), type=pa.list_(pa.float32(), 16)),
                "timestamp": np.asarray([0.0, 0.033, 0.066, 0.099], dtype=np.float32),
                "episode_index": np.asarray([7, 7, 7, 7], dtype=np.int64),
            }
        ),
        data_path,
    )
    for key in video_keys:
        video = root / "videos" / key / "chunk-000" / "file-007.mp4"
        video.parent.mkdir(parents=True)
        video.touch()
    return root


def test_lerobot_episode_uses_base_right_and_right_state(tmp_path: Path) -> None:
    class FakeFK:
        def pose(self, q: np.ndarray) -> CartesianPose:
            return CartesianPose(q[:3], np.eye(3))

    root = _write_lerobot_v3_episode(tmp_path)
    episode = load_lerobot_episode(root, 7)
    assert episode.task == "turn the right hand clockwise"
    assert episode.video_keys == (
        "observation.images.base_0_rgb",
        "observation.images.right_wrist_0_rgb",
    )
    assert episode.right_state.shape == (4, 8)
    np.testing.assert_allclose(episode.right_state[0], np.arange(8, 16))
    np.testing.assert_allclose(episode.right_action[0], np.arange(8, 16) + 0.5)
    assert episode.horizontal_flip == (False, False)
    assert episode.augmentation == "none"
    trace = episode.make_trace(urdf_path="unused.urdf", fk=FakeFK())
    np.testing.assert_allclose(trace.qpos[0], np.arange(8, 15))
    assert trace.robot_state is episode.right_state


def test_tcp_twist_sequence_is_cumulative_and_expressed_in_base_axes() -> None:
    class FakeFK:
        def pose(self, q: np.ndarray) -> CartesianPose:
            angle = float(q[3])
            cosine, sine = np.cos(angle), np.sin(angle)
            rotation = np.asarray(
                [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]]
            )
            return CartesianPose(np.asarray([q[0], q[1], q[2]]), rotation)

    current = np.zeros(7)
    future = np.zeros((2, 7))
    future[0, [0, 3]] = [0.1, 0.2]
    future[1, [0, 3]] = [0.3, 0.4]
    delta = tcp_twist_delta_sequence(current, future, FakeFK())
    assert delta.shape == (2, 6)
    np.testing.assert_allclose(delta[:, 0], [0.1, 0.3], atol=1e-6)
    np.testing.assert_allclose(delta[:, 5], [0.2, 0.4], atol=1e-6)


def test_base_frame_rotation_delta_uses_spatial_not_local_axes() -> None:
    rz = np.asarray([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    rx = np.asarray([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    start = CartesianPose(np.zeros(3), rz)
    end = CartesianPose(np.zeros(3), rx @ rz)
    delta = base_frame_pose_delta(start, end)
    np.testing.assert_allclose(delta[3:], [np.pi / 2, 0.0, 0.0], atol=1e-6)


def test_lerobot_episode_refuses_to_substitute_left_for_missing_right(
    tmp_path: Path,
) -> None:
    root = _write_lerobot_v3_episode(tmp_path, include_right=False)
    try:
        load_lerobot_episode(root, 7)
    except ValueError as error:
        assert "right" in str(error)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("missing right-wrist video should fail")


def test_lerobot_episode_mirrors_left_into_right(tmp_path: Path) -> None:
    class FakeFK:
        def pose(self, q: np.ndarray) -> CartesianPose:
            return CartesianPose(q[:3], np.eye(3))

    root = _write_lerobot_v3_episode(tmp_path, include_left=True, include_right=False)
    episode = load_lerobot_episode(root, 7, mirror_left_to_right=True)
    assert episode.video_keys == (
        "observation.images.base_0_rgb",
        "observation.images.left_wrist_0_rgb",
    )
    np.testing.assert_allclose(episode.right_state[0], np.arange(8) * CR1_LEFT_TO_RIGHT_STATE_SIGN)
    np.testing.assert_allclose(
        episode.right_action[0], (np.arange(8) + 0.5) * CR1_LEFT_TO_RIGHT_STATE_SIGN
    )
    assert episode.horizontal_flip == (True, True)
    assert episode.augmentation == "mirror_left_to_right"
    trace = episode.make_trace(urdf_path="unused.urdf", fk=FakeFK())
    np.testing.assert_allclose(trace.qpos[0], episode.right_state[0, :7])


def test_cr1_left_to_right_mirror_sign_and_frame_flip() -> None:
    left = np.asarray([0.89, 0.28, 0.19, 0.29, -0.0, -1.26, 0.17, 0.8])
    expected = np.asarray([0.89, -0.28, -0.19, 0.29, 0.0, -1.26, -0.17, 0.8])
    np.testing.assert_allclose(mirror_left_arm_values(left), expected)
    bimanual = np.concatenate([left, np.zeros(8)])
    np.testing.assert_allclose(mirror_left_from_bimanual(bimanual), expected)

    frame = np.asarray([[1, 2, 3], [4, 5, 6]], dtype=np.uint8)
    np.testing.assert_array_equal(flip_frame_horizontal(frame), frame[:, ::-1])


def test_materialize_mirrored_episode_bundle(tmp_path: Path) -> None:
    source_videos = []
    for name in ("base_source.mp4", "left_source.mp4"):
        path = tmp_path / name
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (16, 8))
        assert writer.isOpened()
        for _ in range(3):
            frame = np.zeros((8, 16, 3), dtype=np.uint8)
            frame[:, :8, 2] = 240
            frame[:, 8:, 0] = 240
            writer.write(frame)
        writer.release()
        source_videos.append(path)

    state = np.arange(24, dtype=np.float32).reshape(3, 8)
    action = state + 0.5
    bundle = write_mirrored_episode_bundle(
        output_dir=tmp_path / "mirror",
        source_root=tmp_path,
        source_episode_index=7,
        task="move the tool",
        source_base_video=source_videos[0],
        source_left_wrist_video=source_videos[1],
        timestamps=np.asarray([0.0, 0.1, 0.2]),
        right_state=state,
        right_action=action,
    )
    assert bundle.base_video.is_file()
    assert bundle.right_wrist_video.is_file()
    arrays = np.load(bundle.trajectory)
    np.testing.assert_allclose(arrays["right_state"], state)
    np.testing.assert_allclose(arrays["right_action"], action)

    capture = cv2.VideoCapture(str(bundle.base_video))
    ok, mirrored = capture.read()
    capture.release()
    assert ok
    assert mirrored[:, :8, 0].mean() > mirrored[:, :8, 2].mean()
    assert mirrored[:, 8:, 2].mean() > mirrored[:, 8:, 0].mean()


def test_tcp_trace_labels_dominant_positive_x(tmp_path: Path) -> None:
    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_text(
        "\n".join(
            [
                '{"timestamp":0.0,"tcp_pose":[0,0,0,0,0,0]}',
                '{"timestamp":1.0,"tcp_pose":[0.04,0.002,0,0,0,0]}',
            ]
        ),
        encoding="utf-8",
    )
    trace = MotionTrace.load(trace_path)
    label = trace.label_interval(
        0,
        1,
        translation_scale_m=0.02,
        rotation_scale_rad=0.15,
    )
    assert label.skill == AtomicSkill.MOVE_X_POS
    assert label.confidence > 0.8
    assert np.argmax(label.normalized_scores) == 0


def test_tcp_rotation_uses_so3_delta_across_euler_wrap(tmp_path: Path) -> None:
    trace_path = tmp_path / "rotation.jsonl"
    trace_path.write_text(
        "\n".join(
            [
                '{"timestamp":0.0,"tcp_pose":[0,0,0,0,0,2.967059728]}',
                '{"timestamp":1.0,"tcp_pose":[0,0,0,0,0,-2.967059728]}',
            ]
        ),
        encoding="utf-8",
    )
    label = MotionTrace.load(trace_path).label_interval(
        0,
        1,
        translation_scale_m=0.02,
        rotation_scale_rad=0.15,
    )
    assert label.skill == AtomicSkill.ROTATE_Z_POS


def test_fk_prompt_rows_align_one_to_one_with_video_timestamps(tmp_path: Path) -> None:
    trace_path = tmp_path / "aligned.jsonl"
    trace_path.write_text(
        "\n".join(
            [
                '{"timestamp":0.0,"tcp_pose":[0,0,0,0,0,0]}',
                '{"timestamp":1.0,"tcp_pose":[0.2,0,0,0,0,0]}',
            ]
        ),
        encoding="utf-8",
    )
    summary = MotionTrace.load(trace_path).prompt_summary([0.0, 0.5, 1.0])
    rows = summary.splitlines()
    assert len(rows) == 4  # header plus one row per supplied video timestamp
    assert rows[0].startswith("time_s,x,y,z,rotvec_x")
    assert rows[1].startswith("0.000,+0.00000")
    assert rows[2].startswith("0.500,+0.10000")
    assert rows[3].startswith("1.000,+0.20000")
    assert rows[2].endswith(",+0.10000,+0.00000,+0.00000,+0.00000,+0.00000,+0.00000")


def test_joint_hdf5_slices_left_arm_and_normalizes_milliseconds(tmp_path: Path) -> None:
    import h5py

    class FakeFK:
        def __init__(self) -> None:
            self.last_q = None

        def pose(self, q: np.ndarray) -> CartesianPose:
            self.last_q = q.copy()
            return CartesianPose(np.asarray([q[0], q[1], q[2]]), np.eye(3))

    episode_path = tmp_path / "episode.hdf5"
    qpos = np.zeros((3, 16), dtype=np.float64)
    qpos[:, 0] = [0.0, 0.02, 0.04]
    qpos[:, 8] = [1.0, 2.0, 3.0]
    with h5py.File(episode_path, "w") as episode:
        episode.create_dataset("/observations/qpos_120hz", data=qpos)
        episode.create_dataset(
            "/observations/timestamps_120hz", data=np.asarray([1000.0, 1008.0, 1016.0])
        )

    fk = FakeFK()
    trace = JointMotionTrace.load_hdf5(
        episode_path,
        arm="left",
        urdf_path="unused.urdf",
        fk=fk,
    )
    assert trace.timestamps[-1] == 0.016
    label = trace.label_interval(
        0.0,
        0.016,
        translation_scale_m=0.02,
        rotation_scale_rad=0.15,
    )
    assert label.skill == AtomicSkill.MOVE_X_POS
    np.testing.assert_allclose(fk.last_q, qpos[-1, 0:7])


def test_joint_hdf5_uses_camera_clock_as_video_zero(tmp_path: Path) -> None:
    import h5py

    class FakeFK:
        def pose(self, q: np.ndarray) -> CartesianPose:
            return CartesianPose(np.asarray([q[0], 0.0, 0.0]), np.eye(3))

    episode_path = tmp_path / "clocked.hdf5"
    qpos = np.zeros((3, 7), dtype=np.float64)
    with h5py.File(episode_path, "w") as episode:
        episode.create_dataset("/observations/qpos_120hz", data=qpos)
        episode.create_dataset(
            "/observations/timestamps_120hz", data=np.asarray([990.0, 1000.0, 1010.0])
        )
        episode.create_dataset("/observations/timestamps", data=np.asarray([1000.0, 1033.0]))

    trace = JointMotionTrace.load_hdf5(
        episode_path,
        arm="left",
        urdf_path="unused.urdf",
        fk=FakeFK(),
    )
    np.testing.assert_allclose(trace.timestamps, [-0.01, 0.0, 0.01])


def test_joint_hdf5_prefers_regular_30hz_qpos(tmp_path: Path) -> None:
    import h5py

    class FakeFK:
        def pose(self, q: np.ndarray) -> CartesianPose:
            return CartesianPose(np.asarray([q[0], 0.0, 0.0]), np.eye(3))

    episode_path = tmp_path / "regular_30hz.hdf5"
    qpos_30hz = np.zeros((3, 16), dtype=np.float64)
    qpos_30hz[:, 0] = [0.0, 0.1, 0.2]
    qpos_120hz = np.ones((3, 16), dtype=np.float64)
    with h5py.File(episode_path, "w") as episode:
        episode.create_dataset("/observations/qpos", data=qpos_30hz)
        episode.create_dataset(
            "/observations/state_timestamps", data=np.asarray([1000.0, 1033.0, 1066.0])
        )
        episode.create_dataset(
            "/observations/timestamps", data=np.asarray([1000.0, 1033.0, 1066.0])
        )
        episode.create_dataset("/observations/qpos_120hz", data=qpos_120hz)
        episode.create_dataset(
            "/observations/timestamps_120hz", data=np.asarray([1000.0, 1008.0, 1016.0])
        )

    trace = JointMotionTrace.load_hdf5(
        episode_path,
        arm="left",
        urdf_path="unused.urdf",
        fk=FakeFK(),
    )
    np.testing.assert_allclose(trace.qpos[:, 0], [0.0, 0.1, 0.2])
    np.testing.assert_allclose(trace.timestamps, [0.0, 0.033, 0.066])


def test_video_sampler_does_not_seek_past_last_frame(tmp_path: Path) -> None:
    video_path = tmp_path / "tiny.avi"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        10.0,
        (32, 24),
    )
    assert writer.isOpened()
    for index in range(10):
        writer.write(np.full((24, 32, 3), index * 20, dtype=np.uint8))
    writer.release()

    sampled = sample_synchronized_videos(
        [video_path],
        view_names=["base"],
        sample_fps=4,
        max_frames=4,
        tile_width=64,
    )
    assert len(sampled.data_urls) == 4
    assert sampled.timestamps_s[-1] < sampled.duration_s


def test_video_sampler_preserves_strict_requested_rate(tmp_path: Path) -> None:
    video_path = tmp_path / "strict_rate.avi"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        30.0,
        (32, 24),
    )
    assert writer.isOpened()
    for index in range(90):
        writer.write(np.full((24, 32, 3), index, dtype=np.uint8))
    writer.release()

    sampled = sample_synchronized_videos(
        [video_path],
        sample_fps=5,
        max_frames=20,
        tile_width=64,
    )
    np.testing.assert_allclose(np.diff(sampled.timestamps_s), 0.2, atol=1e-6)
    assert sampled.sampled_fps == 5.0


def test_video_sampler_refuses_to_silently_lower_rate(tmp_path: Path) -> None:
    video_path = tmp_path / "capped_rate.avi"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        30.0,
        (32, 24),
    )
    assert writer.isOpened()
    for index in range(90):
        writer.write(np.full((24, 32, 3), index, dtype=np.uint8))
    writer.release()

    try:
        sample_synchronized_videos(
            [video_path],
            sample_fps=5,
            max_frames=10,
            tile_width=64,
        )
    except ValueError as error:
        assert "silently changing" in str(error)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("sampling cap should not change the requested rate")


def test_batch_manifest(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        '{"episode_id":"ep0","task":"move","videos":["base.mp4"]}\n',
        encoding="utf-8",
    )
    entries = load_manifest(manifest)
    assert entries[0].episode_id == "ep0"
    assert entries[0].videos == ["base.mp4"]


def test_pipeline_retries_schema_invalid_json() -> None:
    valid_segment = {
        "segment_id": 0,
        "start_s": 0.0,
        "end_s": 2.0,
        "low_level_instruction": "Move forward.",
        "visual_evidence": "The TCP advances steadily",
        "strong_interaction": False,
    }
    responses = [
        {
            "global_description": "The robot moves through one simple demonstration.",
            "segments": [{**valid_segment, "end_s": 0.0}],
        },
        {
            "global_description": "The robot moves through one simple demonstration.",
            "segments": [valid_segment],
        },
    ]

    class FakeClient:
        model = "qwen3-vl-plus"

        def complete_json(self, messages):
            del messages
            return responses.pop(0), {}

    pipeline = AtomicSegmentationPipeline(FakeClient(), PipelineConfig(review=False))
    fixed = [_fixed_segment([0.9] + [0.1 / 11] * 11, end_s=2.0)]
    annotation, usages = pipeline._request_candidate([], fixed)
    assert annotation.segments[0].end_s == 2.0
    assert len(usages) == 2


def test_pipeline_retries_unsupported_human_control_claim() -> None:
    segment = {
        "segment_id": 0,
        "start_s": 0.0,
        "end_s": 2.0,
        "low_level_instruction": "Move the tool forward.",
        "visual_evidence": "The TCP advances steadily.",
        "strong_interaction": False,
    }
    responses = [
        {
            "global_description": "The robot moves forward. A human operator controls it.",
            "segments": [
                {
                    **segment,
                    "low_level_instruction": (
                        "Move forward while the human operator guides the robot."
                    ),
                }
            ],
        },
        {
            "global_description": "The robot moves the tool forward toward the target.",
            "segments": [segment],
        },
    ]

    class FakeClient:
        model = "qwen3-vl-plus"

        def complete_json(self, messages):
            del messages
            return responses.pop(0), {}

    pipeline = AtomicSegmentationPipeline(FakeClient(), PipelineConfig(review=False))
    fixed = [_fixed_segment([0.9] + [0.1 / 11] * 11, end_s=2.0)]
    annotation, usages = pipeline._request_candidate([], fixed, task="move the tool")
    assert annotation.global_description.startswith("The robot")
    assert len(usages) == 2


def test_pipeline_sanitizes_background_claims_outside_training_instruction() -> None:
    response = {
        "global_description": "The robot moves forward. A person holds a controller nearby.",
        "segments": [
            {
                "segment_id": 0,
                "start_s": 0.0,
                "end_s": 2.0,
                "low_level_instruction": "Move the tool forward.",
                "visual_evidence": (
                    "The TCP advances. A human operator appears in the background."
                ),
                "strong_interaction": False,
            }
        ],
    }

    class FakeClient:
        model = "qwen3-vl-plus"

        def complete_json(self, messages):
            del messages
            return response, {}

    pipeline = AtomicSegmentationPipeline(FakeClient(), PipelineConfig(review=False))
    fixed = [_fixed_segment([0.9] + [0.1 / 11] * 11, end_s=2.0)]
    annotation, usages = pipeline._request_candidate([], fixed, task="move the tool")
    assert annotation.global_description == "The robot moves forward."
    assert annotation.segments[0].visual_evidence == "The TCP advances."
    assert len(usages) == 1


def test_fixed_segment_classification_uses_fk_probabilities() -> None:
    segment = CandidateSegment(
        segment_id=0,
        start_s=0.0,
        end_s=1.0,
        low_level_instruction="Move the tool forward along positive x.",
        visual_evidence="The tool appears to advance.",
        strong_interaction=False,
    )

    class FakeClient:
        model = "qwen3-vl-plus"

    pipeline = AtomicSegmentationPipeline(FakeClient(), PipelineConfig(review=False))
    fixed = _fixed_segment([0.9] + [0.1 / 11] * 11, end_s=1.0)
    targets, source, decision = pipeline._resolve_targets(segment, fixed)
    assert [target.label for target in targets] == [0]
    assert source == "fk"
    assert decision.mode == "single"


def test_fk_timeline_labels_stationary_right_arm_as_stay(tmp_path: Path) -> None:
    trace_path = tmp_path / "stationary.jsonl"
    trace_path.write_text(
        "\n".join(
            [
                '{"timestamp":0.0,"tcp_pose":[0,0,0,0,0,0]}',
                '{"timestamp":3.0,"tcp_pose":[0.0001,0,0,0,0,0]}',
            ]
        ),
        encoding="utf-8",
    )
    pipeline = AtomicSegmentationPipeline(type("Client", (), {"model": "fake"})())
    timeline = build_fk_atomic_timeline(
        MotionTrace.load(trace_path),
        [0.0, 1.0, 2.0],
        3.0,
        translation_scale_m_s=pipeline.config.fk_translation_scale_m,
        rotation_scale_rad_s=pipeline.config.fk_rotation_scale_rad,
        activity_threshold=pipeline.config.fk_activity_threshold,
        minimum_regime_s=0.5,
        gate_config=pipeline.config.gate_config(),
    )
    assert len(timeline) == 1
    assert timeline[0].decision.mode == "single"
    assert timeline[0].decision.labels == (12,)
    assert timeline[0].atomic_probabilities[12] == 1.0
    assert "stationary" in timeline[0].decision.reason


def test_fk_timeline_sets_direction_without_vlm_vote(tmp_path: Path) -> None:
    trace_path = tmp_path / "negative_y.jsonl"
    trace_path.write_text(
        "\n".join(
            [
                '{"timestamp":0.0,"tcp_pose":[0,0,0,0,0,0]}',
                '{"timestamp":3.0,"tcp_pose":[0,-0.05,0,0,0,0]}',
            ]
        ),
        encoding="utf-8",
    )
    pipeline = AtomicSegmentationPipeline(type("Client", (), {"model": "fake"})())
    timeline = build_fk_atomic_timeline(
        MotionTrace.load(trace_path),
        [0.0, 1.0, 2.0],
        3.0,
        translation_scale_m_s=pipeline.config.fk_translation_scale_m,
        rotation_scale_rad_s=pipeline.config.fk_rotation_scale_rad,
        activity_threshold=pipeline.config.fk_activity_threshold,
        minimum_regime_s=0.5,
        gate_config=pipeline.config.gate_config(),
    )
    assert timeline[0].decision.mode == "single"
    assert timeline[0].decision.labels == (3,)
    assert int(np.argmax(timeline[0].atomic_probabilities)) == 3


def test_fk_timeline_records_one_third_second_ratios_inside_fixed_dual() -> None:
    class FakeScoreTrace:
        def atomic_scores_interval(self, start_s, end_s, **kwargs):
            del end_s, kwargs
            scores = np.zeros(12, dtype=np.float64)
            block = int(round(start_s * 3.0))
            scores[0 if block % 2 == 0 else 2] = 1.0
            return scores

    config = PipelineConfig()
    timestamps = [index / 3.0 for index in range(6)]
    timeline = build_fk_atomic_timeline(
        FakeScoreTrace(),
        timestamps,
        2.0,
        translation_scale_m_s=config.fk_translation_scale_m,
        rotation_scale_rad_s=config.fk_rotation_scale_rad,
        activity_threshold=config.fk_activity_threshold,
        minimum_regime_s=2.0,
        gate_config=config.gate_config(),
        top2_temperature=config.fk_top2_temperature,
    )

    assert len(timeline) == 1
    segment = timeline[0]
    assert segment.decision.mode == "dual"
    assert set(segment.decision.labels) == {0, 2}
    assert len(segment.atomic_ratio_blocks) == 6
    assert all(block.valid for block in segment.atomic_ratio_blocks)
    for block in segment.atomic_ratio_blocks:
        assert set(np.flatnonzero(block.weights)) == {0, 2}
        np.testing.assert_allclose(sum(block.weights), 1.0)
    assert segment.atomic_ratio_blocks[0].weights[0] > segment.atomic_ratio_blocks[0].weights[2]
    assert segment.atomic_ratio_blocks[1].weights[0] < segment.atomic_ratio_blocks[1].weights[2]
    assert segment.atomic_ratio_blocks[0].weights[2] > 0.0
    assert segment.atomic_ratio_blocks[1].weights[0] > 0.0


def test_local_dual_ratio_ignores_all_atoms_outside_fixed_pair() -> None:
    scores = np.zeros((2, 12), dtype=np.float64)
    scores[:, [0, 2]] = [0.2, 0.1]
    scores[0, 4] = 0.3
    scores[1, 4] = 30.0
    blocks = _local_ratio_blocks(
        scores,
        np.asarray([0.0, 1 / 3]),
        np.asarray([1 / 3, 2 / 3]),
        segment_start_s=0.0,
        labels=(0, 2),
        top2_temperature=0.10,
    )
    np.testing.assert_allclose(blocks[0].weights, blocks[1].weights)
    assert blocks[0].weights[0] > blocks[0].weights[2] > 0.0
    assert blocks[0].weights[4] == 0.0


def test_fk_probabilities_accumulate_scaled_evidence_not_one_vote_per_frame() -> None:
    scores = np.zeros((3, 12), dtype=np.float64)
    scores[0, 0] = 1.0
    scores[1, 0] = 3.0
    scores[2, 2] = 2.0
    active = scores.max(axis=1) >= 0.15
    probabilities = _scaled_score_probabilities(scores, active)
    np.testing.assert_allclose(probabilities[[0, 2]], [4 / 6, 2 / 6])


def test_fk_top2_evidence_keeps_secondary_mass_without_a_third_atom() -> None:
    scores = np.zeros((2, 12), dtype=np.float64)
    scores[0, [0, 2, 4]] = [0.60, 0.30, 0.10]
    scores[1, [0, 2, 4]] = [0.50, 0.35, 0.15]
    active = scores.max(axis=1) >= 0.15
    gate, reporting = _top2_evidence(scores, active, temperature=0.10)

    # The gate sees only the selected pair and therefore classifies a pair,
    # while reporting preserves the mass that was assigned to the discarded
    # third atom before the final segment normalization.
    np.testing.assert_allclose(gate.sum(axis=1), [1.0, 1.0])
    np.testing.assert_allclose(reporting.sum(axis=1), [0.9, 0.85])
    assert np.all(reporting[:, 4] == 0.0)
    assert np.all(gate[:, 0] > gate[:, 2])


def test_fk_window_preserves_final_gate_separately_from_reporting() -> None:
    scores = np.zeros((5, 12), dtype=np.float64)
    scores[0, [0, 2, 4]] = [0.60, 0.30, 0.10]
    scores[1:, [0, 2, 4]] = [0.45, 0.30, 0.25]
    state = _classify_window(
        scores,
        activity_threshold=0.15,
        gate_config=PipelineConfig().gate_config(),
        top2_temperature=0.10,
    )

    np.testing.assert_allclose(state.gate_probabilities.sum(), 1.0)
    np.testing.assert_allclose(state.probabilities.sum(), 1.0)
    assert not np.allclose(state.gate_probabilities, state.probabilities)


def test_fk_step_identity_uses_top1_while_window_keeps_top2_evidence() -> None:
    scores = np.zeros((1, 12), dtype=np.float64)
    scores[0, [2, 4]] = [0.6, 0.4]

    assert _step_atomic_label(scores[0], activity_threshold=0.15) == 2
    gate, reporting = _top2_evidence(
        scores, np.asarray([True]), temperature=0.10
    )
    assert gate[0, 2] > 0.0
    assert gate[0, 4] > 0.0
    assert reporting[0, 2] > 0.0
    assert reporting[0, 4] > 0.0


def test_fk_opposite_top2_accumulates_but_cannot_form_dual() -> None:
    scores = np.zeros((1, 12), dtype=np.float64)
    scores[0, [0, 1]] = [0.6, 0.4]

    assert _step_atomic_label(scores[0], activity_threshold=0.15) == 0
    gate, reporting = _top2_evidence(
        scores, np.asarray([True]), temperature=0.10
    )
    assert gate[0, 0] > 0.0
    assert gate[0, 1] > 0.0
    assert reporting[0, 0] > 0.0
    assert reporting[0, 1] > 0.0

    decision = classify_atomic_targets(np.asarray([0.6, 0.4] + [0.0] * 10))
    assert decision.mode == "drop"
    assert decision.labels == ()
    assert "opposite_pair=True" in decision.reason


def test_fk_append_gate_recovers_disjoint_step_after_delayed_dual() -> None:
    class FakeScoreTrace:
        def atomic_scores_interval(self, start_s, end_s, **kwargs):
            del end_s, kwargs
            scores = np.zeros(12, dtype=np.float64)
            # The first y+ step is unresolved, but continued base-step probes make
            # the trailing window classify as x+/y+ before x+ fully slides out.
            label = 2 if start_s >= 4.0 else 0
            scores[label] = 1.0
            return scores

    config = PipelineConfig()
    timestamps = [round(index * 0.2, 6) for index in range(40)]
    timeline = build_fk_atomic_timeline(
        FakeScoreTrace(),
        timestamps,
        8.0,
        translation_scale_m_s=config.fk_translation_scale_m,
        rotation_scale_rad_s=config.fk_rotation_scale_rad,
        activity_threshold=config.fk_activity_threshold,
        minimum_regime_s=2.0,
        gate_config=config.gate_config(),
    )
    assert [(segment.start_s, segment.end_s) for segment in timeline] == [(0.0, 8.0)]
    assert timeline[0].decision.mode == "single"
    assert timeline[0].decision.labels == (2,)


def test_fk_greedy_window_discards_short_conflicting_tail() -> None:
    class FakeScoreTrace:
        def atomic_scores_interval(self, start_s, end_s, **kwargs):
            del end_s, kwargs
            scores = np.zeros(12, dtype=np.float64)
            scores[1 if start_s >= 4.2 else 0] = 1.0
            return scores

    config = PipelineConfig()
    timestamps = [round(index * 0.2, 6) for index in range(30)]
    timeline = build_fk_atomic_timeline(
        FakeScoreTrace(),
        timestamps,
        6.0,
        translation_scale_m_s=config.fk_translation_scale_m,
        rotation_scale_rad_s=config.fk_rotation_scale_rad,
        activity_threshold=config.fk_activity_threshold,
        minimum_regime_s=2.0,
        gate_config=config.gate_config(),
    )
    assert [(segment.start_s, segment.end_s) for segment in timeline] == [
        (0.0, 4.2),
        (4.2, 6.0),
    ]
    assert timeline[0].decision.labels == (0,)
    assert timeline[1].decision.mode == "drop"
    assert "no stable 2s" in timeline[1].decision.reason


def test_fk_greedy_window_discards_gap_while_sliding_for_next_stable_seed() -> None:
    class FakeScoreTrace:
        def atomic_scores_interval(self, start_s, end_s, **kwargs):
            del end_s, kwargs
            scores = np.zeros(12, dtype=np.float64)
            if start_s < 2.0:
                label = 0
            elif start_s < 4.0:
                # Ten different atoms make every early two-second search
                # window unstable. The first stable move_y+ seed starts at 3.4 s.
                conflict_labels = [1, 3, 4, 5, 6, 7, 8, 9, 10, 11]
                label = conflict_labels[int(round((start_s - 2.0) / 0.2))]
            else:
                label = 2
            scores[label] = 1.0
            return scores

    config = PipelineConfig()
    timestamps = [round(index * 0.2, 6) for index in range(40)]
    timeline = build_fk_atomic_timeline(
        FakeScoreTrace(),
        timestamps,
        8.0,
        translation_scale_m_s=config.fk_translation_scale_m,
        rotation_scale_rad_s=config.fk_rotation_scale_rad,
        activity_threshold=config.fk_activity_threshold,
        minimum_regime_s=2.0,
        gate_config=config.gate_config(),
    )
    assert [(segment.start_s, segment.end_s) for segment in timeline] == [
        (0.0, 2.0),
        (2.0, 3.4),
        (3.4, 8.0),
    ]
    assert timeline[0].decision.labels == (0,)
    assert timeline[1].decision.mode == "drop"
    assert "discarded conflicting" in timeline[1].decision.reason
    assert timeline[2].decision.labels == (2,)


def test_fk_timeline_separates_stay_from_conflicting_tail() -> None:
    class FakeScoreTrace:
        def atomic_scores_interval(self, start_s, end_s, **kwargs):
            del end_s, kwargs
            scores = np.zeros(12, dtype=np.float64)
            if start_s >= 2.0:
                # A short conflicting tail cannot seed another two-second regime.
                scores[1] = 1.0
            return scores

    config = PipelineConfig()
    timestamps = [round(index * 0.2, 6) for index in range(19)]
    timeline = build_fk_atomic_timeline(
        FakeScoreTrace(),
        timestamps,
        3.8,
        translation_scale_m_s=config.fk_translation_scale_m,
        rotation_scale_rad_s=config.fk_rotation_scale_rad,
        activity_threshold=config.fk_activity_threshold,
        minimum_regime_s=2.0,
        gate_config=config.gate_config(),
    )
    assert len(timeline) == 2
    assert (timeline[0].start_s, timeline[0].end_s) == (0.0, 2.0)
    assert timeline[0].decision.mode == "single"
    assert timeline[0].decision.labels == (12,)
    assert (timeline[1].start_s, timeline[1].end_s) == (2.0, 3.8)
    assert timeline[1].decision.mode == "drop"


def test_fk_two_second_window_can_form_compatible_dual() -> None:
    class FakeScoreTrace:
        def atomic_scores_interval(self, start_s, end_s, **kwargs):
            del end_s, kwargs
            scores = np.zeros(12, dtype=np.float64)
            scores[0 if start_s < 1.0 else 2] = 1.0
            return scores

    config = PipelineConfig()
    timestamps = [round(index * 0.2, 6) for index in range(10)]
    timeline = build_fk_atomic_timeline(
        FakeScoreTrace(),
        timestamps,
        2.0,
        translation_scale_m_s=config.fk_translation_scale_m,
        rotation_scale_rad_s=config.fk_rotation_scale_rad,
        activity_threshold=config.fk_activity_threshold,
        minimum_regime_s=2.0,
        gate_config=config.gate_config(),
    )
    assert len(timeline) == 1
    assert timeline[0].decision.mode == "dual"
    assert set(timeline[0].decision.labels) == {0, 2}


def test_fk_extension_accepts_corresponding_single_and_dual_windows() -> None:
    class FakeScoreTrace:
        def atomic_scores_interval(self, start_s, end_s, **kwargs):
            del end_s, kwargs
            scores = np.zeros(12, dtype=np.float64)
            # The initial two-second seed is dual. The remaining two seconds
            # are single y+, which must still extend the established pair.
            scores[0 if start_s < 1.0 else 2] = 1.0
            return scores

    config = PipelineConfig()
    timestamps = [round(index * 0.2, 6) for index in range(20)]
    timeline = build_fk_atomic_timeline(
        FakeScoreTrace(),
        timestamps,
        4.0,
        translation_scale_m_s=config.fk_translation_scale_m,
        rotation_scale_rad_s=config.fk_rotation_scale_rad,
        activity_threshold=config.fk_activity_threshold,
        minimum_regime_s=2.0,
        gate_config=config.gate_config(),
    )

    assert len(timeline) == 1
    assert (timeline[0].start_s, timeline[0].end_s) == (0.0, 4.0)
    assert timeline[0].decision.mode == "single"
    assert timeline[0].decision.labels == (2,)


def test_fk_extension_rejects_shared_step_containing_previous_opposite() -> None:
    class FakeScoreTrace:
        def atomic_scores_interval(self, start_s, end_s, **kwargs):
            del end_s, kwargs
            scores = np.zeros(12, dtype=np.float64)
            if start_s < 2.0:
                scores[[0, 2]] = 1.0
            else:
                scores[0] = 0.8
                scores[3] = 1.0
            return scores

    config = PipelineConfig()
    timestamps = [round(index * 0.2, 6) for index in range(25)]
    timeline = build_fk_atomic_timeline(
        FakeScoreTrace(),
        timestamps,
        5.0,
        translation_scale_m_s=config.fk_translation_scale_m,
        rotation_scale_rad_s=config.fk_rotation_scale_rad,
        activity_threshold=config.fk_activity_threshold,
        minimum_regime_s=2.0,
        gate_config=config.gate_config(),
    )

    assert [(segment.start_s, segment.end_s) for segment in timeline] == [
        (0.0, 2.0),
        (2.0, 5.0),
    ]
    assert [set(segment.decision.labels) for segment in timeline] == [
        {0, 2},
        {3},
    ]


def test_fk_extension_atomless_step_can_expose_dual_from_trailing_window() -> None:
    class FakeScoreTrace:
        def atomic_scores_interval(self, start_s, end_s, **kwargs):
            del end_s, kwargs
            scores = np.zeros(12, dtype=np.float64)
            if start_s < 1.2:
                scores[0] = 1.0
            elif start_s < 1.8:
                scores[2] = 1.0
            elif start_s < 2.0:
                scores[[0, 2]] = 1.0
            return scores

    config = PipelineConfig()
    timestamps = [round(index * 0.2, 6) for index in range(11)]
    timeline = build_fk_atomic_timeline(
        FakeScoreTrace(),
        timestamps,
        2.2,
        translation_scale_m_s=config.fk_translation_scale_m,
        rotation_scale_rad_s=config.fk_rotation_scale_rad,
        activity_threshold=config.fk_activity_threshold,
        minimum_regime_s=2.0,
        gate_config=config.gate_config(),
    )

    assert len(timeline) == 1
    assert (timeline[0].start_s, timeline[0].end_s) == (0.0, 2.2)
    assert timeline[0].decision.mode == "dual"
    assert set(timeline[0].decision.labels) == {0, 2}


def test_fk_extension_disjoint_step_can_form_connected_dual_with_trailing_window() -> None:
    class FakeScoreTrace:
        def atomic_scores_interval(self, start_s, end_s, **kwargs):
            del end_s, kwargs
            scores = np.zeros(12, dtype=np.float64)
            if start_s < 0.2:
                scores[2] = 1.0
            elif start_s < 0.8:
                scores[0] = 1.0
            elif start_s < 1.0:
                scores[2] = 0.4818
                scores[4] = 0.5182
            elif start_s >= 2.0:
                scores[4] = 1.0
            return scores

    config = PipelineConfig()
    timestamps = [round(index * 0.2, 6) for index in range(11)]
    timeline = build_fk_atomic_timeline(
        FakeScoreTrace(),
        timestamps,
        2.2,
        translation_scale_m_s=config.fk_translation_scale_m,
        rotation_scale_rad_s=config.fk_rotation_scale_rad,
        activity_threshold=config.fk_activity_threshold,
        minimum_regime_s=2.0,
        gate_config=config.gate_config(),
    )

    assert len(timeline) == 1
    assert (timeline[0].start_s, timeline[0].end_s) == (0.0, 2.2)
    assert timeline[0].decision.mode == "dual"
    assert set(timeline[0].decision.labels) == {0, 4}


def test_fk_extension_rejects_atomless_step_after_active_window() -> None:
    class FakeScoreTrace:
        def atomic_scores_interval(self, start_s, end_s, **kwargs):
            del end_s, kwargs
            scores = np.zeros(12, dtype=np.float64)
            if start_s < 2.0:
                scores[0] = 1.0
            return scores

    config = PipelineConfig()
    timestamps = [round(index * 0.2, 6) for index in range(15)]
    timeline = build_fk_atomic_timeline(
        FakeScoreTrace(),
        timestamps,
        3.0,
        translation_scale_m_s=config.fk_translation_scale_m,
        rotation_scale_rad_s=config.fk_rotation_scale_rad,
        activity_threshold=config.fk_activity_threshold,
        minimum_regime_s=2.0,
        gate_config=config.gate_config(),
    )

    assert [(segment.start_s, segment.end_s) for segment in timeline] == [
        (0.0, 2.0),
        (2.0, 3.0),
    ]
    assert timeline[0].decision.labels == (0,)
    assert timeline[1].decision.mode == "drop"


def test_fk_extension_recovers_repeated_disjoint_transitions() -> None:
    class FakeScoreTrace:
        def atomic_scores_interval(self, start_s, end_s, **kwargs):
            del end_s, kwargs
            scores = np.zeros(12, dtype=np.float64)
            if start_s < 2.0:
                label = 0
            elif start_s < 4.0:
                label = 2
            else:
                label = 4
            scores[label] = 1.0
            return scores

    config = PipelineConfig()
    timestamps = [round(index * 0.2, 6) for index in range(35)]
    timeline = build_fk_atomic_timeline(
        FakeScoreTrace(),
        timestamps,
        7.0,
        translation_scale_m_s=config.fk_translation_scale_m,
        rotation_scale_rad_s=config.fk_rotation_scale_rad,
        activity_threshold=config.fk_activity_threshold,
        minimum_regime_s=2.0,
        gate_config=config.gate_config(),
    )

    assert [(segment.start_s, segment.end_s) for segment in timeline] == [(0.0, 7.0)]
    assert timeline[0].decision.mode == "single"
    assert timeline[0].decision.labels == (4,)


def test_fk_extension_cuts_at_first_failure_after_full_unsuccessful_lookahead() -> None:
    class FakeScoreTrace:
        def atomic_scores_interval(self, start_s, end_s, **kwargs):
            del end_s, kwargs
            scores = np.zeros(12, dtype=np.float64)
            if start_s < 2.0:
                label = 0
            else:
                label = 2 + int(round((start_s - 2.0) / 0.2))
            scores[label] = 1.0
            return scores

    config = PipelineConfig()
    timestamps = [round(index * 0.2, 6) for index in range(20)]
    timeline = build_fk_atomic_timeline(
        FakeScoreTrace(),
        timestamps,
        4.0,
        translation_scale_m_s=config.fk_translation_scale_m,
        rotation_scale_rad_s=config.fk_rotation_scale_rad,
        activity_threshold=config.fk_activity_threshold,
        minimum_regime_s=2.0,
        gate_config=config.gate_config(),
    )

    assert [(segment.start_s, segment.end_s) for segment in timeline] == [
        (0.0, 2.0),
        (2.0, 4.0),
    ]
    assert timeline[0].decision.labels == (0,)
    assert timeline[1].decision.mode == "drop"


def test_fk_extension_accepts_completely_different_dual_during_lookahead() -> None:
    class FakeScoreTrace:
        def atomic_scores_interval(self, start_s, end_s, **kwargs):
            del end_s, kwargs
            scores = np.zeros(12, dtype=np.float64)
            if start_s < 2.0:
                label = 0
            else:
                label = 2 if int(round((start_s - 2.0) / 0.2)) % 2 == 0 else 4
            scores[label] = 1.0
            return scores

    config = PipelineConfig()
    timestamps = [round(index * 0.2, 6) for index in range(25)]
    timeline = build_fk_atomic_timeline(
        FakeScoreTrace(),
        timestamps,
        5.0,
        translation_scale_m_s=config.fk_translation_scale_m,
        rotation_scale_rad_s=config.fk_rotation_scale_rad,
        activity_threshold=config.fk_activity_threshold,
        minimum_regime_s=2.0,
        gate_config=config.gate_config(),
    )

    assert len(timeline) == 1
    assert (timeline[0].start_s, timeline[0].end_s) == (0.0, 5.0)
    assert timeline[0].decision.mode == "dual"
    assert set(timeline[0].decision.labels) == {2, 4}


def test_short_real_segment_is_retained_but_not_training_eligible() -> None:
    segment = CandidateSegment(
        segment_id=0,
        start_s=0.0,
        end_s=1.0,
        low_level_instruction="Move forward along positive x.",
        visual_evidence="The TCP advances.",
        strong_interaction=False,
    )

    class FakeClient:
        provider = "qwen"
        model = "qwen3-vl-plus"

    sampled = SampledVideo([], [0.0, 0.5, 0.9], 2.0, 1.0, ["base"])
    pipeline = AtomicSegmentationPipeline(FakeClient(), PipelineConfig(review=False))
    final = pipeline._finalize(
        episode_id="ep0",
        task="move",
        videos=[],
        sampled=sampled,
        annotation=CandidateAnnotation(
            global_description="The robot performs one short forward motion.",
            segments=[segment],
        ),
        fixed_timeline=[_fixed_segment([0.9] + [0.1 / 11] * 11, end_s=1.0)],
        trace=None,
        reviewed=False,
        usage={},
        quantity_value=None,
        quantity_unit=None,
        quantity_scale=None,
    )
    assert len(final.segments) == 1
    assert final.segments[0].training_eligible is False
    assert final.segments[0].gate_mode == "single"
    assert "shorter than" in final.segments[0].gate_reason


def test_strong_interaction_is_auditable_but_excluded_from_stage1() -> None:
    segment = CandidateSegment(
        segment_id=0,
        start_s=0.0,
        end_s=3.0,
        low_level_instruction="Hold the tool against the contacted surface.",
        visual_evidence="The driven tool remains in sustained contact.",
        strong_interaction=True,
    )

    class FakeClient:
        provider = "qwen"
        model = "qwen3-vl-plus"

    sampled = SampledVideo([], [0.0, 1.0, 2.0, 2.8], 1.0, 3.0, ["base"])
    final = AtomicSegmentationPipeline(
        FakeClient(), PipelineConfig(review=False, use_vlm_interaction_gate=True)
    )._finalize(
        episode_id="ep0",
        task="fasten",
        videos=[],
        sampled=sampled,
        annotation=CandidateAnnotation(
            global_description="The robot maintains tool contact while fastening.",
            segments=[segment],
        ),
        fixed_timeline=[_fixed_segment([0.9] + [0.1 / 11] * 11)],
        trace=None,
        reviewed=False,
        usage={},
        quantity_value=None,
        quantity_unit=None,
        quantity_scale=None,
    )
    output = final.segments[0]
    assert output.strong_interaction is True
    assert output.gate_mode == "interaction"
    assert output.training_eligible is False
    assert output.atomic_supervision_mask is False
    assert output.atomic_targets == []


def test_default_pipeline_ignores_vlm_interaction_flag() -> None:
    segment = CandidateSegment(
        segment_id=0,
        start_s=0.0,
        end_s=3.0,
        low_level_instruction="Move the contacted tool forward.",
        visual_evidence="The TCP advances while touching the surface.",
        strong_interaction=True,
    )
    pipeline = AtomicSegmentationPipeline(type("Client", (), {"model": "fake"})(), PipelineConfig())
    fixed = _fixed_segment([0.9] + [0.1 / 11] * 11)
    targets, source, decision = pipeline._resolve_targets(segment, fixed)
    assert [target.label for target in targets] == [0]
    assert source == "fk"
    assert decision.mode == "single"


def test_pipeline_emits_schema2_top2_targets() -> None:
    probabilities = [
        0.43,
        0.02,
        0.37,
        0.02,
        0.05,
        0.02,
        0.02,
        0.02,
        0.02,
        0.01,
        0.01,
        0.03,
    ]
    segment = CandidateSegment(
        segment_id=0,
        start_s=0.0,
        end_s=3.0,
        low_level_instruction="Move forward and left together.",
        visual_evidence="Both base-frame x and y increase throughout.",
        strong_interaction=False,
    )

    class FakeClient:
        provider = "qwen"
        model = "qwen3-vl-plus"

    sampled = SampledVideo([], [0.0, 1.0, 2.0, 2.9], 1.0, 3.0, ["base"])
    final = AtomicSegmentationPipeline(FakeClient(), PipelineConfig(review=False))._finalize(
        episode_id="ep0",
        task="diagonal move",
        videos=[],
        sampled=sampled,
        annotation=CandidateAnnotation(
            global_description="The tool makes one sustained diagonal translation.",
            segments=[segment],
        ),
        fixed_timeline=[_fixed_segment(probabilities, mode="dual", labels=(0, 2))],
        trace=None,
        reviewed=False,
        usage={},
        quantity_value=None,
        quantity_unit=None,
        quantity_scale=None,
    )
    assert final.schema_version == "2.3"
    assert [target.label for target in final.segments[0].atomic_targets] == [0, 2]
    assert final.segments[0].training_eligible is True
    assert final.segments[0].atomic_supervision_mask is True
    np.testing.assert_allclose(
        final.segments[0].atomic_probabilities,
        np.pad(np.asarray(probabilities) / np.sum(probabilities), (0, 1)),
    )
    assert final.segments[0].gate_mode == "dual"
    assert final.segments[0].label_source == "fk"


def test_proposal_prompt_requires_sequential_boundary_scan() -> None:
    prompt = proposal_user_text(
        task="move the object",
        duration_s=5.0,
        sampled_fps=2.0,
        view_names=["base"],
        min_segment_duration_s=2.0,
        fixed_timeline=[_fixed_segment([0.9] + [0.1 / 11] * 11, end_s=5.0)],
    )
    assert "fixed RIGHT-arm intervals" in prompt
    assert "authoritative and immutable" in prompt
    assert '"right_fk_atoms": ["move_x_pos"]' in prompt
    assert '"direction_hints": ["move forward"]' in prompt
    assert "copying each id/start/end exactly" in prompt
    assert "original episode task" in prompt
    assert "`drop` is audit-only" in prompt


def test_system_prompt_orders_global_description_before_segments() -> None:
    prompt = system_prompt(
        "+x forward, +y left, +z up",
        min_segment_duration_s=2.0,
    )
    assert prompt.index("`global_description`, then `segments`") < prompt.index(
        "`segments` is an array"
    )
    assert "Local right-arm forward" in prompt
    assert "Copy every `segment_id`, `start_s`, and `end_s` exactly" in prompt
    assert "exactly one natural, compact" in prompt


def test_cr1_defaults_match_30hz_horizon_candidate_length() -> None:
    config = PipelineConfig()
    assert config.sample_fps == 3.0
    assert config.min_segment_duration_s == 2.0
    assert round(config.min_segment_duration_s * config.sample_fps) == 6
    action_steps = round(config.min_segment_duration_s * 30)
    assert action_steps == 60
    assert action_steps - 50 + 1 == 11
