"""Regression tests for the ordered, read-only PaliGemma atomic-query tail."""

import dataclasses
from types import SimpleNamespace

import pytest

import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
gemma = pytest.importorskip("openpi.models.gemma")
traverse_util = pytest.importorskip("flax.traverse_util")

from atomic_latent_vla.pi05.gemma_adapter import (
    AtomicGemmaModule,
    CoefficientDiTModule,
    DualArmLatentFusion,
    compose_intermediate_arm_latents,
    fuse_intermediate_arm_latents,
)
from atomic_latent_vla.pi05.model import (
    AtomicCompositionTargets,
    AtomicQueries,
    AtomicPi05,
    AtomicTargets,
    dct_prefix,
    huber_angular_distance,
    l2_normalize,
    recover_flow_endpoint,
    visual_rotation_hinge_loss,
)
from atomic_latent_vla.pi05.config import AtomicPi05Config


def _tail_mask(prefix_mask, query_count: int):
    stub = SimpleNamespace(queries=SimpleNamespace(num_queries=query_count))
    return AtomicPi05._query_prefix_mask(stub, prefix_mask, jnp.zeros(prefix_mask.shape[1], dtype=bool))


def test_ordered_query_tail_mask():
    prefix_mask = jnp.asarray([[True, True, False, True]])
    mask = _tail_mask(prefix_mask, query_count=4)[0]
    prefix_len = prefix_mask.shape[1]

    # Prefix cannot be contaminated by Q tokens.
    assert not bool(mask[:prefix_len, prefix_len:].any())
    # Every query sees all valid prefix context and only its same-arm history.
    assert mask[prefix_len + 0].tolist() == [True, True, False, True, True, False, False, False]
    assert mask[prefix_len + 1].tolist() == [True, True, False, True, True, True, False, False]
    assert mask[prefix_len + 2].tolist() == [True, True, False, True, False, False, True, False]
    assert mask[prefix_len + 3].tolist() == [True, True, False, True, False, False, True, True]


def test_subtask_teacher_tokens_are_a_prefix_rooted_causal_branch():
    prefix_mask = jnp.asarray([[True, True]])
    subtask_mask = jnp.asarray([[True, True, False]])
    stub = SimpleNamespace(queries=SimpleNamespace(num_queries=4))
    stub._query_prefix_mask = lambda mask, ar: AtomicPi05._query_prefix_mask(stub, mask, ar)
    mask = AtomicPi05._subtask_prefix_mask(
        stub,
        prefix_mask,
        jnp.zeros(prefix_mask.shape[1], dtype=bool),
        subtask_mask,
    )[0]
    # V0,V1,Q1,Q2,Q3,Q4,T0,T1,Tpad.
    q1, subtask_start = 2, 6
    assert not bool(mask[:subtask_start, subtask_start:].any())
    assert mask[q1].tolist() == [True, True, True, False, False, False, False, False, False]
    # Text reads the clean VLM prefix and its own causal teacher history, but
    # cannot read Q1--Q4.
    assert mask[subtask_start].tolist() == [
        True, True, False, False, False, False, True, False, False,
    ]
    assert mask[subtask_start + 1].tolist() == [
        True, True, False, False, False, False, True, True, False,
    ]
    assert not bool(mask[subtask_start + 2].any())


def test_subtask_ce_is_next_token_loss_rooted_at_last_valid_prefix_hidden():
    class ZeroDecode:
        def __call__(self, hidden, *, method):
            assert method == "decode"
            return jnp.zeros((*hidden.shape[:2], 5), dtype=jnp.float32)

    stub = SimpleNamespace(
        config=SimpleNamespace(subtask_ce_decode_chunk_size=2),
        PaliGemma=SimpleNamespace(llm=ZeroDecode()),
    )
    prefix_hidden = jnp.zeros((2, 3, 4))
    prefix_mask = jnp.asarray([[True, False, True], [True, False, False]])
    teacher_hidden = jnp.zeros((2, 3, 4))
    targets = jnp.asarray([[1, 2, 3], [4, 0, 0]])
    target_mask = jnp.asarray([[True, True, True], [True, False, False]])
    loss = AtomicPi05._subtask_ce_loss(
        stub, prefix_hidden, prefix_mask, teacher_hidden, targets, target_mask
    )
    assert jnp.allclose(loss, jnp.log(5.0))


def test_subtask_ce_is_invariant_to_fully_masked_padding_rows():
    """Static compact CE capacity must not dilute the selected-token loss."""

    class ZeroDecode:
        def __call__(self, hidden, *, method):
            assert method == "decode"
            return jnp.zeros((*hidden.shape[:2], 7), dtype=jnp.float32)

    stub = SimpleNamespace(
        config=SimpleNamespace(subtask_ce_decode_chunk_size=4),
        PaliGemma=SimpleNamespace(llm=ZeroDecode()),
    )
    selected_prefix = jnp.zeros((1, 2, 4))
    selected_prefix_mask = jnp.asarray([[True, True]])
    selected_teacher = jnp.zeros((1, 3, 4))
    selected_targets = jnp.asarray([[1, 2, 3]])
    selected_mask = jnp.asarray([[True, True, True]])
    selected_loss = AtomicPi05._subtask_ce_loss(
        stub,
        selected_prefix,
        selected_prefix_mask,
        selected_teacher,
        selected_targets,
        selected_mask,
    )

    pad_count = 5
    padded_loss = AtomicPi05._subtask_ce_loss(
        stub,
        jnp.concatenate([selected_prefix, jnp.zeros((pad_count, 2, 4))]),
        jnp.concatenate([selected_prefix_mask, jnp.ones((pad_count, 2), dtype=bool)]),
        jnp.concatenate([selected_teacher, jnp.zeros((pad_count, 3, 4))]),
        jnp.concatenate([selected_targets, jnp.zeros((pad_count, 3), dtype=jnp.int32)]),
        jnp.concatenate([selected_mask, jnp.zeros((pad_count, 3), dtype=bool)]),
    )
    assert jnp.allclose(selected_loss, jnp.log(7.0))
    assert jnp.allclose(padded_loss, selected_loss)


def test_dct_prefix_is_a_compact_coefficient_target():
    """Q1 supervises DCT coefficients directly, never an IDCT low-pass path."""

    horizon = 50
    time = jnp.arange(horizon, dtype=jnp.float32)
    # A constant translation plus a frame-to-frame alternating jitter.
    trajectory = (2.0 + 0.5 * (-1.0) ** time)[None, :, None]
    coefficients = dct_prefix(trajectory, coefficients=1)
    assert coefficients.shape == (1, 1, 1)
    # The orthonormal DC coefficient of a constant 2.0 length-50 signal is
    # 2 * sqrt(50); alternating jitter does not leak into the DC coefficient.
    assert jnp.allclose(coefficients[0, 0, 0], 2.0 * jnp.sqrt(horizon), atol=1e-5)


def test_bimanual_coefficient_target_is_one_four_by_twelve_loss_space():
    trajectory = jnp.zeros((3, 50, 12), dtype=jnp.float32)
    coefficients = dct_prefix(trajectory, coefficients=4)
    assert coefficients.shape == (3, 4, 12)


def test_masked_coefficient_loss_cannot_update_the_unlabeled_arm():
    """A right-only atom must not backpropagate through the left zT input."""

    stub = SimpleNamespace(
        config=SimpleNamespace(
            action_horizon=5,
            coefficient_target_dim=12,
            coefficient_count=2,
            arm_count=2,
            coefficient_velocity_warmup_steps=5_000,
            coefficient_wall_transition_steps=2_000,
        )
    )
    stub._fuse_text_arm_latents = lambda z: z.reshape(z.shape[0], -1)
    stub._coefficient_velocity = lambda noisy, time, fused: jnp.broadcast_to(
        fused[:, None, :], noisy.shape
    )
    target = jnp.zeros((1, 5, 12), dtype=jnp.float32)
    mask = jnp.asarray([[True, False]])

    def loss(z):
        return AtomicPi05._coefficient_loss(
            stub,
            jax.random.key(101),
            z,
            target,
            jnp.asarray(0),
            mask,
        ).total

    z = jax.random.normal(jax.random.key(102), (1, 2, 6))
    gradient = jax.grad(loss)(z)
    assert jnp.linalg.norm(gradient[:, 0]) > 0
    assert jnp.array_equal(gradient[:, 1], jnp.zeros_like(gradient[:, 1]))


def test_recovered_coefficient_loss_is_exactly_high_noise_weighted_velocity_loss():
    """Wall-style action-space supervision induces the expected t**2 weight."""

    target = jnp.asarray([[[1.0]], [[1.0]]])
    noise = jnp.asarray([[[3.0]], [[3.0]]])
    time = jnp.asarray([0.25, 0.75])
    noisy = time[:, None, None] * noise + (1.0 - time[:, None, None]) * target
    true_velocity = noise - target
    predicted_velocity = true_velocity + 1.0
    recovered = recover_flow_endpoint(noisy, time, predicted_velocity)
    action_error = jnp.square(recovered - target).reshape(-1)
    assert jnp.allclose(action_error, jnp.square(time), atol=1e-6)


def _atomic_loss_stub(codes):
    num_codes = codes.shape[-2] if codes.ndim == 3 else codes.shape[0]
    config = SimpleNamespace(
        num_atomic_codes=num_codes,
        arm_count=2,
        atomic_temperature=0.07,
        atomic_two_way_positive_temperature=1.0,
        atomic_two_way_negative_temperature=1.0,
        atomic_ratio_temperature=0.07,
        codebook_huber_angle_scale=0.25,
        codebook_huber_delta_deg=1.0,
        codebook_angular_start_step=0,
        latent_dim=codes.shape[-1],
        full_atomic_huber_delta_deg=20.0,
        full_atomic_huber_angle_scale=0.25,
    )
    model = SimpleNamespace(config=config, codebook=SimpleNamespace(value=codes))
    model._atomic_losses = lambda direction, targets, **kwargs: AtomicPi05._atomic_losses(
        model, direction, targets, **kwargs
    )
    return model


def test_bimanual_atomic_losses_route_to_two_distinct_codebooks():
    # The same local label has a deliberately different direction on each arm.
    right_codes = jnp.eye(3, dtype=jnp.float32)
    left_codes = jnp.asarray(
        [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
        dtype=jnp.float32,
    )
    model = _atomic_loss_stub(jnp.stack([right_codes, left_codes]))
    targets = AtomicTargets(
        labels=jnp.asarray([[[0, -1], [0, -1]]], dtype=jnp.int32),
        weights=jnp.asarray([[[1.0, 0.0], [1.0, 0.0]]], dtype=jnp.float32),
        supervision_mask=jnp.asarray([[True, True]]),
    )
    matched = jnp.asarray([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
    swapped = jnp.asarray([[[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]])

    good = AtomicPi05._arm_atomic_losses(
        model, matched, targets, update_codes=True, global_step=jnp.asarray(1)
    )
    bad = AtomicPi05._arm_atomic_losses(
        model, swapped, targets, update_codes=True, global_step=jnp.asarray(1)
    )

    assert good.ranking < bad.ranking
    assert good.codebook < bad.codebook
    assert good.right_perplexity > 0.0
    assert good.left_perplexity > 0.0
    assert jnp.allclose(
        good.perplexity, good.right_perplexity + good.left_perplexity
    )


def test_huber_angular_codebook_pull_does_not_decay_like_cosine():
    """Outside the 1-degree cap, its tangent gradient remains unit magnitude."""

    query = jnp.asarray([1.0, 0.0], dtype=jnp.float32)

    def angular_loss(raw_code):
        code = l2_normalize(raw_code)
        return huber_angular_distance(query, code, delta_rad=jnp.deg2rad(1.0))

    angle = jnp.deg2rad(5.0)
    code = jnp.asarray([jnp.cos(angle), jnp.sin(angle)])
    angular_grad = jnp.linalg.norm(jax.grad(angular_loss)(code))
    cosine_grad = jnp.sin(angle)

    assert jnp.allclose(angular_grad, 1.0, atol=1e-5)
    assert angular_grad > 10.0 * cosine_grad


def test_two_way_atomic_loss_requires_every_dual_positive_to_outrank_negatives():
    codes = jnp.eye(3, dtype=jnp.float32)
    model = _atomic_loss_stub(codes)
    targets = AtomicTargets(
        labels=jnp.asarray([[0, 1]], dtype=jnp.int32),
        weights=jnp.asarray([[0.5, 0.5]], dtype=jnp.float32),
        supervision_mask=jnp.asarray([True]),
    )
    both_positive = jnp.asarray([[1.0, 1.0, -1.0]]) / jnp.sqrt(3.0)
    only_one_positive = jnp.asarray([[1.0, -1.0, -1.0]]) / jnp.sqrt(3.0)
    good = AtomicPi05._atomic_losses(
        model, both_positive, targets, update_codes=True
    )
    bad = AtomicPi05._atomic_losses(
        model, only_one_positive, targets, update_codes=True
    )
    assert good.ranking < bad.ranking
    assert jnp.allclose(good.ratio_kl, 0.0, atol=1e-6)


def test_positive_only_ratio_kl_tracks_dual_strength_without_negative_classes():
    # The fourth direction dimension normalizes q without changing similarities
    # to the three code vectors. Choose the first two logits to reproduce 0.4/0.6.
    codes = jnp.eye(4, dtype=jnp.float32)[:3]
    model = _atomic_loss_stub(codes)
    delta = 0.07 * jnp.log(0.6 / 0.4)
    first, second = 0.30, 0.30 + delta
    remainder = jnp.sqrt(1.0 - first**2 - second**2)
    matched_direction = jnp.asarray([[first, second, 0.0, remainder]])
    reversed_direction = jnp.asarray([[second, first, 0.0, remainder]])
    targets = AtomicTargets(
        labels=jnp.asarray([[0, 1]], dtype=jnp.int32),
        weights=jnp.asarray([[0.4, 0.6]], dtype=jnp.float32),
        supervision_mask=jnp.asarray([True]),
    )
    matched = AtomicPi05._atomic_losses(
        model, matched_direction, targets, update_codes=True
    )
    reversed_ = AtomicPi05._atomic_losses(
        model, reversed_direction, targets, update_codes=True
    )
    assert jnp.allclose(matched.ratio_kl, 0.0, atol=1e-6)
    assert reversed_.ratio_kl > matched.ratio_kl


def test_ratio_kl_is_averaged_over_dual_rows_not_diluted_by_singles():
    codes = jnp.eye(3, dtype=jnp.float32)
    model = _atomic_loss_stub(codes)
    dual_targets = AtomicTargets(
        labels=jnp.asarray([[0, 1]], dtype=jnp.int32),
        weights=jnp.asarray([[0.4, 0.6]], dtype=jnp.float32),
        supervision_mask=jnp.asarray([True]),
    )
    dual_direction = jnp.asarray([[0.6, 0.4, 0.0]]) / jnp.sqrt(0.52)
    dual_only = AtomicPi05._atomic_losses(
        model, dual_direction, dual_targets, update_codes=True
    )
    mixed_targets = AtomicTargets(
        labels=jnp.asarray([[2, -1], [0, 1]], dtype=jnp.int32),
        weights=jnp.asarray([[1.0, 0.0], [0.4, 0.6]], dtype=jnp.float32),
        supervision_mask=jnp.asarray([True, True]),
    )
    mixed = AtomicPi05._atomic_losses(
        model,
        jnp.concatenate([jnp.asarray([[0.0, 0.0, 1.0]]), dual_direction], axis=0),
        mixed_targets,
        update_codes=True,
    )
    assert jnp.allclose(mixed.ratio_kl, dual_only.ratio_kl, atol=1e-6)


def test_atomic_loss_is_finite_and_zero_for_unsupervised_rows():
    codes = jnp.eye(3, dtype=jnp.float32)
    model = _atomic_loss_stub(codes)
    targets = AtomicTargets(
        labels=jnp.asarray([[-1, -1]], dtype=jnp.int32),
        weights=jnp.zeros((1, 2), dtype=jnp.float32),
        supervision_mask=jnp.asarray([False]),
    )
    losses = AtomicPi05._atomic_losses(
        model, jnp.asarray([[1.0, 0.0, 0.0]]), targets, update_codes=True
    )
    assert jnp.isfinite(losses.ranking)
    assert jnp.isfinite(losses.ratio_kl)
    assert losses.ranking == 0.0
    assert losses.ratio_kl == 0.0
    assert losses.codebook == 0.0


def test_frozen_two_way_codebook_is_read_only():
    codes = jnp.eye(3, dtype=jnp.float32)
    model = _atomic_loss_stub(codes)
    targets = AtomicTargets(
        labels=jnp.asarray([[0, -1]], dtype=jnp.int32),
        weights=jnp.asarray([[1.0, 0.0]], dtype=jnp.float32),
        supervision_mask=jnp.asarray([True]),
    )
    direction = jnp.asarray([[0.8, 0.6, 0.0]], dtype=jnp.float32)

    def loss(raw_codes):
        local = _atomic_loss_stub(raw_codes)
        output = AtomicPi05._atomic_losses(
            local, direction, targets, update_codes=False, global_step=jnp.asarray(30_000)
        )
        return output.ranking + output.ratio_kl + output.codebook

    gradient = jax.grad(loss)(codes)
    assert jnp.array_equal(gradient, jnp.zeros_like(gradient))
    output = AtomicPi05._atomic_losses(
        model, direction, targets, update_codes=False, global_step=jnp.asarray(30_000)
    )
    assert output.codebook == 0.0


def test_codebook_projection_tracks_normalized_gate_weighted_code_direction():
    codes = jnp.stack([jnp.eye(4, dtype=jnp.float32)] * 2)
    model = _atomic_loss_stub(codes)
    gate = jnp.asarray(
        [[[0.6, 0.4, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]],
        dtype=jnp.float32,
    )
    mask = jnp.asarray([[True, False]])
    aligned = jnp.asarray(
        [[[0.6, 0.4, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]],
        dtype=jnp.float32,
    )
    wrong = jnp.asarray(
        [[[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0]]],
        dtype=jnp.float32,
    )

    good = AtomicPi05._frozen_codebook_projection_loss(model, aligned, gate, mask)
    bad = AtomicPi05._frozen_codebook_projection_loss(model, wrong, gate, mask)
    assert good < 1e-6
    assert bad > good

    def loss(raw_codes):
        local = _atomic_loss_stub(raw_codes)
        return AtomicPi05._frozen_codebook_projection_loss(
            local, wrong, gate, mask
        )

    assert jnp.array_equal(jax.grad(loss)(codes), jnp.zeros_like(codes))


def test_composition_projection_uses_gate_weights_without_extra_confidence_scale():
    codes = jnp.stack([jnp.eye(4, dtype=jnp.float32)] * 2)
    model = _atomic_loss_stub(codes)
    model._frozen_codebook_projection_loss = (
        lambda directions, weights, mask: AtomicPi05._frozen_codebook_projection_loss(
            model, directions, weights, mask
        )
    )
    directions = jnp.asarray(
        [[[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0]]],
        dtype=jnp.float32,
    )
    targets = AtomicCompositionTargets(
        weights=jnp.asarray(
            [[[0.6, 0.4, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]],
            dtype=jnp.float32,
        ),
        confidence=jnp.asarray([[0.25, 0.0]], dtype=jnp.float32),
        supervision_mask=jnp.asarray([[True, False]]),
    )
    unweighted = AtomicPi05._frozen_codebook_projection_loss(
        model,
        directions,
        targets.weights,
        targets.supervision_mask,
    )
    projected = AtomicPi05._frozen_composition_projection_loss(
        model, directions, targets
    )
    assert jnp.allclose(projected, unweighted)


def test_full_atomic_huber_uses_frozen_weighted_single_and_dual_targets():
    codes = jnp.stack([jnp.eye(3, dtype=jnp.float32)] * 2)
    model = _atomic_loss_stub(codes)
    dual = l2_normalize(jnp.asarray([[0.6, 0.4, 0.0]], dtype=jnp.float32))[0]
    aligned = jnp.asarray([[[1.0, 0.0, 0.0], dual]], dtype=jnp.float32)
    wrong = jnp.asarray([[[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]], dtype=jnp.float32)
    targets = AtomicTargets(
        labels=jnp.asarray([[[0, -1], [0, 1]]], dtype=jnp.int32),
        weights=jnp.asarray([[[1.0, 0.0], [0.6, 0.4]]], dtype=jnp.float32),
        supervision_mask=jnp.asarray([[True, True]]),
    )

    aligned_loss = AtomicPi05._frozen_full_atomic_huber(model, aligned, targets)
    wrong_loss = AtomicPi05._frozen_full_atomic_huber(model, wrong, targets)
    assert aligned_loss < 1e-6
    assert wrong_loss > aligned_loss

    def loss(raw_codes):
        local = _atomic_loss_stub(raw_codes)
        return AtomicPi05._frozen_full_atomic_huber(local, wrong, targets)

    assert jnp.array_equal(jax.grad(loss)(codes), jnp.zeros_like(codes))


def test_zero_full_atomic_huber_transition_is_plain_angular_distance():
    assert AtomicPi05Config().full_atomic_huber_delta_deg == pytest.approx(0.0)
    left = jnp.asarray([[1.0, 0.0, 0.0]], dtype=jnp.float32)
    right = jnp.asarray([[0.0, 1.0, 0.0]], dtype=jnp.float32)
    loss = huber_angular_distance(left, right, delta_rad=0.0)
    assert jnp.allclose(loss, jnp.pi / 2)
    assert jnp.all(jnp.isfinite(jax.grad(
        lambda value: jnp.sum(huber_angular_distance(l2_normalize(value), right, delta_rad=0.0))
    )(left)))


def test_full_atomic_direction_supervises_q1_while_flow_uses_both_arm_pairs():

    config = dataclasses.replace(
        AtomicPi05Config(),
        latent_dim=8,
        active_state_dim=4,
        state_encoder_hidden_dim=8,
    )
    queries = AtomicQueries(prefix_dim=8, config=config, rngs=nnx.Rngs(31))
    hidden = jax.random.normal(jax.random.key(32), (1, 4, 8))
    state = jax.random.normal(jax.random.key(33), (1, 4))

    def atomic_proxy(query_hidden):
        return queries(query_hidden, state)[1][0, 0]

    def flow_proxy(query_hidden):
        return jnp.sum(jnp.square(queries(query_hidden, state)[2]))

    atomic_grad = jax.grad(atomic_proxy)(hidden)
    flow_grad = jax.grad(flow_proxy)(hidden)
    assert jnp.linalg.norm(atomic_grad[:, 0]) > 0
    assert jnp.allclose(atomic_grad[:, 1:], 0.0, atol=1e-8)
    assert jnp.linalg.norm(flow_grad[:, 1]) > 0
    assert jnp.linalg.norm(flow_grad[:, 2]) > 0
    assert jnp.linalg.norm(flow_grad[:, 3]) > 0


def test_atomic_queries_do_not_use_the_dormant_continuous_state_mlp():
    config = dataclasses.replace(
        AtomicPi05Config(),
        latent_dim=8,
        detail_dim=3,
        active_state_dim=4,
        state_encoder_hidden_dim=8,
    )
    queries = AtomicQueries(prefix_dim=8, config=config, rngs=nnx.Rngs(34))
    hidden = jax.random.normal(jax.random.key(35), (2, 4, 8))
    first_state = jnp.zeros((2, 4), dtype=jnp.float32)
    second_state = 100.0 * jnp.ones((2, 4), dtype=jnp.float32)
    first = queries(hidden, first_state)
    second = queries(hidden, second_state)
    for first_value, second_value in zip(first, second, strict=True):
        assert jnp.array_equal(first_value, second_value)


def test_default_two_way_temperature_is_the_non_saturating_zt_value():
    assert AtomicPi05Config().atomic_temperature == pytest.approx(0.10)
    assert AtomicPi05Config().atomic_ratio_temperature == pytest.approx(0.3)
    assert AtomicPi05Config().atomic_ratio_loss_weight == pytest.approx(0.0)
    assert AtomicPi05Config().atomic_composition_temperature == pytest.approx(1.0)
    assert AtomicPi05Config().text_flow_loss_weight == pytest.approx(0.0)


def test_text_stage_shared_flow_bypasses_the_compact_coefficient_dit():
    """ZT uses the stock 50x32 Action Expert when text flow is enabled."""

    batch, horizon, action_dim, latent_dim = 2, 50, 32, 8
    calls = []

    class Queries:
        @staticmethod
        def text_arm_latents(query_hidden, active_state):
            del query_hidden, active_state
            return jnp.ones((batch, 2, latent_dim), dtype=jnp.float32)

        @staticmethod
        def direction(z_text_arms):
            return z_text_arms

    stub = SimpleNamespace(
        config=SimpleNamespace(
            text_flow_loss_weight=1.0,
            coefficient_loss_weight=0.0,
            text_atomic_loss_weight=0.0,
            atomic_composition_loss_weight=0.0,
            fast_action_ce_loss_weight=0.0,
            atomic_ratio_loss_weight=0.0,
            codebook_loss_weight=0.0,
        ),
        queries=Queries(),
    )
    prefix_mask = jnp.ones((batch, 3), dtype=bool)
    stub._prefix_forward = lambda observation: (
        jnp.zeros((batch, 4, latent_dim)),
        prefix_mask,
        object(),
        None,
    )
    stub._controlled_state = lambda state: state
    stub._mask_action_condition = lambda actions: actions

    def suffix_velocity(mask, cache, noisy, time, directions):
        del cache, time
        calls.append((mask.shape, noisy.shape, directions.shape))
        return jnp.zeros_like(noisy)

    stub._suffix_velocity = suffix_velocity
    zero = jnp.zeros((), dtype=jnp.float32)
    stub._arm_atomic_losses = lambda *args, **kwargs: SimpleNamespace(
        ranking=zero,
        ratio_kl=zero,
        codebook=zero,
        perplexity=zero,
        right_perplexity=zero,
        left_perplexity=zero,
    )
    stub._coefficient_loss = lambda *args, **kwargs: pytest.fail(
        "compact coefficient DiT must not execute when its weight is zero"
    )
    observation = SimpleNamespace(
        images={},
        image_masks={},
        state=jnp.zeros((batch, 32), dtype=jnp.float32),
    )
    actions = jnp.zeros((batch, horizon, action_dim), dtype=jnp.float32)
    targets = AtomicTargets(
        labels=jnp.asarray([[[0, -1], [-1, -1]], [[-1, -1], [1, -1]]]),
        weights=jnp.asarray([[[1.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [1.0, 0.0]]]),
        supervision_mask=jnp.asarray([[True, False], [False, True]]),
    )
    output = AtomicPi05.compute_text_stage_loss(
        stub,
        jax.random.key(120),
        observation,
        actions=actions,
        atomic_targets=targets,
        return_output=True,
    )

    assert calls == [((batch, 3), (batch, horizon, action_dim), (batch, 2, latent_dim))]
    assert output.flow_loss > 0
    assert output.coefficient_loss == 0
    assert jnp.allclose(output.total_loss, output.flow_loss)


def test_zm_cosine_alignment_tracks_armwise_stop_gradient_zt_teacher():
    teacher = jnp.asarray(
        [
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
        ],
        dtype=jnp.float32,
    )
    student = jnp.asarray(
        [
            [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        ],
        dtype=jnp.float32,
    )
    mask = jnp.asarray([[True, False], [False, True]])

    loss, similarity = AtomicPi05._text_teacher_cosine_alignment(
        student, teacher, mask
    )
    assert jnp.allclose(loss, 0.5)
    assert jnp.allclose(similarity, 0.5)

    teacher_gradient = jax.grad(
        lambda value: AtomicPi05._text_teacher_cosine_alignment(
            student, value, mask
        )[0]
    )(teacher)
    assert jnp.array_equal(teacher_gradient, jnp.zeros_like(teacher_gradient))

    student_gradient = jax.grad(
        lambda value: AtomicPi05._text_teacher_cosine_alignment(
            value, teacher, mask
        )[0]
    )(student)
    assert jnp.allclose(student_gradient[0, 1], 0.0)
    assert jnp.allclose(student_gradient[1, 0], 0.0)
    assert jnp.linalg.norm(student_gradient[1, 1]) > 0


def test_text_teacher_observation_keeps_selected_prompt_and_state_only():
    state = jnp.arange(64, dtype=jnp.float32).reshape(2, 32)
    prompt = jnp.asarray([[1, 2, 0], [3, 4, 5]], dtype=jnp.int32)
    prompt_mask = prompt != 0
    observation = SimpleNamespace(
        images={"base": jnp.ones((2, 4, 4, 3))},
        image_masks={"base": jnp.ones((2,), dtype=bool)},
        state=state,
        tokenized_prompt=prompt,
        tokenized_prompt_mask=prompt_mask,
    )

    teacher = AtomicPi05._text_only(SimpleNamespace(), observation)
    assert teacher.images == {}
    assert teacher.image_masks == {}
    assert jnp.array_equal(teacher.state, state)
    assert jnp.array_equal(teacher.tokenized_prompt, prompt)
    assert jnp.array_equal(teacher.tokenized_prompt_mask, prompt_mask)


def test_q2_and_q4_are_same_arm_orthogonal_details():
    config = dataclasses.replace(
        AtomicPi05Config(),
        latent_dim=8,
        detail_dim=3,
        active_state_dim=4,
        state_encoder_hidden_dim=8,
    )
    queries = AtomicQueries(prefix_dim=8, config=config, rngs=nnx.Rngs(41))
    hidden = jax.random.normal(jax.random.key(42), (1, 4, 8))
    state = jax.random.normal(jax.random.key(43), (1, 4))
    _, right_direction, latents, left_direction, _ = queries(hidden, state)
    right, left = latents[:, 0], latents[:, 1]

    q2_changed = hidden.at[:, 1].add(0.5)
    _, _, latents_q2, _, _ = queries(q2_changed, state)
    q2_delta = latents_q2[:, 0] - right
    assert jnp.linalg.norm(q2_delta) > 0
    assert jnp.allclose(jnp.sum(q2_delta * right_direction, axis=-1), 0.0, atol=2e-5)
    assert jnp.allclose(latents_q2[:, 1], left, atol=1e-8)

    q3_changed = hidden.at[:, 2].add(0.5)
    _, _, latents_q3, _, _ = queries(q3_changed, state)
    assert jnp.linalg.norm(latents_q3[:, 1] - left) > 0
    assert jnp.allclose(latents_q3[:, 0], right, atol=1e-8)

    q4_changed = hidden.at[:, 3].add(0.5)
    _, _, latents_q4, _, _ = queries(q4_changed, state)
    q4_delta = latents_q4[:, 1] - left
    assert jnp.linalg.norm(q4_delta) > 0
    assert jnp.allclose(jnp.sum(q4_delta * left_direction, axis=-1), 0.0, atol=2e-5)
    assert jnp.allclose(latents_q4[:, 0], right, atol=1e-8)


def test_q2_detail_has_unit_norm_ceiling_with_default_scale():
    config = dataclasses.replace(
        AtomicPi05Config(),
        latent_dim=8,
        detail_dim=4,
        active_state_dim=4,
        state_encoder_hidden_dim=8,
    )
    queries = AtomicQueries(prefix_dim=8, config=config, rngs=nnx.Rngs(44))
    hidden = jax.random.normal(jax.random.key(45), (3, 4, 8))
    state = jax.random.normal(jax.random.key(46), (3, 4))
    _, right_direction, latents, left_direction, _ = queries(hidden, state)
    tangent = jnp.stack(
        [latents[:, 0] - right_direction, latents[:, 1] - left_direction], axis=1
    )
    directions = jnp.stack([right_direction, left_direction], axis=1)

    assert config.max_shift_scale == 0.5
    assert jnp.all(jnp.linalg.norm(tangent, axis=-1) <= 1.0 + 1e-5)
    assert jnp.allclose(jnp.sum(tangent * directions, axis=-1), 0.0, atol=2e-5)


def test_visual_rotation_hinge_has_free_cone_and_normalized_excess():
    base = jnp.asarray([[[1.0, 0.0]]], dtype=jnp.float32)

    def rotated(degrees):
        radians = jnp.deg2rad(degrees)
        return jnp.asarray(
            [[[jnp.cos(radians), jnp.sin(radians)]]], dtype=jnp.float32
        )

    inside_loss, inside_angle = visual_rotation_hinge_loss(
        base,
        rotated(10.0),
        free_angle_rad=jnp.deg2rad(20.0),
        max_angle_rad=jnp.deg2rad(45.0),
    )
    outside_loss, outside_angle = visual_rotation_hinge_loss(
        base,
        rotated(30.0),
        free_angle_rad=jnp.deg2rad(20.0),
        max_angle_rad=jnp.deg2rad(45.0),
    )
    assert inside_loss == pytest.approx(0.0, abs=1e-8)
    assert jnp.rad2deg(inside_angle) == pytest.approx(10.0, abs=1e-4)
    assert outside_loss == pytest.approx((10.0 / 45.0) ** 2, rel=1e-5)
    assert jnp.rad2deg(outside_angle) == pytest.approx(30.0, abs=1e-4)


def test_visual_rotation_hinge_is_finite_at_identity():
    base = jnp.asarray([[[1.0, 0.0]]], dtype=jnp.float32)

    def loss(latent):
        return visual_rotation_hinge_loss(
            base,
            latent,
            free_angle_rad=jnp.deg2rad(20.0),
            max_angle_rad=jnp.deg2rad(45.0),
        )[0]

    assert loss(base) == pytest.approx(0.0, abs=1e-8)
    assert jnp.all(jnp.isfinite(jax.grad(loss)(base)))


def test_visual_rotation_config_validation():
    with pytest.raises(ValueError, match="visual_rotation_loss_weight"):
        dataclasses.replace(AtomicPi05Config(), visual_rotation_loss_weight=-0.1)
    with pytest.raises(ValueError, match="visual_rotation_free_angle_deg"):
        dataclasses.replace(
            AtomicPi05Config(), visual_rotation_free_angle_deg=45.0
        )
    with pytest.raises(ValueError, match="visual_rotation_loss_warmup_steps"):
        dataclasses.replace(
            AtomicPi05Config(), visual_rotation_loss_warmup_steps=-1
        )


def test_force_rotation_config_validation():
    with pytest.raises(ValueError, match="force_rotation_loss_weight"):
        dataclasses.replace(AtomicPi05Config(), force_rotation_loss_weight=-0.1)
    with pytest.raises(ValueError, match="force_rotation_free_angle_deg"):
        dataclasses.replace(
            AtomicPi05Config(),
            force_max_update_angle_deg=45.0,
            force_rotation_free_angle_deg=45.0,
        )


def test_q1_coefficient_dit_is_a_joint_four_token_flow_trunk():
    """The Q1 coefficient model is a transformer, not an independent MLP."""

    config = gemma.get_config("dummy")
    module = CoefficientDiTModule(config=config, embed_dtype="float32")
    tokens = jax.random.normal(jax.random.key(20), (2, 4, config.width))
    positions = jnp.broadcast_to(jnp.arange(4, dtype=jnp.int32)[None], (2, 4))
    condition = jax.random.normal(jax.random.key(21), (2, config.width))
    output, _ = module.init_with_output(jax.random.key(22), tokens, positions, condition)
    assert output.shape == tokens.shape

    # AtomicPi05 hosts this Linen DiT through the same NNX bridge used for the
    # PaliGemma backbone, so validate lazy initialization and a bridged call.
    bridged = nnx_bridge.ToNNX(CoefficientDiTModule(config=config, embed_dtype="float32"))
    bridged.lazy_init(rngs=nnx.Rngs(23), method="init")
    assert bridged(tokens, positions, condition).shape == tokens.shape


def test_dual_arm_fusion_normalizes_after_concatenation_only():
    module = DualArmLatentFusion(
        latent_dim=8,
        arm_mlp_hidden_dim=8,
        fusion_hidden_dim=8,
    )
    arm_latents = jax.random.normal(jax.random.key(27), (2, 2, 8))
    output, variables = module.init_with_output(jax.random.key(28), arm_latents)
    assert output.shape == (2, 8)
    flat = traverse_util.flatten_dict(variables)
    names = {path[-2] for path in flat if len(path) >= 2}
    assert "arm_fusion_norm" in names
    assert not any("shared_arm_norm" in part for path in flat for part in path)


def test_intermediate_composer_reuses_exact_final_query_and_fusion_heads():
    config = dataclasses.replace(
        AtomicPi05Config(),
        latent_dim=8,
        detail_dim=2,
        arm_mlp_hidden_dim=8,
        arm_fusion_hidden_dim=8,
    )
    queries = AtomicQueries(64, config, rngs=nnx.Rngs(101))
    query_hidden = jax.random.normal(jax.random.key(102), (3, 4, 64))
    state = jnp.zeros((3, config.active_state_dim), dtype=jnp.float32)
    _, _, expected_arms, _, _ = queries(query_hidden, state)
    actual_arms = compose_intermediate_arm_latents(
        query_hidden,
        queries.layerwise_composer_params(),
        config.max_shift_scale,
    )
    assert jnp.allclose(actual_arms, expected_arms, atol=2e-5)

    fusion = DualArmLatentFusion(
        latent_dim=8,
        arm_mlp_hidden_dim=8,
        fusion_hidden_dim=8,
    )
    expected_fused, variables = fusion.init_with_output(
        jax.random.key(103), expected_arms
    )
    actual_fused = fuse_intermediate_arm_latents(
        actual_arms, variables["params"]
    )
    assert jnp.allclose(actual_fused, expected_fused, atol=2e-5)


def test_joint_layerwise_queries_condition_each_action_depth():
    prefix_len, query_len, action_len, width, latent_dim = (3, 4, 2, 64, 8)
    config = gemma.get_config("dummy")
    atomic_config = dataclasses.replace(
        AtomicPi05Config(),
        latent_dim=latent_dim,
        detail_dim=2,
        arm_mlp_hidden_dim=8,
        arm_fusion_hidden_dim=8,
    )
    queries = AtomicQueries(width, atomic_config, rngs=nnx.Rngs(110))
    module = AtomicGemmaModule(
        configs=[config, config],
        embed_dtype="float32",
        latent_dim=latent_dim,
        adapter_condition_hidden_dim=8,
        adapter_bottleneck_dim=4,
        arm_mlp_hidden_dim=8,
        arm_fusion_hidden_dim=8,
        adarms=True,
    )
    prefix_queries = jax.random.normal(
        jax.random.key(111), (1, prefix_len + query_len, width)
    )
    actions = jax.random.normal(jax.random.key(112), (1, action_len, width))
    prefix_mask = jnp.ones((1, prefix_len), dtype=bool)
    prefix_query_mask = _tail_mask(prefix_mask, query_len)
    prefix_rows = jnp.concatenate(
        [prefix_query_mask, jnp.zeros((1, prefix_len + query_len, action_len), dtype=bool)],
        axis=-1,
    )
    action_prefix_mask = jnp.concatenate(
        [prefix_mask, jnp.zeros((1, query_len), dtype=bool)], axis=1
    )
    action_rows = jnp.concatenate(
        [
            jnp.broadcast_to(action_prefix_mask[:, None], (1, action_len, prefix_len + query_len)),
            jnp.ones((1, action_len, action_len), dtype=bool),
        ],
        axis=-1,
    )
    full_mask = jnp.concatenate([prefix_rows, action_rows], axis=1)
    positions = jnp.arange(prefix_len + query_len + action_len)[None]
    _, variables = module.init_with_output(
        jax.random.key(113),
        [prefix_queries, actions],
        positions,
        full_mask,
        adarms_cond=[None, jnp.zeros((1, width))],
        latent_condition=jnp.zeros((1, 2, latent_dim)),
    )
    params = variables["params"]
    query_params = queries.layerwise_composer_params()
    kwargs = dict(
        adarms_cond=[None, jnp.zeros((1, width))],
        query_start=prefix_len,
        query_params=query_params,
        fusion_params=params["arm_fusion"],
        query_final_norm_scale=params["final_norm"]["scale"],
        max_shift_scale=atomic_config.max_shift_scale,
        return_layerwise_latents=True,
        return_layerwise_arm_latents=True,
    )
    (outputs, (_, layerwise_latents, layerwise_arm_latents)) = module.apply(
        variables,
        [prefix_queries, actions],
        positions,
        full_mask,
        **kwargs,
    )
    assert layerwise_latents.shape == (config.depth, 1, latent_dim)
    assert layerwise_arm_latents.shape == (config.depth, 1, 2, latent_dim)
    final_arms = compose_intermediate_arm_latents(
        outputs[0][:, prefix_len:], query_params, atomic_config.max_shift_scale
    )
    final_fused = fuse_intermediate_arm_latents(
        final_arms, params["arm_fusion"]
    )
    assert jnp.allclose(layerwise_arm_latents[-1], final_arms, atol=2e-5)
    assert jnp.allclose(layerwise_latents[-1], final_fused, atol=2e-5)

    # Once the zero-init adapter output has trained, changing only Q tokens
    # changes Action hidden although the direct Action attention mask still
    # blocks every Q key. This isolates the new z_M^(l) side path.
    flat = traverse_util.flatten_dict(variables)
    up_kernel = ("params", "layers", "atomic_adapter", "atomic_up", "kernel")
    flat[up_kernel] = jnp.full_like(flat[up_kernel], 0.01)
    trained = traverse_util.unflatten_dict(flat)
    changed_queries = prefix_queries.at[:, prefix_len:].add(3.0)
    (base_out, _) = module.apply(
        trained, [prefix_queries, actions], positions, full_mask, **kwargs
    )
    (changed_out, _) = module.apply(
        trained, [changed_queries, actions], positions, full_mask, **kwargs
    )
    assert float(jnp.max(jnp.abs(base_out[1] - changed_out[1]))) > 0.0


def test_query_tail_is_read_only_and_action_masks_queries():
    """Exercise a 4-layer Gemma: all Q tail tokens enter every layer cache."""

    prefix_len, query_count, action_len, width = 5, 4, 2, 64
    prefix_mask = jnp.ones((1, prefix_len), dtype=bool)
    query_mask = _tail_mask(prefix_mask, query_count)
    combined = jax.random.normal(jax.random.key(1), (1, prefix_len + query_count, width))
    actions = jax.random.normal(jax.random.key(2), (1, action_len, width))
    config = gemma.get_config("dummy")  # depth=4; same head layout in both streams.
    module = AtomicGemmaModule(
        configs=[config, config],
        embed_dtype="float32",
        latent_dim=8,
        adapter_condition_hidden_dim=8,
        adapter_bottleneck_dim=4,
        arm_mlp_hidden_dim=8,
        arm_fusion_hidden_dim=8,
        adarms=True,
    )

    # Initialize prefix, Action Expert and zero-init atomic adapter parameters.
    _, variables = module.init_with_output(
        jax.random.key(3),
        [combined[:, :prefix_len], actions],
        jnp.arange(prefix_len + action_len)[None],
        jnp.ones((1, prefix_len + action_len, prefix_len + action_len), dtype=bool),
        adarms_cond=[None, jnp.zeros((1, width))],
        latent_condition=jnp.zeros((1, 8)),
    )
    (prefix_out, prefix_cache) = module.apply(
        variables,
        [combined, None],
        jnp.arange(prefix_len + query_count)[None],
        query_mask,
        latent_condition=None,
    )

    # Change only future Q2--Q4 inputs. Prefix and Q1 must be invariant.
    future_changed = combined.at[:, prefix_len + 1 :].add(5.0)
    (future_out, future_cache) = module.apply(
        variables,
        [future_changed, None],
        jnp.arange(prefix_len + query_count)[None],
        query_mask,
        latent_condition=None,
    )
    assert float(jnp.max(jnp.abs(prefix_out[0][:, :prefix_len] - future_out[0][:, :prefix_len]))) == 0.0
    assert float(jnp.max(jnp.abs(prefix_out[0][:, prefix_len : prefix_len + 1] - future_out[0][:, prefix_len : prefix_len + 1]))) == 0.0
    assert float(jnp.max(jnp.abs(prefix_out[0][:, prefix_len + 1 : prefix_len + 2] - future_out[0][:, prefix_len + 1 : prefix_len + 2]))) > 0.0

    # The cache contains all P+Q positions at each of the four layers. During
    # action decoding, the last Q positions are explicitly false in its mask.
    assert prefix_cache[0].shape == (config.depth, 1, prefix_len + query_count, 1, config.head_dim)
    action_prefix_mask = jnp.concatenate([prefix_mask, jnp.zeros((1, query_count), dtype=bool)], axis=1)
    action_mask = jnp.concatenate(
        [
            jnp.broadcast_to(action_prefix_mask[:, None], (1, action_len, prefix_len + query_count)),
            jnp.ones((1, action_len, action_len), dtype=bool),
        ],
        axis=-1,
    )
    position = jnp.arange(prefix_len, prefix_len + action_len)[None]
    (action_out, _) = module.apply(
        variables,
        [None, actions],
        position,
        action_mask,
        kv_cache=prefix_cache,
        adarms_cond=[None, jnp.zeros((1, width))],
        latent_condition=jnp.zeros((1, 8)),
    )
    (action_future_out, _) = module.apply(
        variables,
        [None, actions],
        position,
        action_mask,
        kv_cache=future_cache,
        adarms_cond=[None, jnp.zeros((1, width))],
        latent_condition=jnp.zeros((1, 8)),
    )
    assert float(jnp.max(jnp.abs(action_out[1] - action_future_out[1]))) == 0.0

    # RTC reuses this exact Action Expert parameter tree, but gives every
    # action position its own flow-time condition and masks z_M adaptation on
    # the already executed prefix.
    tokenwise_time = jnp.stack(
        [jnp.zeros((width,), dtype=jnp.float32), jnp.ones((width,), dtype=jnp.float32)],
        axis=0,
    )[None]
    (rtc_action_out, _) = module.apply(
        variables,
        [None, actions],
        position,
        action_mask,
        kv_cache=prefix_cache,
        adarms_cond=[None, tokenwise_time],
        latent_condition=jnp.zeros((1, 8)),
        latent_update_mask=jnp.asarray([[False, True]]),
    )
    assert rtc_action_out[1].shape == actions.shape

    # The adapter is zero-initialized, so z_M initially cannot perturb action
    # hidden. Once its final 4->64 projection has learned nonzero weights, two
    # z values produce different Action-Expert hidden outputs.
    z0 = jnp.zeros((1, 8))
    z1 = jnp.arange(8, dtype=jnp.float32)[None]
    (zero_z_out, _) = module.apply(
        variables, [None, actions], position, action_mask, kv_cache=prefix_cache,
        adarms_cond=[None, jnp.zeros((1, width))], latent_condition=z0,
    )
    (one_z_out, _) = module.apply(
        variables, [None, actions], position, action_mask, kv_cache=prefix_cache,
        adarms_cond=[None, jnp.zeros((1, width))], latent_condition=z1,
    )
    assert float(jnp.max(jnp.abs(zero_z_out[1] - one_z_out[1]))) == 0.0

    flat = traverse_util.flatten_dict(variables)
    assert not any("atomic_condition_norm" in part for path in flat for part in path)
    up_kernel = ("params", "layers", "atomic_adapter", "atomic_up", "kernel")
    flat[up_kernel] = jnp.full_like(flat[up_kernel], 0.01)
    trained_adapter_variables = traverse_util.unflatten_dict(flat)
    (trained_zero_z_out, _) = module.apply(
        trained_adapter_variables, [None, actions], position, action_mask, kv_cache=prefix_cache,
        adarms_cond=[None, jnp.zeros((1, width))], latent_condition=z0,
    )
    (trained_one_z_out, _) = module.apply(
        trained_adapter_variables, [None, actions], position, action_mask, kv_cache=prefix_cache,
        adarms_cond=[None, jnp.zeros((1, width))], latent_condition=z1,
    )
    assert float(jnp.max(jnp.abs(trained_zero_z_out[1] - trained_one_z_out[1]))) > 0.0


def test_subtask_tail_cannot_contaminate_atomic_or_action_kv():
    prefix_len, query_len, subtask_len, action_len, width = (3, 4, 3, 2, 64)
    prefix_mask = jnp.ones((1, prefix_len), dtype=bool)
    subtask_mask = jnp.ones((1, subtask_len), dtype=bool)
    stub = SimpleNamespace(queries=SimpleNamespace(num_queries=query_len))
    stub._query_prefix_mask = lambda mask, ar: AtomicPi05._query_prefix_mask(stub, mask, ar)
    hierarchy_mask = AtomicPi05._subtask_prefix_mask(
        stub,
        prefix_mask,
        jnp.zeros(prefix_len, dtype=bool),
        subtask_mask,
    )
    tail_len = query_len + subtask_len
    combined = jax.random.normal(
        jax.random.key(31), (1, prefix_len + tail_len, width)
    )
    actions = jax.random.normal(jax.random.key(32), (1, action_len, width))
    config = gemma.get_config("dummy")
    module = AtomicGemmaModule(
        configs=[config, config],
        embed_dtype="float32",
        latent_dim=8,
        adapter_condition_hidden_dim=8,
        adapter_bottleneck_dim=4,
        arm_mlp_hidden_dim=8,
        arm_fusion_hidden_dim=8,
        adarms=True,
    )
    _, variables = module.init_with_output(
        jax.random.key(33),
        [combined[:, :prefix_len], actions],
        jnp.arange(prefix_len + action_len)[None],
        jnp.ones((1, prefix_len + action_len, prefix_len + action_len), dtype=bool),
        adarms_cond=[None, jnp.zeros((1, width))],
        latent_condition=jnp.zeros((1, 8)),
    )
    (base_out, base_cache) = module.apply(
        variables,
        [combined, None],
        jnp.arange(prefix_len + tail_len)[None],
        hierarchy_mask,
        latent_condition=None,
    )
    text_changed = combined.at[:, -subtask_len:].add(7.0)
    (text_changed_out, text_changed_cache) = module.apply(
        variables,
        [text_changed, None],
        jnp.arange(prefix_len + tail_len)[None],
        hierarchy_mask,
        latent_condition=None,
    )
    # Teacher-forced text is a parallel child of the clean prefix and cannot
    # leak backward into either the prefix or Q1--Q4.
    assert jnp.array_equal(
        base_out[0][:, :prefix_len], text_changed_out[0][:, :prefix_len]
    )
    q1_index = prefix_len
    assert jnp.array_equal(
        base_out[0][:, q1_index : q1_index + query_len],
        text_changed_out[0][:, q1_index : q1_index + query_len],
    )

    # The physical cache contains all tail tokens, but action attention keeps
    # only the clean VLM prefix visible. Hold zM fixed to isolate this path.
    action_prefix_mask = jnp.concatenate(
        [prefix_mask, jnp.zeros((1, tail_len), dtype=bool)], axis=1
    )
    action_mask = jnp.concatenate(
        [
            jnp.broadcast_to(
                action_prefix_mask[:, None], (1, action_len, prefix_len + tail_len)
            ),
            jnp.ones((1, action_len, action_len), dtype=bool),
        ],
        axis=-1,
    )
    action_positions = prefix_len + jnp.arange(action_len)[None]
    (base_action, _) = module.apply(
        variables,
        [None, actions],
        action_positions,
        action_mask,
        kv_cache=base_cache,
        adarms_cond=[None, jnp.zeros((1, width))],
        latent_condition=jnp.zeros((1, 8)),
    )
    (text_changed_action, _) = module.apply(
        variables,
        [None, actions],
        action_positions,
        action_mask,
        kv_cache=text_changed_cache,
        adarms_cond=[None, jnp.zeros((1, width))],
        latent_condition=jnp.zeros((1, 8)),
    )
    assert jnp.allclose(base_action[1], text_changed_action[1], atol=1e-6)


def test_right_only_keeps_full_state_but_masks_left_action_coordinates():
    model = SimpleNamespace(
        config=SimpleNamespace(
            active_state_dim=16,
            action_dim=32,
            controlled_action_start=8,
            controlled_action_dim=8,
        )
    )
    state = jnp.ones((2, 32))
    actions = jnp.ones((2, 50, 32))

    masked_state = AtomicPi05._controlled_state(model, state)
    masked_actions = AtomicPi05._mask_action_condition(model, actions)

    assert jnp.all(masked_state == 1)
    assert jnp.all(masked_actions[..., :8] == 0)
    assert jnp.all(masked_actions[..., 8:16] == 1)
    assert jnp.all(masked_actions[..., 16:] == 0)
    assert jnp.array_equal(
        AtomicPi05._controlled_actions(model, actions), actions[..., 8:16]
    )


def test_bimanual_flow_keeps_all_32_pi05_coordinates():
    model = SimpleNamespace(
        config=SimpleNamespace(
            active_state_dim=16,
            action_dim=32,
            active_action_dim=16,
            controlled_action_start=0,
            controlled_action_dim=16,
        )
    )
    actions = jnp.arange(32, dtype=jnp.float32)[None, None, :]

    assert jnp.array_equal(AtomicPi05._mask_action_condition(model, actions), actions)
