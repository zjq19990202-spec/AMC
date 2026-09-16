"""Shape, alignment and zero-init tests for the optional force second stage."""

import dataclasses

import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
nnx = pytest.importorskip("flax.nnx")
pytest.importorskip("openpi.models.gemma")

from atomic_latent_vla.pi05.config import AtomicPi05Config  # noqa: E402
from atomic_latent_vla.pi05.force import (  # noqa: E402
    ForceConditioner,
    canonicalize_force_state_views,
    force_token_lag_positions,
    future_force_forecast_metrics,
    future_force_delta_target,
    sincos_embedding,
    temporal_masked_mean,
)
from atomic_latent_vla.pi05.model import (  # noqa: E402
    rtc_clamp_prefix,
    rtc_flow_batch,
    rtc_flow_loss,
)


def _small_force_config() -> AtomicPi05Config:
    return dataclasses.replace(
        AtomicPi05Config(),
        enable_force_stage=True,
        action_horizon=5,
        force_history_samples=12,
        force_fast_history_samples=4,
        force_history_train_lengths=(4, 8, 12),
        force_future_samples=20,
        force_update_action_steps=1,
        force_encoder_width=32,
        force_encoder_depth=2,
        force_encoder_num_heads=4,
        force_encoder_mlp_dim=64,
        force_latent_dim=16,
        force_future_decoder_depth=1,
    )


def test_force_default_architecture_matches_ablation_winner():
    config = AtomicPi05Config()
    assert config.force_encoder_depth == 2
    assert config.force_future_decoder_stride == 4
    assert config.force_position_base == 10_000.0
    assert config.force_history_train_lengths == (120,)


def test_force_clock_contract_is_exactly_four_to_one():
    config = _small_force_config()
    assert config.force_sample_rate_hz == 4 * config.force_action_rate_hz
    assert config.force_future_samples // config.force_temporal_stride == config.action_horizon
    with pytest.raises(ValueError, match="force rate"):
        dataclasses.replace(config, force_sample_rate_hz=100)
    with pytest.raises(ValueError, match="fast_history"):
        dataclasses.replace(config, force_fast_history_samples=8)


def test_fast_token_positions_are_suffix_of_slow_positions():
    slow = force_token_lag_positions(30)
    fast = force_token_lag_positions(10)
    assert jnp.allclose(slow[-10:], fast)
    assert float(slow[0]) == -29.0
    assert float(fast[0]) == -9.0
    assert float(slow[-1]) == 0.0


def test_force_position_embedding_uses_standard_10000_base():
    embedded = sincos_embedding(jnp.asarray([1.0]), 4)
    frequencies = jnp.asarray([1.0, 1.0 / 100.0])
    expected = jnp.concatenate([jnp.sin(frequencies), jnp.cos(frequencies)])[None]
    assert jnp.allclose(embedded, expected)


def test_force_position_embedding_supports_short_horizon_base():
    embedded = sincos_embedding(jnp.asarray([1.0]), 4, base=100.0)
    frequencies = jnp.asarray([1.0, 1.0 / 10.0])
    expected = jnp.concatenate([jnp.sin(frequencies), jnp.cos(frequencies)])[None]
    assert jnp.allclose(embedded, expected)


def test_hidden_force_tokens_preserve_explicit_rtc_phase():
    config = dataclasses.replace(
        _small_force_config(),
        enable_force_hidden_cross_attention=True,
        force_hidden_cross_attention_heads=2,
    )
    module = ForceConditioner(
        prefix_dim=64,
        action_dim=32,
        latent_dim=config.latent_dim,
        config=config,
        rngs=nnx.Rngs(23),
    )
    batch = 2
    force_latent = jnp.ones((batch, 2, config.force_latent_dim), dtype=jnp.float32)
    recent_tokens = jnp.ones(
        (
            batch,
            2,
            config.force_fast_history_samples // config.force_temporal_stride,
            config.force_encoder_width,
        ),
        dtype=jnp.float32,
    )
    history_mask = jnp.ones(
        (batch, 2, config.force_fast_history_samples), dtype=jnp.bool_
    )

    offset0 = module.compose_force_tokens(
        force_latent,
        recent_tokens,
        history_mask,
        jnp.asarray([0, 0], dtype=jnp.int32),
    )
    offset4 = module.compose_force_tokens(
        force_latent,
        recent_tokens,
        history_mask,
        jnp.asarray([4, 4], dtype=jnp.int32),
    )

    assert offset0.shape == (batch, 2, config.force_encoder_width)
    assert not jnp.allclose(offset0, offset4)
    # RTC phase is shared across arms; arm-specific content remains the only
    # source of a left/right difference.
    assert jnp.allclose(offset0[:, 0], offset0[:, 1])


def test_future_force_metrics_measure_derivative_and_peak_timing():
    target = jnp.zeros((1, 2, 8, 1), dtype=jnp.float32)
    target = target.at[:, :, 4, 0].set(2.0)
    prediction = jnp.zeros_like(target).at[:, :, 5, 0].set(1.5)
    mask = jnp.ones(target.shape[:-1], dtype=jnp.bool_)
    metrics = future_force_forecast_metrics(
        prediction,
        target,
        mask,
        coarse_stride=4,
        sample_rate_hz=120.0,
    )
    assert float(metrics["raw_rmse"]) > 0.0
    assert float(metrics["derivative_rmse_per_s"]) > 0.0
    assert jnp.isclose(metrics["peak_amplitude_mae"], 0.5)
    assert jnp.isclose(metrics["peak_timing_mae_samples"], 1.0)
    assert jnp.isclose(metrics["peak_timing_mae_ms"], 1000.0 / 120.0)


def test_shared_state_is_canonicalized_local_arm_first():
    left = jnp.arange(8, dtype=jnp.float32)
    right = 100.0 + jnp.arange(8, dtype=jnp.float32)
    shared = jnp.concatenate([left, right])[None, None]

    views = canonicalize_force_state_views(shared)

    assert views.shape == (1, 2, 1, 16)
    # Atomic arm order is [right,left], and every encoder view is
    # [local_arm_state, opposite_arm_state].
    assert jnp.array_equal(views[0, 0, 0], jnp.concatenate([right, left]))
    assert jnp.array_equal(views[0, 1, 0], jnp.concatenate([left, right]))


def test_future_force_target_retains_raw_120hz_delta_and_coarse_view():
    history = jnp.ones((2, 2, 12, 6), dtype=jnp.float32)
    future = 3.0 * jnp.ones((2, 2, 20, 6), dtype=jnp.float32)
    mask = jnp.ones((2, 2, 20), dtype=jnp.bool_)
    target, raw_mask = future_force_delta_target(history, future, mask)
    assert target.shape == (2, 2, 20, 6)
    assert raw_mask.shape == (2, 2, 20)
    coarse, coarse_mask = temporal_masked_mean(target, raw_mask, stride=4)
    assert coarse.shape == (2, 2, 5, 6)
    assert coarse_mask.shape == (2, 2, 5)
    assert jnp.allclose(target, 2.0)
    assert jnp.allclose(coarse, 2.0)
    assert bool(jnp.all(coarse_mask))


def test_force_conditioner_shapes_and_zero_initial_delta():
    config = _small_force_config()
    module = ForceConditioner(
        prefix_dim=64,
        action_dim=32,
        latent_dim=config.latent_dim,
        config=config,
        rngs=nnx.Rngs(0),
    )
    batch = 2
    prefix = jnp.ones((batch, 7, 64), dtype=jnp.float32)
    prefix_mask = jnp.ones((batch, 7), dtype=jnp.bool_)
    z_model = jnp.ones((batch, 2, config.latent_dim), dtype=jnp.float32)
    force = jnp.ones((batch, 2, config.force_history_samples, config.force_dim), dtype=jnp.float32)
    state = jnp.ones(
        (batch, config.force_history_samples, config.force_state_dim), dtype=jnp.float32
    )
    history_mask = jnp.ones((batch, 2, config.force_history_samples), dtype=jnp.bool_)

    context = module.encode_context(prefix, prefix_mask, z_model, force, state, history_mask)
    assert context.latent.shape == (batch, 2, config.force_latent_dim)
    assert context.history_tokens.shape == (batch, 2, 3, config.force_encoder_width)
    prediction = module.predict_future_force_delta(
        context.latent,
    )
    assert prediction.shape == (batch, 2, config.force_future_samples, config.force_dim)
    assert hasattr(module, "force_query")
    assert hasattr(module, "future_gru_cells")
    assert not hasattr(module, "future_blocks")
    assert hasattr(module, "slow_projection")
    assert hasattr(module, "fast_projection")
    assert module.slow_projection is not module.fast_projection
    # z_M is the fast cross-attention query by itself; slow force enters K/V.
    assert module.fast_query.kernel.value.shape == (
        config.latent_dim,
        config.force_encoder_width,
    )
    assert module.slow_memory_from_latent.kernel.value.shape == (
        config.force_latent_dim,
        config.force_encoder_width,
    )
    assert module.force_scale_embedding.value.shape == (
        2,
        config.force_encoder_width,
    )
    assert jnp.array_equal(module.layer_gates(), jnp.ones((18,)))

    modulation = module.modulate(
        z_model,
        context.latent,
        force,
        state,
        history_mask,
        jnp.asarray([0, 4]),
    )
    assert modulation.delta_z.shape == z_model.shape
    assert jnp.array_equal(modulation.delta_z, jnp.zeros_like(modulation.delta_z))
    assert jnp.array_equal(modulation.z_exec, z_model)


def test_action_aligned_gru_decodes_four_raw_force_samples_per_step():
    config = dataclasses.replace(
        _small_force_config(),
        force_future_decoder_stride=4,
        force_future_decoder_kind="linear_chunk",
    )
    module = ForceConditioner(
        prefix_dim=64,
        action_dim=32,
        latent_dim=config.latent_dim,
        config=config,
        rngs=nnx.Rngs(17),
    )
    latent = jnp.ones((2, 2, config.force_latent_dim), dtype=jnp.float32)
    prediction = module.predict_future_force_delta(latent)
    assert prediction.shape == (2, 2, config.force_future_samples, config.force_dim)
    assert module.future_out.kernel.value.shape[-1] == (
        config.arm_count * config.force_future_decoder_stride * config.force_dim
    )
    gradient = jax.grad(lambda value: jnp.mean(module.predict_future_force_delta(value)))(latent)
    assert bool(jnp.all(jnp.isfinite(gradient)))
    assert float(jnp.linalg.norm(gradient)) > 0.0


def test_phase_conditioned_decoder_shares_head_across_four_subsamples():
    config = dataclasses.replace(
        _small_force_config(),
        force_future_decoder_stride=4,
        force_future_decoder_kind="phase_mlp",
    )
    module = ForceConditioner(
        prefix_dim=64,
        action_dim=32,
        latent_dim=config.latent_dim,
        config=config,
        rngs=nnx.Rngs(19),
    )
    latent = jnp.ones((2, 2, config.force_latent_dim), dtype=jnp.float32)
    prediction = module.predict_future_force_delta(latent)
    assert prediction.shape == (2, 2, config.force_future_samples, config.force_dim)
    assert module.future_phase_embedding.value.shape == (
        config.force_future_decoder_stride,
        config.force_encoder_width,
    )
    assert module.future_phase_out.kernel.value.shape[-1] == (
        config.arm_count * config.force_dim
    )
    gradient = jax.grad(lambda value: jnp.mean(module.predict_future_force_delta(value)))(latent)
    assert bool(jnp.all(jnp.isfinite(gradient)))
    assert float(jnp.linalg.norm(gradient)) > 0.0


def test_force_decoder_kind_is_validated():
    with pytest.raises(ValueError, match="force_future_decoder_kind"):
        dataclasses.replace(AtomicPi05Config(), force_future_decoder_kind="unknown")


def test_b1_freeze_filter_excludes_every_fast_only_parameter():
    config = _small_force_config()

    class Wrapper(nnx.Module):
        def __init__(self):
            self.force_conditioner = ForceConditioner(
                64, 32, config.latent_dim, config, rngs=nnx.Rngs(11)
            )

    trainable = nnx.state(Wrapper()).filter(
        nnx.All(
            nnx.Param,
            nnx.Not(config.get_force_freeze_filter(train_atomic_adapters=False)),
        )
    )
    paths = ["/".join(map(str, path)) for path in trainable.flat_state()]
    assert any("history_blocks" in path for path in paths)
    assert any("slow_projection" in path for path in paths)
    forbidden = (
        "fast_projection",
        "fast_query",
        "fast_attention",
        "fast_hidden",
        "force_scale_embedding",
        "slow_memory_from_latent",
        "delta_out",
        "layer_gate_logits",
    )
    assert not [path for path in paths if any(name in path for name in forbidden)]
    frozen = nnx.state(Wrapper()).filter(config.get_force_freeze_filter(train_atomic_adapters=False))
    assert not [
        path
        for path, value in frozen.flat_state().items()
        if isinstance(value, nnx.Param) and value.value is None
    ]


def test_force_action_only_filter_trains_force_path_but_not_future_decoder():
    config = _small_force_config()

    class Wrapper(nnx.Module):
        def __init__(self):
            self.force_conditioner = ForceConditioner(
                64, 32, config.latent_dim, config, rngs=nnx.Rngs(17)
            )
            self.action_in_proj = nnx.Linear(4, 32, rngs=nnx.Rngs(18))

    trainable = nnx.state(Wrapper()).filter(
        nnx.All(
            nnx.Param,
            nnx.Not(config.get_force_action_only_freeze_filter()),
        )
    )
    paths = ["/".join(map(str, path)) for path in trainable.flat_state()]
    assert any("history_blocks" in path for path in paths)
    assert any("fast_attention" in path for path in paths)
    assert any("delta_out" in path for path in paths)
    assert not any("rtc_commit_embedding" in path for path in paths)
    assert not any("future_" in path for path in paths)
    assert not any("action_in_proj" in path for path in paths)
    frozen = nnx.state(Wrapper()).filter(config.get_force_action_only_freeze_filter())
    assert not [
        path
        for path, value in frozen.flat_state().items()
        if isinstance(value, nnx.Param) and value.value is None
    ]


def test_full_token_adapter_filter_freezes_b1_and_action_path():
    config = dataclasses.replace(
        _small_force_config(),
        force_full_token_adapter=True,
        force_full_token_adapter_heads=2,
    )

    class Wrapper(nnx.Module):
        def __init__(self):
            self.force_conditioner = ForceConditioner(
                64, 32, config.latent_dim, config, rngs=nnx.Rngs(70)
            )
            self.action_in_proj = nnx.Linear(4, 32, rngs=nnx.Rngs(71))

    trainable = nnx.state(Wrapper()).filter(
        nnx.All(
            nnx.Param,
            nnx.Not(config.get_force_full_token_adapter_freeze_filter()),
        )
    )
    paths = ["/".join(map(str, path)) for path in trainable.flat_state()]
    assert any("fast_projection" in path for path in paths)
    assert any("full_token_adapter_attention" in path for path in paths)
    assert any("full_token_adapter_out" in path for path in paths)
    assert any("layer_gate_logits" in path for path in paths)
    assert not any("history_blocks" in path for path in paths)
    assert not any("slow_projection" in path for path in paths)
    assert not any("future_" in path for path in paths)
    assert not any("action_in_proj" in path for path in paths)


def test_full_token_adapter_reads_projected_slow_and_fast_tokens_per_arm():
    config = dataclasses.replace(
        _small_force_config(),
        force_full_token_adapter=True,
        force_full_token_adapter_heads=2,
    )
    module = ForceConditioner(64, 32, config.latent_dim, config, rngs=nnx.Rngs(72))
    batch = 1
    z_model = jnp.ones((batch, 2, config.latent_dim), dtype=jnp.float32)
    force = jnp.ones((batch, 2, config.force_history_samples, config.force_dim))
    state = jnp.ones((batch, config.force_history_samples, config.force_state_dim))
    mask = jnp.ones((batch, 2, config.force_history_samples), dtype=jnp.bool_)
    prefix = jnp.ones((batch, 7, 64), dtype=jnp.float32)
    prefix_mask = jnp.ones((batch, 7), dtype=jnp.bool_)
    context = module.encode_context(prefix, prefix_mask, z_model, force, state, mask)

    modulation = module.modulate(
        z_model,
        context.latent,
        force,
        state,
        mask,
        jnp.zeros((batch,), dtype=jnp.int32),
        slow_history_tokens=context.history_tokens,
        slow_history_token_mask=context.history_token_mask,
    )
    assert module.full_token_adapter_attention.num_heads == 2
    assert context.history_tokens.shape[2] == (
        config.force_history_samples // config.force_temporal_stride
    )
    assert modulation.recent_tokens.shape[2] == (
        config.force_fast_history_samples // config.force_temporal_stride
    )
    assert jnp.array_equal(modulation.delta_z, jnp.zeros_like(modulation.delta_z))

    module.full_token_adapter_out.kernel.value = 0.01 * jax.random.normal(
        jax.random.key(73), module.full_token_adapter_out.kernel.value.shape
    )
    opened = module.modulate(
        z_model,
        context.latent,
        force,
        state,
        mask,
        jnp.zeros((batch,), dtype=jnp.int32),
        slow_history_tokens=context.history_tokens,
        slow_history_token_mask=context.history_token_mask,
    )
    changed = module.modulate(
        z_model,
        context.latent,
        force,
        state,
        mask,
        jnp.zeros((batch,), dtype=jnp.int32),
        slow_history_tokens=context.history_tokens.at[:, 0, 0].add(10.0),
        slow_history_token_mask=context.history_token_mask,
    )
    assert not jnp.allclose(opened.delta_z, changed.delta_z)


def test_full_b2_filter_freezes_future_decoder_when_objective_is_disabled():
    config = _small_force_config()

    class Wrapper(nnx.Module):
        def __init__(self):
            self.force_conditioner = ForceConditioner(
                64, 32, config.latent_dim, config, rngs=nnx.Rngs(20)
            )
            self.atomic_adapter = nnx.Linear(32, 32, rngs=nnx.Rngs(21))
            self.action_in_proj = nnx.Linear(4, 32, rngs=nnx.Rngs(22))

    freeze = config.get_force_freeze_filter(
        train_atomic_adapters=True,
        train_future_decoder=False,
    )
    trainable = nnx.state(Wrapper()).filter(nnx.All(nnx.Param, nnx.Not(freeze)))
    paths = ["/".join(map(str, path)) for path in trainable.flat_state()]
    assert any("history_blocks" in path for path in paths)
    assert any("fast_attention" in path for path in paths)
    assert any("atomic_adapter" in path for path in paths)
    assert any("action_in_proj" in path for path in paths)
    assert not any("future_" in path for path in paths)
    assert not any("rtc_commit_embedding" in path for path in paths)


def test_force_gru_uses_stable_explicit_recurrent_bias():
    config = _small_force_config()
    module = ForceConditioner(64, 32, config.latent_dim, config, rngs=nnx.Rngs(19))
    cell = module.future_gru_cells["layer_0"]
    assert cell.dense_h.use_bias
    assert cell.dense_h.bias.value.shape == (3 * config.force_encoder_width,)
    assert jnp.array_equal(
        cell.dense_h.bias.value,
        jnp.zeros_like(cell.dense_h.bias.value),
    )
    assert not [
        path
        for path, value in nnx.state(module).flat_state().items()
        if isinstance(value, nnx.Param) and value.value is None
    ]
    assert not [
        path for path in nnx.state(module).flat_state() if "rngs" in "/".join(map(str, path))
    ]


def test_force_conditioner_graphdef_is_stable_across_constructions():
    config = _small_force_config()
    first = ForceConditioner(64, 32, config.latent_dim, config, rngs=nnx.Rngs(20))
    second = ForceConditioner(64, 32, config.latent_dim, config, rngs=nnx.Rngs(21))
    # JAX evaluates the initialization once for shape inference and constructs
    # it again for the real sharded state. Static callable metadata must be
    # identical across those two constructions.
    assert nnx.graphdef(first) == nnx.graphdef(second)


def test_empty_fast_window_uses_only_slow_latent_for_initial_residual():
    config = _small_force_config()
    module = ForceConditioner(64, 32, config.latent_dim, config, rngs=nnx.Rngs(1))
    # Open the intentionally zero-initialized output gate only for this test.
    module.delta_out.kernel.value = 0.01 * jax.random.normal(
        jax.random.key(2), module.delta_out.kernel.value.shape
    )
    batch = 1
    z_model = jnp.ones((batch, 2, config.latent_dim), dtype=jnp.float32)
    force = jnp.zeros((batch, 2, config.force_history_samples, config.force_dim))
    state = jnp.zeros((batch, config.force_history_samples, config.force_state_dim))
    mask = jnp.zeros((batch, 2, config.force_history_samples), dtype=jnp.bool_)
    offset = jnp.zeros((batch,), dtype=jnp.int32)

    zero_slow = module.modulate(
        z_model,
        jnp.zeros((batch, 2, config.force_latent_dim)),
        force,
        state,
        mask,
        offset,
    )
    slow_only = module.modulate(
        z_model,
        jnp.ones((batch, 2, config.force_latent_dim)),
        force,
        state,
        mask,
        offset,
    )
    masked_values_changed = module.modulate(
        z_model,
        jnp.ones((batch, 2, config.force_latent_dim)),
        force + 10_000.0,
        state + 10_000.0,
        mask,
        offset,
    )
    assert not jnp.allclose(zero_slow.delta_z, slow_only.delta_z)
    assert jnp.allclose(slow_only.delta_z, masked_values_changed.delta_z)
    assert bool(jnp.all(jnp.isfinite(slow_only.z_exec)))


def test_slow_and_fast_heads_share_encoder_but_specialize_tokens():
    config = _small_force_config()
    module = ForceConditioner(64, 32, config.latent_dim, config, rngs=nnx.Rngs(6))
    force = jnp.arange(
        2 * config.force_history_samples * config.force_dim, dtype=jnp.float32
    ).reshape(1, 2, config.force_history_samples, config.force_dim)
    state = jnp.zeros((1, config.force_history_samples, config.force_state_dim))
    mask = jnp.ones((1, 2, config.force_history_samples), dtype=jnp.bool_)

    shared, token_mask = module.encode_history(
        force.reshape(2, config.force_history_samples, config.force_dim),
        jnp.broadcast_to(state[:, None], (1, 2, *state.shape[1:])).reshape(
            2, config.force_history_samples, config.force_state_dim
        ),
        mask.reshape(2, config.force_history_samples),
    )
    slow = module.slow_projection(shared, token_mask)
    fast = module.fast_projection(shared, token_mask)
    assert shared.shape == slow.shape == fast.shape
    assert not jnp.allclose(slow, fast)


def test_fast_context_cannot_see_samples_before_its_raw_window():
    """The fast path must slice 0.33 s before temporal self-attention."""

    config = _small_force_config()
    module = ForceConditioner(64, 32, config.latent_dim, config, rngs=nnx.Rngs(9))
    module.delta_out.kernel.value = 0.01 * jax.random.normal(
        jax.random.key(10), module.delta_out.kernel.value.shape
    )
    batch = 1
    force = jnp.arange(
        2 * config.force_history_samples * config.force_dim, dtype=jnp.float32
    ).reshape(batch, 2, config.force_history_samples, config.force_dim)
    changed_old_force = force.at[:, :, : -config.force_fast_history_samples].add(10_000.0)
    state = jnp.zeros((batch, config.force_history_samples, config.force_state_dim))
    mask = jnp.ones((batch, 2, config.force_history_samples), dtype=jnp.bool_)
    z_model = jnp.ones((batch, 2, config.latent_dim))
    slow_latent = jnp.ones((batch, 2, config.force_latent_dim))
    offset = jnp.zeros((batch,), dtype=jnp.int32)

    first = module.modulate(z_model, slow_latent, force, state, mask, offset)
    second = module.modulate(z_model, slow_latent, changed_old_force, state, mask, offset)
    assert first.recent_tokens.shape[2] == (
        config.force_fast_history_samples // config.force_temporal_stride
    )
    assert jnp.allclose(first.recent_tokens, second.recent_tokens)
    assert jnp.allclose(first.delta_z, second.delta_z)


def test_fast_cross_attention_uses_slow_latent_as_memory_not_query():
    config = _small_force_config()
    module = ForceConditioner(64, 32, config.latent_dim, config, rngs=nnx.Rngs(7))
    # Open the intentionally zero-initialized output gate only for this test.
    module.delta_out.kernel.value = 0.01 * jax.random.normal(
        jax.random.key(8), module.delta_out.kernel.value.shape
    )
    batch = 1
    z_model = jnp.ones((batch, 2, config.latent_dim), dtype=jnp.float32)
    force = jnp.ones((batch, 2, config.force_history_samples, config.force_dim))
    state = jnp.ones((batch, config.force_history_samples, config.force_state_dim))
    mask = jnp.ones((batch, 2, config.force_history_samples), dtype=jnp.bool_)
    offset = jnp.zeros((batch,), dtype=jnp.int32)

    first = module.modulate(
        z_model,
        jnp.zeros((batch, 2, config.force_latent_dim)),
        force,
        state,
        mask,
        offset,
    )
    second = module.modulate(
        z_model,
        jnp.ones((batch, 2, config.force_latent_dim)),
        force,
        state,
        mask,
        offset,
    )
    assert not jnp.allclose(first.delta_z, second.delta_z)


def test_external_qforce_is_sensitive_to_force_history_order():
    """Qforce must retain temporal order rather than mean-pool force tokens."""

    config = _small_force_config()
    module = ForceConditioner(64, 32, config.latent_dim, config, rngs=nnx.Rngs(2))
    batch = 1
    prefix = jnp.ones((batch, 7, 64), dtype=jnp.float32)
    prefix_mask = jnp.ones((batch, 7), dtype=jnp.bool_)
    z_model = jnp.ones((batch, 2, config.latent_dim), dtype=jnp.float32)
    # Each group of four samples is constant. Reversal changes only the order
    # of 30 Hz tokens, not their multiset or within-token layout.
    group_value = jnp.repeat(jnp.arange(3, dtype=jnp.float32), 4)
    force = jnp.broadcast_to(group_value[None, None, :, None], (batch, 2, 12, config.force_dim))
    reversed_force = force[:, :, ::-1]
    state = jnp.zeros((batch, 12, config.force_state_dim), dtype=jnp.float32)
    mask = jnp.ones((batch, 2, 12), dtype=jnp.bool_)

    forward = module.encode_context(prefix, prefix_mask, z_model, force, state, mask)
    backward = module.encode_context(prefix, prefix_mask, z_model, reversed_force, state, mask)
    assert not jnp.allclose(forward.latent, backward.latent)


def test_future_gru_is_latent_only_and_has_no_action_shortcut():
    config = _small_force_config()
    module = ForceConditioner(64, 32, config.latent_dim, config, rngs=nnx.Rngs(3))
    latent = jnp.ones((2, 2, config.force_latent_dim), dtype=jnp.float32)
    first = module.predict_future_force_delta(latent)
    assert not hasattr(module, "future_action_in")

    changed_latent = latent.at[:, 0, 0].set(2.0)
    changed = module.predict_future_force_delta(changed_latent)
    assert not jnp.allclose(first, changed)
    latent_gradient = jax.grad(lambda value: jnp.mean(module.predict_future_force_delta(value)))(
        latent
    )
    assert bool(jnp.all(jnp.isfinite(latent_gradient)))
    assert float(jnp.linalg.norm(latent_gradient)) > 0.0


def test_rtc_batch_uses_clean_prefix_and_fixed_zero_to_horizon_target():
    actions = jnp.arange(2 * 5 * 3, dtype=jnp.float32).reshape(2, 5, 3)
    noise = -jnp.ones_like(actions)
    time = jnp.asarray([0.8, 0.5], dtype=jnp.float32)
    offsets = jnp.asarray([0, 2], dtype=jnp.int32)
    noisy, token_time, committed, target_velocity = rtc_flow_batch(actions, noise, time, offsets)
    assert not bool(jnp.any(committed[0]))
    assert bool(jnp.all(committed[1, :2]))
    assert jnp.array_equal(noisy[1, :2], actions[1, :2])
    assert jnp.array_equal(token_time[1, :2], jnp.zeros((2,)))
    assert jnp.array_equal(target_velocity, noise - actions)


def test_rtc_loss_ignores_committed_prefix_and_weights_suffix_uniformly():
    error = jnp.zeros((1, 5, 2), dtype=jnp.float32)
    offsets = jnp.asarray([2], dtype=jnp.int32)
    committed = jnp.arange(5)[None] < offsets[:, None]
    prefix_only_error = error.at[:, :2].set(1000.0)
    assert float(rtc_flow_loss(prefix_only_error, committed)) == 0.0
    immediate_error = error.at[:, 2].set(1.0)
    distal_error = error.at[:, 3].set(1.0)
    immediate = rtc_flow_loss(immediate_error, committed)
    distal = rtc_flow_loss(distal_error, committed)
    assert float(immediate) == float(distal)


def test_rtc_clamp_preserves_executed_prefix_exactly():
    candidate = jnp.zeros((1, 5, 3), dtype=jnp.float32)
    executed = jnp.arange(15, dtype=jnp.float32).reshape(1, 5, 3)
    committed = jnp.asarray([[True, True, False, False, False]])
    clamped = rtc_clamp_prefix(candidate, executed, committed)
    assert jnp.array_equal(clamped[:, :2], executed[:, :2])
    assert jnp.array_equal(clamped[:, 2:], candidate[:, 2:])
