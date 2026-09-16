from __future__ import annotations

import jax.numpy as jnp
import pytest

from atomic_latent_vla.pi05.model import bimanual_flow_losses


def test_stock_flow_averages_all_32_dimensions() -> None:
    predicted = jnp.zeros((2, 5, 32))
    target = jnp.zeros_like(predicted).at[..., :8].set(1.0)
    actions = jnp.zeros_like(predicted).at[..., :7].set(1.0)

    total, unweighted, left, right, active, left_share, right_share = (
        bimanual_flow_losses(predicted, target, actions)
    )

    assert float(left) == pytest.approx(1.0)
    assert float(right) == pytest.approx(0.0)
    assert float(unweighted) == pytest.approx(0.25)
    assert float(active) == pytest.approx(1.0)
    assert float(total) == pytest.approx(0.25)
    assert float(left_share) == pytest.approx(1.0)
    assert float(right_share) == pytest.approx(0.0)


def test_no_motion_changes_telemetry_not_the_stock_loss() -> None:
    predicted = jnp.zeros((1, 3, 32))
    target = jnp.zeros_like(predicted).at[..., :8].set(1.0)
    actions = jnp.zeros_like(predicted)

    total, unweighted, _, _, active, left_share, right_share = bimanual_flow_losses(
        predicted, target, actions
    )

    assert float(unweighted) == pytest.approx(0.25)
    assert float(active) == pytest.approx(0.5)
    assert float(total) == pytest.approx(0.25)
    assert float(left_share) == pytest.approx(0.5)
    assert float(right_share) == pytest.approx(0.5)


def test_padding_dimensions_participate_in_stock_flow_loss() -> None:
    predicted = jnp.zeros((1, 2, 32))
    target = jnp.zeros_like(predicted).at[..., 16:].set(1.0)
    actions = jnp.zeros_like(predicted)

    total, _, left, right, _, _, _ = bimanual_flow_losses(predicted, target, actions)

    assert float(total) == pytest.approx(0.5)
    assert float(left) == pytest.approx(0.0)
    assert float(right) == pytest.approx(0.0)


def test_zt_flow_masks_unsupervised_arm_grippers_and_padding() -> None:
    predicted = jnp.zeros((1, 2, 32))
    target = jnp.zeros_like(predicted)
    target = target.at[..., :7].set(2.0)
    target = target.at[..., 7].set(100.0)
    target = target.at[..., 8:15].set(3.0)
    target = target.at[..., 15:].set(100.0)
    actions = jnp.zeros_like(predicted)

    total, stock, left, right, active, _, _ = bimanual_flow_losses(
        predicted,
        target,
        actions,
        atomic_arm_mask=jnp.asarray([[True, False]]),
    )

    assert float(total) == pytest.approx(9.0)
    assert float(right) == pytest.approx(9.0)
    assert float(left) == pytest.approx(0.0)
    assert float(active) == pytest.approx(9.0)
    assert float(stock) > 9.0


def test_zt_flow_averages_only_both_arms_seven_joint_coordinates() -> None:
    predicted = jnp.zeros((1, 2, 32))
    target = jnp.zeros_like(predicted)
    target = target.at[..., :7].set(2.0)
    target = target.at[..., 7].set(100.0)
    target = target.at[..., 8:15].set(3.0)
    target = target.at[..., 15:].set(100.0)

    total, _, left, right, _, _, _ = bimanual_flow_losses(
        predicted,
        target,
        jnp.zeros_like(predicted),
        atomic_arm_mask=jnp.asarray([[True, True]]),
    )

    assert float(total) == pytest.approx((4.0 + 9.0) / 2.0)
    assert float(left) == pytest.approx(4.0)
    assert float(right) == pytest.approx(9.0)
