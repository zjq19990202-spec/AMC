"""Configuration for the clean JAX π0.5-derived atomic policy.

This is intentionally not a wrapper around ``Pi0``.  It only shares the
released OpenPI building blocks needed to load π0.5 parameters: PaliGemma,
SigLIP, the Gemma Action Expert and the OpenPI observation contract.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import gemma as _gemma
from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

if TYPE_CHECKING:
    from .model import AtomicPi05


@dataclasses.dataclass(frozen=True)
class AtomicPi05Config(_model.BaseModelConfig):
    """π0.5 checkpoint-compatible backbone plus only the atomic additions."""

    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"
    action_expert_variant: _gemma.Variant = "gemma_300m"
    action_dim: int = 32
    active_action_dim: int = 16
    active_state_dim: int = 16
    # Native CR1 tensors are [left_arm(7), left_gripper, right_arm(7),
    # right_gripper]. The target corpus is bimanual, so flow supervises all
    # 16 real coordinates while dimensions 16:32 remain checkpoint padding.
    controlled_action_start: int = 0
    controlled_action_dim: int = 16
    action_horizon: int = 50
    max_token_len: int = 200

    # Twelve signed Cartesian motions plus one semantic stay/hold atom.
    # Per-arm vocabulary size. Production bimanual models instantiate
    # arm_count * num_atomic_codes independent spherical prototypes (2*13=26).
    num_atomic_codes: int = 13
    # Auxiliary subtask text is a standard autoregressive continuation rooted
    # at the last valid clean VLM-prefix hidden. Its teacher-forced suffix is
    # mask-isolated from Q1--Q4 and the Action Expert.
    subtask_ce_loss_weight: float = 0.25
    # 64 covers every single/adjacent-pair target in the target corpus
    # (observed maximum 55) without truncating the second crossed subtask.
    subtask_max_token_len: int = 64
    subtask_ce_decode_chunk_size: int = 16
    # Each Action-Expert block reads the ordered right/left z_M pair through
    # the same AFRO 2-head side cross-attention used by the matched 50K base.
    latent_dim: int = 512
    atomic_cross_attention_num_heads: int = 2
    atomic_cross_attention_head_dim: int = 256
    # Q2 predicts coefficients over a small learned basis.  The basis is
    # projected and orthonormalized in Q1's tangent plane for every sample,
    # so it can add detail without changing Q1's atomic direction component.
    detail_dim: int = 4
    state_encoder_hidden_dim: int = 256
    # Each of the 18 side adapters receives z_M through an independent
    # 512->512->(2*96) FiLM generator and writes a 1024->96->1024 residual.
    adapter_condition_hidden_dim: int = 512
    adapter_dim: int = 96
    arm_count: int = 2
    arm_mlp_hidden_dim: int = 512
    arm_fusion_hidden_dim: int = 512
    # Training runs Prefix/Q and Action Expert in one layerwise pass.  At each
    # depth, the current Q1--Q4 states are composed with the same final query
    # heads and arm fusion, then injected through that block's existing
    # zero-init FiLM adapter. Prefix-only inference caches the resulting 18
    # fused latents beside its KV cache and reuses them during denoising.
    enable_layerwise_atomic_flow: bool = True
    # Four tanh-bounded coefficients have an aggregate norm of at most two.
    # A 0.5 multiplier therefore caps the tangent residual norm at one, equal
    # to the unit Q1 direction and corresponding to at most a 45-degree turn.
    max_shift_scale: float = 0.5
    # Optional unit-sphere zM composition. Historical checkpoints retain the
    # exact Euclidean direction+tangent path unless this flag is enabled.
    spherical_visual_latent: bool = False
    visual_max_update_angle_deg: float = 45.0
    # Optional geodesic hinge that keeps visual zM close to its unit zT
    # direction. Angles inside the free cone are unpenalized; outside it the
    # squared excess is normalized by the visual hard cap.
    visual_rotation_loss_weight: float = 0.0
    visual_rotation_free_angle_deg: float = 20.0
    visual_rotation_loss_warmup_steps: int = 0
    # A modestly softer ranking temperature prevents Two-Way from saturating
    # as soon as the target barely outranks the negatives. Ratio KL uses an
    # intermediate tau=0.3: softer than the old 0.07 without requiring the
    # large positive-code cosine gap induced by tau=1.
    atomic_temperature: float = 0.10
    # Multi-label atomic supervision uses the sample-wise term of Two-Way
    # loss: all active atoms must outrank every inactive atom, while
    # simultaneous positives do not compete in one softmax denominator.
    # We intentionally omit its class-wise cross-batch term because only
    # 6--14% of this naturally shuffled batch has strict atomic labels.
    atomic_two_way_positive_temperature: float = 1.0
    atomic_two_way_negative_temperature: float = 1.0
    # Retained only for checkpoint/config compatibility. Current ZT/ZM
    # training does not optimize a ratio KL.
    atomic_ratio_temperature: float = 0.3
    atomic_ratio_loss_weight: float = 0.0
    # Align Q1/Q3 to a normalized weighted code sum: Dual Top-2 in ZT and Drop
    # Top-5 in ZM. No softmax/KL is involved.
    # The old field name is retained so existing checkpoints and launch
    # commands remain structurally resumable.
    atomic_composition_temperature: float = 1.0
    atomic_composition_loss_weight: float = 0.10
    # Deprecated compatibility fields. Stage-B now distills same-prompt/state
    # text-only Q1/Q3 with cosine loss and does not evaluate angular Huber.
    full_atomic_huber_delta_deg: float = 0.0
    full_atomic_huber_angle_scale: float = 0.25

    # The fused right/left z_T predicts one coordinated low-frequency action
    # code. ``tcp_twist`` uses [right 6D, left 6D] cumulative base-frame TCP
    # deltas; ``joint_delta`` uses normalized [right 7D, left 7D] joint deltas.
    # These are supervised in coefficient space.  We deliberately do not
    # reconstruct a hard low-pass trajectory or use frequency/time weights.
    coefficient_count: int = 4
    coefficient_target_kind: str = "tcp_twist"
    coefficient_target_dim: int = 12
    coefficient_loss_weight: float = 1.0
    # Stage-A can train the released PI0.5 Action Expert directly from the
    # same text/state z_T that owns the atomic objectives. Keeping this
    # configurable preserves old coefficient-only checkpoints while the new
    # recipe sets it to one and disables the compact coefficient DiT.
    text_flow_loss_weight: float = 0.0
    # Train the coefficient vector field directly before applying Wall-OSS
    # action-space supervision. A short blend prevents a loss-scale jump at
    # the handoff; set transition to zero for a hard switch.
    coefficient_velocity_warmup_steps: int = 5_000
    coefficient_wall_transition_steps: int = 2_000
    # Four coefficient tokens do not warrant the old 100M, 50-token DiT. This
    # 6 x 512 conditional flow transformer is substantial enough to model the
    # text/state-conditioned coefficient distribution without duplicating the
    # π0.5 300M Action Expert.
    coefficient_dit_width: int = 512
    coefficient_dit_depth: int = 6
    coefficient_dit_mlp_dim: int = 2048
    coefficient_dit_num_heads: int = 4
    coefficient_dit_num_kv_heads: int = 1
    coefficient_dit_head_dim: int = 128

    # π0.5-FAST auxiliary supervision for the text/state z_T phase.  This is
    # the released FAST tokenization (text + discretized normalized state as
    # prefix, FAST action tokens as causal suffix), not a second policy.
    # It complements the continuous Kx6 DCT target: FAST keeps the complete
    # 50x32 action chunk visible to Q1 without feeding an action target into
    # the z_T encoder itself.
    fast_action_ce_loss_weight: float = 0.25
    fast_action_ce_prefix_len: int = 160
    fast_action_ce_suffix_len: int = 250
    fast_action_ce_decode_chunk_size: int = 32

    atomic_loss_weight: float = 1.0
    # Q3/Q4 now belong to the left arm. Numeric-force goals need a dedicated
    # future query rather than sharing Q4 with left-arm detail.
    quantity_loss_weight: float = 0.0
    text_atomic_loss_weight: float = 1.0
    # Semantic codebook anchors are unit vectors. The Huberized geodesic term
    # updates only the selected code (Q1 is stop-gradient); ranking remains
    # responsible for updating Q1 and all discriminative codes.
    codebook_loss_weight: float = 1.0
    # A raw angle is much larger than the former 1-cos(theta) once the codebook
    # is moderately aligned. At the observed loss 1-cos(theta) ~= 0.065,
    # 0.25 * HuberAngle(theta) ~= 0.088: a modest increase that strengthens
    # late alignment while staying far below the former random-init loss.
    codebook_huber_angle_scale: float = 0.25
    codebook_huber_delta_deg: float = 10.0
    # Use the non-diluted spherical codebook loss from the first update. This
    # remains configurable for an ablation, but the default intentionally has
    # no VQ-MSE warm-up.
    codebook_angular_start_step: int = 0
    # Keep visual features stable in the first stage. The language/Action
    # Expert weights remain trainable so Q1--Q4 can reshape task semantics.
    freeze_vision_encoder: bool = True

    # Optional force-conditioned second stage. It is disabled by default so
    # the parameter tree and execution path of an already-trained force-free
    # AtomicPi05 checkpoint remain unchanged. Force training constructs the
    # same model with ``enable_force_stage=True`` and initializes only the new
    # ``force_conditioner`` subtree from scratch.
    enable_force_stage: bool = False
    force_dim: int = 6
    force_state_dim: int = 16
    force_sample_rate_hz: int = 120
    force_action_rate_hz: int = 30
    # Four high-rate force/state samples are packed into one token, producing
    # a token stream aligned one-to-one with the 30 Hz action clock.
    force_temporal_stride: int = 4
    # Slow context sees one second. Fast context sees only the most recent
    # 0.33 s (40 samples / 10 action-aligned tokens) before temporal encoding.
    force_history_samples: int = 120
    force_fast_history_samples: int = 40
    # B1 learns the slow one-second predictive representation from the complete
    # history. The 40-sample fast route is trained by the B2 action objective.
    force_history_train_lengths: tuple[int, ...] = (120,)
    force_future_samples: int = 200
    # Number of raw 120 Hz samples emitted by each recurrent decoder step.
    # 1 gives a 200-step GRU; 4 gives a 50-step action-aligned GRU.
    force_future_decoder_stride: int = 4
    # ``phase_mlp`` shares one MLP across learned within-block phase tokens.
    # It is the default after the held-out 16K decoder A/B; ``linear_chunk``
    # remains available as the independent-slot reconstruction baseline.
    force_future_decoder_kind: str = "phase_mlp"
    force_update_action_steps: int = 10
    force_encoder_width: int = 256
    force_encoder_depth: int = 2
    force_encoder_num_heads: int = 4
    force_encoder_mlp_dim: int = 1024
    force_latent_dim: int = 256
    # Compatibility defaults to the original multimodal zF.  New force-only
    # runs disable this so prefix/zM cannot shortcut the wrench history.
    force_context_from_prefix: bool = True
    # The training-only forecast may use frozen atomic intent while the zF
    # exported to the Action Expert remains force/state-only.
    force_future_condition_on_zm: bool = False
    force_position_base: float = 10_000.0
    # Stacked training-only GRU depth. One layer is sufficient because z_F is
    # used both as its initial state and as a condition at every future step.
    force_future_decoder_depth: int = 1
    # Raw 120 Hz prediction is the main representation objective. A light
    # action-rate pooled term keeps the long-horizon trend coherent.
    force_future_coarse_loss_weight: float = 0.25
    force_future_loss_weight: float = 0.2
    force_flow_loss_weight: float = 1.0
    force_delta_regularization_weight: float = 1.0e-4
    # Require the force-corrected action loss to beat the identical zM-only
    # forward pass.  The baseline is stop-gradient, so it cannot be made worse
    # to satisfy the margin.
    force_improvement_loss_weight: float = 0.0
    force_improvement_margin: float = 0.0
    # Parallel Action-Expert side path: slow+fast force is reduced to one
    # right and one left token, then read by a dedicated cross-attention
    # residual at every Action-Expert depth.
    enable_force_hidden_cross_attention: bool = False
    force_hidden_cross_attention_heads: int = 2
    # Force-to-action adapter used after a completed B1 stage.  It keeps all
    # 30 projected slow tokens and the newest 10 projected fast tokens as K/V;
    # a per-arm zM + RTC-phase query reduces them to one zero-initialized dZ_M.
    force_full_token_adapter: bool = False
    force_full_token_adapter_heads: int = 2
    # Optional unit-sphere force correction. Force still predicts one direct
    # 512-D vector per arm; it is projected onto each layer's zM tangent plane.
    spherical_force_update: bool = False
    force_max_update_angle_deg: float = 15.0
    # Leave useful force corrections inside this cone unpenalized, then apply
    # the same normalized geodesic hinge used by the visual zM route.
    force_rotation_loss_weight: float = 0.0
    force_rotation_free_angle_deg: float = 0.0
    # During the force-only stage the pretrained VLM/z_M route is a fixed
    # teacher. This also avoids retaining its activation graph solely for the
    # auxiliary future-force objective.
    force_stop_gradient_backbone: bool = True

    def __post_init__(self):
        if not 0 < self.visual_max_update_angle_deg < 90:
            raise ValueError("visual_max_update_angle_deg must be in (0, 90)")
        if self.visual_rotation_loss_weight < 0:
            raise ValueError("visual_rotation_loss_weight must be non-negative")
        if not 0 <= self.visual_rotation_free_angle_deg < self.visual_max_update_angle_deg:
            raise ValueError(
                "visual_rotation_free_angle_deg must be in "
                "[0, visual_max_update_angle_deg)"
            )
        if self.visual_rotation_loss_warmup_steps < 0:
            raise ValueError("visual_rotation_loss_warmup_steps must be non-negative")
        if not 0 < self.force_max_update_angle_deg < 90:
            raise ValueError("force_max_update_angle_deg must be in (0, 90)")
        if self.force_rotation_loss_weight < 0:
            raise ValueError("force_rotation_loss_weight must be non-negative")
        if not 0 <= self.force_rotation_free_angle_deg < self.force_max_update_angle_deg:
            raise ValueError(
                "force_rotation_free_angle_deg must be in "
                "[0, force_max_update_angle_deg)"
            )
        if self.force_hidden_cross_attention_heads <= 0:
            raise ValueError("force_hidden_cross_attention_heads must be positive")
        if self.force_full_token_adapter_heads <= 0:
            raise ValueError("force_full_token_adapter_heads must be positive")
        if not 0 < self.active_action_dim <= self.action_dim:
            raise ValueError("active_action_dim must be in [1, action_dim]")
        if not 0 < self.active_state_dim <= self.action_dim:
            raise ValueError("active_state_dim must be in [1, action_dim]")
        if not 0 <= self.controlled_action_start < self.active_action_dim:
            raise ValueError("controlled_action_start must lie inside active actions")
        if self.controlled_action_start + self.controlled_action_dim > self.active_action_dim:
            raise ValueError("controlled action slice exceeds active actions")
        if self.adapter_condition_hidden_dim <= 0 or self.adapter_dim <= 0:
            raise ValueError("adapter dimensions must be positive")
        if self.arm_count != 2:
            raise ValueError("the bimanual atomic policy requires exactly two arms")
        if self.arm_mlp_hidden_dim <= 0 or self.arm_fusion_hidden_dim <= 0:
            raise ValueError("arm fusion dimensions must be positive")
        if (
            self.atomic_cross_attention_num_heads <= 0
            or self.atomic_cross_attention_head_dim <= 0
        ):
            raise ValueError("atomic cross-attention dimensions must be positive")
        if self.subtask_ce_loss_weight < 0:
            raise ValueError("subtask_ce_loss_weight must be non-negative")
        if self.subtask_max_token_len <= 0 or self.subtask_ce_decode_chunk_size <= 0:
            raise ValueError("subtask token lengths must be positive")
        if not 0 < self.detail_dim <= self.latent_dim:
            raise ValueError("detail_dim must be in [1, latent_dim]")
        if not 0 < self.coefficient_count <= self.action_horizon:
            raise ValueError("coefficient_count must be in [1, action_horizon]")
        if self.coefficient_velocity_warmup_steps < 0 or self.coefficient_wall_transition_steps < 0:
            raise ValueError("coefficient warmup/transition steps must be non-negative")
        expected_coefficient_dim = {"tcp_twist": 12, "joint_delta": 14}.get(
            self.coefficient_target_kind
        )
        if expected_coefficient_dim is None:
            raise ValueError("coefficient_target_kind must be tcp_twist or joint_delta")
        if self.coefficient_target_dim != expected_coefficient_dim:
            raise ValueError(
                f"{self.coefficient_target_kind} requires coefficient_target_dim="
                f"{expected_coefficient_dim}, got {self.coefficient_target_dim}"
            )
        if (
            self.coefficient_dit_width <= 0
            or self.coefficient_dit_depth <= 0
            or self.coefficient_dit_mlp_dim <= 0
        ):
            raise ValueError("coefficient DiT width, depth, and MLP width must be positive")
        if (
            self.coefficient_dit_num_heads <= 0
            or self.coefficient_dit_num_kv_heads <= 0
            or self.coefficient_dit_head_dim <= 0
        ):
            raise ValueError("coefficient DiT attention dimensions must be positive")
        if self.coefficient_dit_num_heads % self.coefficient_dit_num_kv_heads:
            raise ValueError(
                "coefficient_dit_num_heads must be divisible by coefficient_dit_num_kv_heads"
            )
        if self.fast_action_ce_loss_weight < 0:
            raise ValueError("fast_action_ce_loss_weight must be non-negative")
        if self.text_flow_loss_weight < 0:
            raise ValueError("text_flow_loss_weight must be non-negative")
        if self.codebook_loss_weight < 0:
            raise ValueError("codebook_loss_weight must be non-negative")
        if self.codebook_huber_angle_scale < 0:
            raise ValueError("codebook_huber_angle_scale must be non-negative")
        if not 0 < self.codebook_huber_delta_deg < 180:
            raise ValueError("codebook_huber_delta_deg must be in (0, 180)")
        if self.atomic_temperature <= 0:
            raise ValueError("atomic_temperature must be positive")
        if (
            self.atomic_two_way_positive_temperature <= 0
            or self.atomic_two_way_negative_temperature <= 0
            or self.atomic_ratio_temperature <= 0
            or self.atomic_composition_temperature <= 0
        ):
            raise ValueError("atomic multi-label temperatures must be positive")
        if self.atomic_ratio_loss_weight < 0:
            raise ValueError("atomic_ratio_loss_weight must be non-negative")
        if self.atomic_composition_loss_weight < 0:
            raise ValueError("atomic_composition_loss_weight must be non-negative")
        if not 0 <= self.full_atomic_huber_delta_deg < 180:
            raise ValueError("full_atomic_huber_delta_deg must be in [0, 180)")
        if self.full_atomic_huber_angle_scale < 0:
            raise ValueError("full_atomic_huber_angle_scale must be non-negative")
        if self.codebook_angular_start_step < 0:
            raise ValueError("codebook_angular_start_step must be non-negative")
        if self.fast_action_ce_prefix_len <= 0 or self.fast_action_ce_suffix_len <= 0:
            raise ValueError("FAST prefix/suffix lengths must be positive")
        if self.fast_action_ce_decode_chunk_size <= 0:
            raise ValueError("FAST decode chunk size must be positive")
        if self.force_dim <= 0 or self.force_state_dim <= 0:
            raise ValueError("force and force-state dimensions must be positive")
        if self.force_sample_rate_hz <= 0 or self.force_action_rate_hz <= 0:
            raise ValueError("force/action sample rates must be positive")
        if self.force_sample_rate_hz != self.force_action_rate_hz * self.force_temporal_stride:
            raise ValueError("force rate must equal action rate times force_temporal_stride")
        if (
            self.force_history_samples <= 0
            or self.force_history_samples % self.force_temporal_stride
        ):
            raise ValueError(
                "force_history_samples must be a positive multiple of the temporal stride"
            )
        if (
            self.force_fast_history_samples <= 0
            or self.force_fast_history_samples % self.force_temporal_stride
            or self.force_fast_history_samples > self.force_history_samples
        ):
            raise ValueError(
                "force_fast_history_samples must be a positive temporal-stride multiple "
                "not exceeding force_history_samples"
            )
        if (
            self.force_fast_history_samples
            != self.force_update_action_steps * self.force_temporal_stride
        ):
            raise ValueError("force_fast_history_samples must align with force_update_action_steps")
        if not self.force_history_train_lengths:
            raise ValueError("force_history_train_lengths must not be empty")
        if any(
            length <= 0
            or length > self.force_history_samples
            or length % self.force_temporal_stride
            for length in self.force_history_train_lengths
        ):
            raise ValueError(
                "force_history_train_lengths must be positive temporal-stride multiples "
                "not exceeding force_history_samples"
            )
        if self.force_future_samples != self.action_horizon * self.force_temporal_stride:
            raise ValueError("force_future_samples must align one-to-one with the action horizon")
        if (
            self.force_future_decoder_stride <= 0
            or self.force_future_samples % self.force_future_decoder_stride
        ):
            raise ValueError(
                "force_future_decoder_stride must be positive and divide force_future_samples"
            )
        if self.force_future_decoder_kind not in ("linear_chunk", "phase_mlp"):
            raise ValueError(
                "force_future_decoder_kind must be 'linear_chunk' or 'phase_mlp'"
            )
        if not 0 < self.force_update_action_steps <= self.action_horizon:
            raise ValueError("force_update_action_steps must be in [1, action_horizon]")
        if (
            self.force_encoder_width <= 0
            or self.force_encoder_depth <= 0
            or self.force_encoder_num_heads <= 0
            or self.force_encoder_mlp_dim <= 0
            or self.force_latent_dim <= 0
            or self.force_future_decoder_depth <= 0
        ):
            raise ValueError("force encoder/latent/decoder dimensions must be positive")
        if self.force_encoder_width % self.force_encoder_num_heads:
            raise ValueError("force_encoder_width must be divisible by force_encoder_num_heads")
        if self.force_position_base <= 1:
            raise ValueError("force_position_base must be greater than one")
        if (
            self.force_future_coarse_loss_weight < 0
            or self.force_future_loss_weight < 0
            or self.force_flow_loss_weight < 0
            or self.force_delta_regularization_weight < 0
        ):
            raise ValueError("force-stage loss weights must be non-negative")
    def coefficient_dit_config(self) -> _gemma.Config:
        """Gemma-compatible configuration for Q1's four-token coefficient DiT."""

        return _gemma.Config(
            width=self.coefficient_dit_width,
            depth=self.coefficient_dit_depth,
            mlp_dim=self.coefficient_dit_mlp_dim,
            num_heads=self.coefficient_dit_num_heads,
            num_kv_heads=self.coefficient_dit_num_kv_heads,
            head_dim=self.coefficient_dit_head_dim,
        )

    @property
    @override
    def model_type(self) -> _model.ModelType:
        # Reuse π0.5 transforms: state is discretized into the PaliGemma prompt
        # and state/action are padded to 32 after normalization.
        return _model.ModelType.PI05

    @override
    def create(self, rng: at.KeyArrayLike) -> "AtomicPi05":
        from .model import AtomicPi05

        return AtomicPi05(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        image = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask = jax.ShapeDtypeStruct([batch_size], jnp.bool_)
        with at.disable_typechecking():
            observation = _model.Observation(
                images={
                    "base_0_rgb": image,
                    "left_wrist_0_rgb": image,
                    "right_wrist_0_rgb": image,
                },
                image_masks={
                    "base_0_rgb": image_mask,
                    "left_wrist_0_rgb": image_mask,
                    "right_wrist_0_rgb": image_mask,
                },
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
            )
        actions = jax.ShapeDtypeStruct(
            [batch_size, self.action_horizon, self.action_dim], jnp.float32
        )
        return observation, actions

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Freeze only SigLIP by default; LLM, Action Expert and new heads learn."""

        filters = []
        if self.freeze_vision_encoder:
            # NNX flat-state paths use ``/`` as their separator, e.g.
            # ``PaliGemma/img/Transformer/...``; a dotted pattern silently
            # fails to freeze the visual encoder.
            filters.append(nnx_utils.PathRegex(r".*PaliGemma/img(?:/.*)?"))
        gemma_params = nnx_utils.PathRegex(r".*PaliGemma/llm(?:/.*)?")
        action_expert_params = nnx_utils.PathRegex(r".*PaliGemma/llm(?:/.*)?_1(?:/.*)?")
        has_lora = False
        if "lora" in self.paligemma_variant:
            filters.append(gemma_params)
            if "lora" not in self.action_expert_variant:
                filters.append(nnx.Not(action_expert_params))
            has_lora = True
        elif "lora" in self.action_expert_variant:
            filters.append(action_expert_params)
            has_lora = True
        if has_lora:
            filters.append(nnx.Not(nnx_utils.PathRegex(".*lora.*")))
        if not filters:
            return nnx.Nothing
        if self.freeze_vision_encoder and len(filters) > 1:
            # Vision is always frozen; LoRA-related predicates jointly define
            # any additional freeze rule.
            return nnx.Any(filters[0], nnx.All(*filters[1:]))
        return filters[0] if len(filters) == 1 else nnx.All(*filters)

    def get_force_freeze_filter(
        self,
        *,
        train_atomic_adapters: bool = False,
        train_future_decoder: bool = True,
    ) -> nnx.filterlib.Filter:
        """Freeze the force-free VLA during force-stage training.

        Stage B1 trains only ``force_conditioner``. Stage B2 additionally
        fine-tunes the *same* Action Expert, its action/time projections, the
        atomic side adapters. Per-token flow time is the only clean-prefix role
        signal, matching Training-Time RTC without additional learned role
        parameters. SigLIP, the VLM stream, Q heads and codebook remain fixed.
        """

        if not self.enable_force_stage:
            raise ValueError("get_force_freeze_filter requires enable_force_stage=True")
        force = nnx_utils.PathRegex(r".*force_conditioner(?:/.*)?")
        if not train_atomic_adapters:
            # B1 obtains the shared temporal representation exclusively from
            # slow future-force prediction.  Keep fast-only projection/fusion
            # parameters bitwise at initialization instead of letting AdamW
            # decay tensors that have no gradient path in this stage.
            fast_only = nnx_utils.PathRegex(
                r".*force_conditioner/(?:fast_projection|fast_query|"
                r"fast_query_norm|slow_memory_from_latent|force_scale_embedding|"
                r"fast_attention|fast_hidden|delta_out|layer_gate_logits)(?:/.*)?"
            )
            return nnx.Any(nnx.Not(force), fast_only)
        future_decoder = nnx_utils.PathRegex(r".*force_conditioner/future_.*")
        force_trainable = (
            force if train_future_decoder else nnx.All(force, nnx.Not(future_decoder))
        )
        adapters = nnx_utils.PathRegex(r".*atomic_adapter(?:/.*)?")
        action_io = nnx_utils.PathRegex(
            r".*(action_in_proj|action_out_proj|time_mlp_in|time_mlp_out)(?:/.*)?"
        )
        action_expert = nnx_utils.PathRegex(r".*PaliGemma/llm(?:/.*)?_1(?:/.*)?")
        # These modules live outside ``force_conditioner`` under names such as
        # ``force_cross_attention`` and ``force_cross_out``. Match the full
        # component name: the old ``force_cross_(?:/.*)?`` pattern stopped at
        # the underscore and silently froze the entire force side path.
        force_cross_attention = nnx_utils.PathRegex(
            r".*force_cross_(?:attention|out|query_norm)(?:/.*)?"
        )
        return nnx.Not(
            nnx.Any(force_trainable, adapters, action_io, action_expert, force_cross_attention)
        )

    def get_force_action_only_freeze_filter(self) -> nnx.filterlib.Filter:
        """Train force-to-action conditioning without an Action-Expert shortcut.

        The shared slow/fast force encoder, cross-attention correction and layer
        gates remain trainable. The training-only future-force decoder is frozen together with the VLA,
        Action Expert, action/time projections and atomic adapters.
        """

        if not self.enable_force_stage:
            raise ValueError(
                "get_force_action_only_freeze_filter requires enable_force_stage=True"
            )
        force = nnx_utils.PathRegex(r".*force_conditioner(?:/.*)?")
        force_cross_attention = nnx_utils.PathRegex(
            r".*force_cross_(?:attention|out|query_norm)(?:/.*)?"
        )
        future_decoder = nnx_utils.PathRegex(r".*force_conditioner/future_.*")
        return nnx.Any(
            nnx.Not(nnx.Any(force, force_cross_attention)),
            future_decoder,
        )

    def get_force_full_token_adapter_freeze_filter(
        self, *, train_action_path: bool = False, train_zf_path: bool = False
    ) -> nnx.filterlib.Filter:
        """Freeze B1 and the complete VLA; train only the new force-to-dZ_M adapter.

        B1 already trained the shared temporal encoder and slow projection to
        encode predictive force.  B2 learns only the previously-unused fast
        projection plus the 30+10-token cross-attention adapter and its layer
        gates.  Gradients still pass through the frozen Action Expert so the
        adapter learns a correction in the existing AFRO control manifold.
        """

        if not self.enable_force_stage or not self.force_full_token_adapter:
            raise ValueError(
                "full-token adapter freeze filter requires enable_force_stage=True "
                "and force_full_token_adapter=True"
            )
        adapter = nnx_utils.PathRegex(
            r".*force_conditioner/(?:fast_projection|full_token_adapter_[^/]+|"
            r"layer_gate_logits)(?:/.*)?"
        )
        if train_zf_path:
            force = nnx_utils.PathRegex(r".*force_conditioner(?:/.*)?")
            future_decoder = nnx_utils.PathRegex(r".*force_conditioner/future_.*")
            adapter = nnx.All(force, nnx.Not(future_decoder))
        if not train_action_path:
            return nnx.Not(adapter)

        # Training-Time RTC needs the suffix generator itself to learn how to
        # continue a clean committed prefix.  Keep the visual/VLM and atomic
        # planning path frozen, but allow the Action Expert and its action/time
        # projections to adapt together with the force adapter.
        action_io = nnx_utils.PathRegex(
            r".*(action_in_proj|action_out_proj|time_mlp_in|time_mlp_out)(?:/.*)?"
        )
        action_expert = nnx_utils.PathRegex(r".*PaliGemma/llm(?:/.*)?_1(?:/.*)?")
        return nnx.Not(nnx.Any(adapter, action_io, action_expert))
