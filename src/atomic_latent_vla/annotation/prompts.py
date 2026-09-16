from __future__ import annotations

import json
import textwrap

from .schema import CandidateAnnotation
from .timeline import FixedAtomicSegment


def system_prompt(
    axis_convention: str,
    *,
    min_segment_duration_s: float = 2.0,
    enable_interaction_gate: bool = False,
) -> str:
    interaction_rule = (
        """
        4. Set `strong_interaction=true` only when the RIGHT arm is visibly in sustained
           force/contact-rich interaction or its motion is dominated by grasp/release/contact.
           Do not mark interaction merely because the LEFT arm is contacting something.
        """
        if enable_interaction_gate
        else ""
    )
    segment_fields = (
        "`segment_id`, `start_s`, `end_s`, `low_level_instruction`, `visual_evidence`, "
        "and `strong_interaction`."
        if enable_interaction_gate
        else "`segment_id`, `start_s`, `end_s`, `low_level_instruction`, and `visual_evidence`."
    )
    instruction_rule_number = 5 if enable_interaction_gate else 4
    return textwrap.dedent(
        f"""
        # Role
        You are the semantic vision annotator for a robot demonstration. Local right-arm forward
        kinematics has already fixed every segment boundary, signed atomic direction, and gate.
        You must only explain what those fixed right-arm intervals mean in the video: objects,
        task purpose, grasp/contact state, and a natural low-level instruction.

        Robot coordinate convention (provided only to interpret the fixed FK hints):
        {axis_convention}

        # Arm identity and annotation scope
        The base view may show BOTH robot arms. The wrist view named `right_wrist` and every supplied
        FK timeline belong only to the RIGHT arm. The top-level `global_description` must describe
        the complete episode and explicitly distinguish LEFT-arm-only, RIGHT-arm-only, and
        simultaneous bimanual phases. Never reinterpret visible LEFT-arm motion as right-arm motion.

        # Non-negotiable fixed-timeline rules
        1. Return exactly one semantic record for every supplied fixed segment, in the same order.
        2. Copy every `segment_id`, `start_s`, and `end_s` exactly. Never split, merge, reorder,
           delete, stretch, or invent an interval.
        3. `right_fk_mode`, `right_fk_atoms`, and `right_fk_gate_reason` in the supplied timeline
           are read-only local facts. `right_fk_gate_reason` is audit context: it may help you
           distinguish a stationary, conflicting, or otherwise unresolved neighboring interval
           when writing semantics for a retained interval, but it is not visual evidence and must
           not be copied as a motion claim. Do not output replacement probabilities or labels.
        {interaction_rule}
        {instruction_rule_number}. Write exactly one natural, compact `low_level_instruction`, usually
           12--24 English words, for every interval. Derive this current-segment prompt from the original episode task,
           your global understanding of
           the complete demonstration, the visible object/purpose, and the immutable FK atom. For a
           clear FK atom, preserve every move/rotate family, base-frame axis, and signed direction in
           `right_fk_atoms`; enrich it with task semantics instead of producing generic paraphrases.
           Every `single`/`dual` instruction must express where the motion goes. For translation,
           use the natural base-frame words in `direction_hints`; never write
           `+x/-x/+y/-y/+z/-z` for a translation phrase,
           or express an unambiguous visible target-relative translation, such as moving the tool
           toward/into the screw hole or withdrawing it away from the fixture.
           A translation written relative to a target must state where that target lies relative to
           the current TCP or held object in the same instruction (for example, "the hole ahead and
           left of the tip"). Do not rely on `visual_evidence` to supply this missing direction.
           Natural base-frame words are the only allowed axis wording for translation: forward=`+x`,
           backward=`-x`, left=`+y`, right=`-y`, up=`+z`, and down=`-z`.
           Prefer natural task language when the target relation is visually clear; do not force an
           axis phrase merely to restate the FK label. A dual instruction must ground both motion
           components. Never use only vague verbs such as adjust or position for a retained atom.
           Vary verbs and sentence structure naturally across intervals instead of repeatedly copying
           `Right arm moves ...` or the supplied direction hints verbatim. This linguistic variation
           must never alter the fixed atom, direction, object, or visible task purpose.
           Rotation is the exception: clockwise/counterclockwise depends on viewpoint, so every
           rotation atom must explicitly retain its signed base-frame axis (for example, positive
           about `+z` or negative about `+y`). A target orientation such as aligning with a slot may
           be added for task meaning, but cannot replace the rotation axis and sign.
           For `drop`/stationary, write only a brief auditable state description; downstream training
           discards it, so do not invent a motion or axis. Never invent an object, amount, distance,
           speed, or force.
           Use concise wording with mild variation. Prefer 10--24 words. Do not over-diversify:
           keep the same atomic meaning clear. Avoid starting every instruction with `Right arm`;
           imperative phrasing is allowed.
           Use conservative object names throughout `global_description` and segment instructions:
           reuse object names from the original episode task verbatim. Never rename a task object,
           infer an unseen function, or invent another/second tool from appearance. If an identity is
           uncertain, say `tool`, `object`, `target`, or `surface` instead of guessing. Never mention
           a visible brand/model unless that exact name occurs in the episode task. Never introduce
           a second/another instance of a task object unless the episode task explicitly says there
           are multiple instances. Do not mention people, controllers, teleoperation, or human
           assistance unless the episode task explicitly requires them. A visible background person
           or device is not evidence that it controls or assists the robot.

        # Required internal order in this one generation
        Step 1 — Global understanding: watch the entire episode, then write the top-level
        `global_description` with its goal, objects, chronological phases, visible grasp/contact
        changes, and which phases use the LEFT arm, RIGHT arm, or both.
        Step 2 — Fixed-interval semantic pass: revisit each supplied interval in order, interpret
        the read-only right-FK hint against both camera views and the global task understanding, then
        fill its current-segment instruction and concise visual evidence.

        Perform these steps internally, but because the API requires JSON mode, do not emit a
        free-form reasoning preamble.
        Return one JSON object only. Emit top-level keys in this order:
        `global_description`, then `segments`.
        `segments` is an array, and every segment must contain exactly:
        {segment_fields}
        """
    ).strip()


def proposal_user_text(
    *,
    task: str,
    duration_s: float,
    sampled_fps: float,
    view_names: list[str],
    min_segment_duration_s: float,
    fixed_timeline: list[FixedAtomicSegment],
    extra_context: str = "",
    enable_interaction_gate: bool = False,
) -> str:
    timeline_json = json.dumps(
        [segment.prompt_payload() for segment in fixed_timeline],
        ensure_ascii=False,
    )
    semantic_fields = (
        "`low_level_instruction`, `visual_evidence`, and `strong_interaction`"
        if enable_interaction_gate
        else "`low_level_instruction` and `visual_evidence`"
    )
    return textwrap.dedent(
        f"""
        Analyze this synchronized robot demonstration.

        Episode task: {task}
        Duration: {duration_s:.3f} seconds
        Supplied frame rate: {sampled_fps:.4f} frames/second
        Synchronized views in each montage frame: {", ".join(view_names)}
        Additional context: {extra_context or "none"}

        Annotation scope: describe both arms in `global_description`; annotate the visual meaning
        of the fixed RIGHT-arm intervals below. The local program has already computed this list
        from right-arm FK. It is authoritative and immutable:
        {timeline_json}

        First watch the whole video for global context. Then return exactly {len(fixed_timeline)}
        segment records, copying each id/start/end exactly and adding only {semantic_fields}.
        Generate the one
        instruction for each interval from the original episode task, the complete-video global
        understanding, and the current fixed atom. The supplied FK atom hints fix direction; its
        gate reason is audit context for neighboring rejected intervals only. The video supplies
        object, purpose, contact, arm identity, and target-relative spatial relation.
        Every translation must use its matching natural base direction or an unambiguous visible
        target-relative direction; never write `±xyz` for translation. A target-relative translation must describe the
        target's location relative to the current TCP/tool in that same instruction. Every rotation
        must explicitly state its signed base-frame axis.
        `drop` is audit-only and may use a brief state description.
        Aim for 10--24 English words per instruction; a stationary audit sentence may be shorter.
        Use mild wording variation without changing any fixed direction, and avoid starting every
        instruction with `Right arm`; imperative phrasing is allowed. Reuse object
        names from `Episode task` verbatim; do not invent a clamp, another tool, or a functional
        identity not stated in the task. Ignore visible brand/model text and do not introduce a
        second instance of a task object unless the task explicitly says it exists. Ignore background
        people/controllers and never infer human assistance unless the task explicitly requires it.
        Every single/dual interval satisfies the local minimum-duration policy. A short `drop`
        interval may be supplied solely to preserve rejected timeline coverage; return all intervals
        unchanged.
        """
    ).strip()


def review_user_text(
    *,
    task: str,
    duration_s: float,
    sampled_fps: float,
    proposal: CandidateAnnotation,
    min_segment_duration_s: float,
    fixed_timeline: list[FixedAtomicSegment],
    extra_context: str = "",
    enable_interaction_gate: bool = False,
) -> str:
    proposal_json = json.dumps(proposal.model_dump(mode="json"), ensure_ascii=False)
    timeline_json = json.dumps(
        [segment.prompt_payload() for segment in fixed_timeline],
        ensure_ascii=False,
    )
    interaction_review = (
        "Correct the right-arm interaction flag only for visibly sustained contact."
        if enable_interaction_gate
        else "Do not add an interaction flag; interaction gating is disabled."
    )
    return textwrap.dedent(
        f"""
        Re-review the same video and correct the candidate annotation below.

        Task: {task}
        Duration: {duration_s:.3f} seconds
        Supplied frame rate: {sampled_fps:.4f} frames/second
        Additional context: {extra_context or "none"}
        Immutable local right-FK timeline:
        {timeline_json}

        Candidate JSON:
        {proposal_json}

        Re-watch the full video, correct the global description, arm identity, objects, contact
        state, current-segment instructions, and evidence. Each instruction must remain consistent
        with the original task, global episode description, and exact fixed atom/axis direction.
        Reject vague retained instructions: every `single`/`dual` component must be grounded by a
        matching natural base direction, signed base-frame axis, or an unambiguous visible
        target-relative direction. For target-relative translation, verify that the same instruction
        states where the target lies relative to the TCP/tool. Rotation must always preserve its
        signed base-frame axis because viewpoint-relative clockwise wording is ambiguous.
        Aim for 12--24 English words, vary wording naturally without changing the fixed atom, retain
        task object names verbatim, and replace any uncertain visual identity with `tool`, `object`,
        `target`, or `surface` rather than guessing. Remove brand/model names and any invented second
        instance unless the episode task explicitly contains them. Remove unsupported claims about
        people, controllers, teleoperation, or human assistance.
        Drop/stationary text is audit-only and must not invent motion.
        {interaction_review}
        Do not change any segment id/time and do not create atomic probabilities: those are fixed by
        the local FK timeline.

        Return the complete corrected JSON object only, using the original schema.
        """
    ).strip()


def build_messages(system: str, video_item: dict, user_text: str) -> list[dict]:
    return [
        {"role": "system", "content": [{"type": "text", "text": system}]},
        {
            "role": "user",
            "content": [video_item, {"type": "text", "text": user_text}],
        },
    ]
