from atomic_latent_vla.annotation.prompts import system_prompt


def test_prompt_keeps_vlm_semantic_only() -> None:
    prompt = system_prompt("test convention")
    assert "`strong_interaction`" not in prompt
    assert "Do not output replacement probabilities or labels" in prompt
    assert "right_fk_gate_reason" in prompt
    assert "audit context" in prompt
    assert "fixed segment" in prompt
    assert "RIGHT arm" in prompt
    assert "use the natural base-frame words" in prompt
    assert "visible target-relative translation" in prompt
    assert "do not force an" in prompt
    assert "forward=`+x`" in prompt
    assert "left=`+y`" in prompt
    assert "same instruction" in prompt
    assert "Do not rely on `visual_evidence`" in prompt
    assert "Rotation is the exception" in prompt
    assert "cannot replace the rotation axis and sign" in prompt
    assert "downstream training" in prompt
    assert "12--24 English words" in prompt
    assert "Vary verbs and sentence structure" in prompt
    assert "mild variation" in prompt
    assert "Avoid starting every instruction with `Right arm`" in prompt
    assert "imperative phrasing is allowed" in prompt
    assert "visible brand/model" in prompt
    assert "second/another instance" in prompt
    assert "Do not mention people, controllers" in prompt
    assert "invent another/second tool" in prompt
    assert "temporal_support" not in prompt
    assert "pair_overlap" not in prompt


def test_prompt_can_opt_in_to_interaction_gate() -> None:
    prompt = system_prompt("test convention", enable_interaction_gate=True)
    assert "`strong_interaction`" in prompt
    assert "sustained" in prompt
