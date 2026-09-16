#!/usr/bin/env python3
"""Build a direction-safe final Cabinet prompt file without further API calls."""

from __future__ import annotations

import json
from pathlib import Path


DIRECTION = {
    "move_x_pos": "forward", "move_x_neg": "backward",
    "move_y_pos": "left", "move_y_neg": "right",
    "move_z_pos": "up", "move_z_neg": "down",
}


def directions(request: dict) -> list[str]:
    return [DIRECTION[item["skill"]] for item in request["segment"]["gate_labels"] if item["skill"] in DIRECTION]


def needs_repair(request: dict, text: str) -> bool:
    expected = directions(request)
    if not expected:
        return False
    normalized = text.lower().replace("upward", "up").replace("downward", "down").replace("leftward", "left").replace("rightward", "right")
    forbidden = ("+x", "-x", "+y", "-y", "+z", "-z", "negative x", "positive x", "negative y", "positive y", "negative z", "positive z", "along x", "along y", "along z")
    return any(word not in normalized for word in expected) or any(token in normalized for token in forbidden)


def object_phrase(text: str) -> str:
    lower = text.lower()
    if "door handle" in lower or "handle" in lower:
        return "the cabinet door handle"
    if "circuit breaker" in lower:
        return "the circuit breaker"
    if "on button" in lower:
        return "the ON button"
    if "cabinet door" in lower or "door" in lower:
        return "the cabinet door"
    return "the current cabinet target"


def fallback_instruction(request: dict, old_text: str) -> str:
    move = directions(request)
    if len(move) == 1:
        movement = move[0]
    elif len(move) == 2:
        movement = f"{move[0]} and {move[1]}"
    else:
        movement = ", ".join(move[:-1]) + f", and {move[-1]}"
    rotation = []
    for item in request["segment"]["gate_labels"]:
        skill = item["skill"]
        if skill.startswith("rotate_"):
            _, axis, sign = skill.split("_")
            signed_word = "positively" if sign == "pos" else "negatively"
            rotation.append(f"{signed_word} about the base-frame {axis} axis")
    suffix = ""
    if rotation:
        suffix = " while rotating " + " and ".join(rotation)
    return f"Move {movement} toward {object_phrase(old_text)}{suffix}."


def main() -> None:
    package = Path("/home/admin123/ckrc/cabinet_latest_qwen_split")
    manifest = {row["request_id"]: row for row in map(json.loads, (package / "qwen_relabel_manifest.jsonl").read_text().splitlines())}
    original = [row for row in map(json.loads, (package / "qwen_relabel_results.jsonl").read_text().splitlines()) if row.get("ok")]
    repairs = {row["request_id"]: row for row in map(json.loads, (package / "qwen_translation_repair_results.jsonl").read_text().splitlines()) if row.get("ok")}
    final_rows = []
    fallback_count = 0
    for row in original:
        request = manifest[row["request_id"]]
        if not needs_repair(request, row["response"]["low_level_instruction"]):
            final_rows.append({**row, "provenance": "initial_qwen"})
            continue
        repaired = repairs.get(row["request_id"])
        if repaired is not None and not needs_repair(request, repaired["response"]["low_level_instruction"]):
            final_rows.append({**repaired, "provenance": "qwen_direction_repair"})
            continue
        fallback_count += 1
        final_rows.append(
            {
                "request_id": row["request_id"],
                "ok": True,
                "response": {
                    "low_level_instruction": fallback_instruction(request, row["response"]["low_level_instruction"]),
                    "visual_evidence": row["response"]["visual_evidence"],
                },
                "usage": {},
                "provenance": "deterministic_fk_fallback",
            }
        )
    bad = [row["request_id"] for row in final_rows if needs_repair(manifest[row["request_id"]], row["response"]["low_level_instruction"])]
    if bad:
        raise RuntimeError(f"direction audit still fails for {len(bad)} rows: {bad[:5]}")
    output = package / "qwen_relabel_results_final.jsonl"
    output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in final_rows), encoding="utf-8")
    print(json.dumps({"total": len(final_rows), "qwen_direction_repairs": len(repairs), "deterministic_fallbacks": fallback_count, "audit_failures": len(bad), "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
