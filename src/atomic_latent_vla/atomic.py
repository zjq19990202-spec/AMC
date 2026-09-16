from __future__ import annotations

from enum import Enum


class AtomicSkill(str, Enum):
    MOVE_X_POS = "move_x_pos"
    MOVE_X_NEG = "move_x_neg"
    MOVE_Y_POS = "move_y_pos"
    MOVE_Y_NEG = "move_y_neg"
    MOVE_Z_POS = "move_z_pos"
    MOVE_Z_NEG = "move_z_neg"
    ROTATE_X_POS = "rotate_x_pos"
    ROTATE_X_NEG = "rotate_x_neg"
    ROTATE_Y_POS = "rotate_y_pos"
    ROTATE_Y_NEG = "rotate_y_neg"
    ROTATE_Z_POS = "rotate_z_pos"
    ROTATE_Z_NEG = "rotate_z_neg"
    STAY = "stay"


ATOMIC_NAMES = tuple(skill.value for skill in AtomicSkill)
ATOMIC_SKILL_TO_ID = {skill: index for index, skill in enumerate(AtomicSkill)}
ATOMIC_ID_TO_SKILL = {index: skill for skill, index in ATOMIC_SKILL_TO_ID.items()}
OPPOSITE_ATOMIC_ID = {
    0: 1,
    1: 0,
    2: 3,
    3: 2,
    4: 5,
    5: 4,
    6: 7,
    7: 6,
    8: 9,
    9: 8,
    10: 11,
    11: 10,
}

MOTION_ATOMIC_COUNT = 12
NUM_ATOMIC_SKILLS = len(AtomicSkill)
STAY_ATOMIC_ID = ATOMIC_SKILL_TO_ID[AtomicSkill.STAY]


ATOMIC_BASE_INSTRUCTIONS = {
    AtomicSkill.MOVE_X_POS: "Move the TCP forward along base-frame +x.",
    AtomicSkill.MOVE_X_NEG: "Move the TCP backward along base-frame -x.",
    AtomicSkill.MOVE_Y_POS: "Move the TCP left along base-frame +y.",
    AtomicSkill.MOVE_Y_NEG: "Move the TCP right along base-frame -y.",
    AtomicSkill.MOVE_Z_POS: "Move the TCP upward along base-frame +z.",
    AtomicSkill.MOVE_Z_NEG: "Move the TCP downward along base-frame -z.",
    AtomicSkill.ROTATE_X_POS: "Rotate the TCP positively about the base-frame x axis.",
    AtomicSkill.ROTATE_X_NEG: "Rotate the TCP negatively about the base-frame x axis.",
    AtomicSkill.ROTATE_Y_POS: "Rotate the TCP positively about the base-frame y axis.",
    AtomicSkill.ROTATE_Y_NEG: "Rotate the TCP negatively about the base-frame y axis.",
    AtomicSkill.ROTATE_Z_POS: "Rotate the TCP positively about the base-frame z axis.",
    AtomicSkill.ROTATE_Z_NEG: "Rotate the TCP negatively about the base-frame z axis.",
    AtomicSkill.STAY: "Keep the TCP stationary in the current base-frame pose.",
}


def vocabulary_prompt(axis_convention: str) -> str:
    meanings = (
        "translate the TCP along base-frame +x",
        "translate the TCP along base-frame -x",
        "translate the TCP along base-frame +y",
        "translate the TCP along base-frame -y",
        "translate the TCP along base-frame +z",
        "translate the TCP along base-frame -z",
        "rotate the TCP about +x by the right-hand rule",
        "rotate the TCP about -x by the right-hand rule",
        "rotate the TCP about +y by the right-hand rule",
        "rotate the TCP about -y by the right-hand rule",
        "rotate the TCP about +z by the right-hand rule",
        "rotate the TCP about -z by the right-hand rule",
        "keep the TCP stationary at its current pose",
    )
    table = "\n".join(
        f"- {index}: `{skill.value}` = {meaning}"
        for index, (skill, meaning) in enumerate(
            zip(AtomicSkill, meanings, strict=True)
        )
    )
    return f"Robot coordinate convention:\n{axis_convention}\n\nClosed atomic vocabulary:\n{table}"
