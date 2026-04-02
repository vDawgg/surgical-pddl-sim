import re
from enum import StrEnum, auto
from pathlib import Path

import numpy as np
from isaacsim.core.prims import SingleXFormPrim

from src.base.task import Arms, DvrkBaseTask


class ActionType(StrEnum):
    MOVE = auto()
    PICK = auto()
    PLACE = auto()


class ParsingException(Exception): ...


class Action:
    def __init__(
        self,
        action_type: ActionType,
        target_position: np.ndarray | None = None,
        target_orientation: np.ndarray | None = None,
        robot_name: str | None = None,
        target_prim_name: str | None = None,
        target_position_offset: np.ndarray | None = None,
        waypoint_reached: bool = False,
        waypoint_threshold: float | None = None,
        waypoint_targets: list[tuple[np.ndarray, np.ndarray | None]] | None = None,
        waypoint_index: int = 0,
        target_joint_indices: np.ndarray | None = None,
        target_joint_positions: np.ndarray | None = None,
    ):
        self.action_type = action_type
        self.target_position = target_position
        self.target_orientation = target_orientation
        self.robot_name = robot_name
        self.target_prim_name = target_prim_name
        self.target_position_offset = target_position_offset
        self.waypoint_reached = waypoint_reached
        self.waypoint_threshold = waypoint_threshold
        self.waypoint_targets = waypoint_targets or []
        self.waypoint_index = waypoint_index
        self.target_joint_indices = target_joint_indices
        self.target_joint_positions = target_joint_positions

    @classmethod
    def from_prim(
        cls,
        action_type: ActionType,
        prim: SingleXFormPrim | None = None,
        robot_name: str | None = None,
        target_prim_name: str | None = None,
    ):
        """
        Creates an action of any type with the specified target position and orientation of the given prim.
        The target can be omitted for the pick and place action, as these actions only control the gripper.
        """
        if prim is None:
            return cls(
                action_type=action_type,
                robot_name=robot_name,
                target_prim_name=target_prim_name,
            )
        else:
            target_position, target_orientation = prim.get_world_pose()
            return cls(
                action_type=action_type,
                target_position=target_position,
                target_orientation=target_orientation,
                robot_name=robot_name,
                target_prim_name=target_prim_name,
                target_position_offset=np.zeros(3),
            )


class Plan:
    def __init__(self, action_sequence: list[Action]):
        self.action_sequence = action_sequence

    @classmethod
    def from_actions(cls, action_sequence: list[Action]):
        return cls(action_sequence)

    @classmethod
    def from_pddl(cls, task: DvrkBaseTask, plan_path: str | Path):
        move_pattern = r"\(\s*move\b([^)]*)\)"
        pick_pattern = r"\(\s*pick\b([^)]*)\)"
        place_pattern = r"\(\s*place\b([^)]*)\)"

        def _extract_robot_name(tokens: list[str]) -> tuple[str | None, list[str]]:
            if len(tokens) == 0:
                return None, tokens
            if tokens[0].lower() in [arm.value for arm in Arms]:
                return tokens[0], tokens[1:]
            return None, tokens

        action_sequence = []
        with open(plan_path) as f:
            for line in f.readlines():
                stripped_line = line.strip()
                if stripped_line.startswith("; cost") or stripped_line == "":
                    continue

                move_match = re.match(move_pattern, stripped_line)
                pick_match = re.match(pick_pattern, stripped_line)
                place_match = re.match(place_pattern, stripped_line)

                if move_match:
                    move_args = move_match.group(1).split()
                    robot_name, remaining_tokens = _extract_robot_name(move_args)
                    if len(remaining_tokens) == 0:
                        raise ParsingException(
                            "Move action missing target prim after optional robot name."
                        )
                    move_prim = remaining_tokens[0]
                    prim = task.get_prim(move_prim)
                    if prim is None:
                        raise ParsingException(
                            "No corresponding prim found for move action."
                            f"Prim in move action: {move_prim}"
                        )
                    action_sequence.append(
                        Action.from_prim(
                            ActionType.MOVE,
                            prim,
                            robot_name=robot_name,
                            target_prim_name=move_prim,
                        )
                    )
                elif pick_match:
                    pick_args = pick_match.group(1).split()
                    robot_name, _ = _extract_robot_name(pick_args)
                    action_sequence.append(
                        Action.from_prim(ActionType.PICK, robot_name=robot_name)
                    )
                elif place_match:
                    place_args = place_match.group(1).split()
                    robot_name, _ = _extract_robot_name(place_args)
                    action_sequence.append(
                        Action.from_prim(ActionType.PLACE, robot_name=robot_name)
                    )
                else:
                    raise ParsingException(
                        f"Line in plan does not match any known pattern.Line: '{line}'"
                    )
            return cls(action_sequence)
