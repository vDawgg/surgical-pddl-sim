import re
from enum import StrEnum, auto

import numpy as np
from isaacsim.core.prims import SingleXFormPrim

from src.tasks.ring_and_peg import RingAndPeg


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
    ):
        self.action_type = action_type
        self.target_position = target_position
        self.target_orientation = target_orientation

    @classmethod
    def from_position_and_orientation(
        cls, target_position: np.ndarray, target_orientation: np.ndarray | None = None
    ):
        """Creates a move action from target position and orientation"""
        return cls(
            action_type=ActionType.MOVE,
            target_position=target_position,
            target_orientation=target_orientation,
        )

    @classmethod
    def from_prim(cls, action_type: ActionType, prim: SingleXFormPrim | None = None):
        """
        Creates an action of any type with the specified target position and orientation of the given prim.
        The target can be omitted for the pick and place action, as these actions only control the gripper.
        """
        if prim is None:
            return cls(action_type=action_type)
        else:
            target_position, target_orientation = prim.get_world_pose()
            return cls(
                action_type=action_type,
                target_position=target_position,
                target_orientation=target_orientation,
            )


class Plan:
    def __init__(self, action_sequence: list[Action]):
        self.action_sequence = action_sequence

    @classmethod
    def from_actions(cls, action_sequence: list[Action]):
        return cls(action_sequence)

    @classmethod
    def from_pddl(cls, task: RingAndPeg, plan_path: str):
        move_pattern = r"\(\s*move\s+(\w+)[^)]*\)"
        pick_pattern = r"\(\s*pick\b[^)]*\)"
        place_pattern = r"\(\s*place\b[^)]*\)"
        action_sequence = []
        with open(plan_path) as f:
            for line in f.readlines():
                if re.match(move_pattern, line):
                    move_prim = re.match(move_pattern, line).group(1)
                    prim = task.get_prim(move_prim)
                    if prim is None:
                        raise ParsingException(
                            "No corresponding prim found for move action."
                            f"Prim in move action: {move_prim}"
                        )
                    action_sequence.append(Action.from_prim(ActionType.MOVE, prim))
                elif re.match(pick_pattern, line):
                    action_sequence.append(Action.from_prim(ActionType.PICK))
                elif re.match(place_pattern, line):
                    action_sequence.append(Action.from_prim(ActionType.PLACE))
                elif line.startswith("; cost"):
                    continue
                else:
                    raise ParsingException(
                        f"Line in plan does not match any known pattern.Line: '{line}'"
                    )
            return cls(action_sequence)

    def add_via_points_to_plan(self):
        """
        This function adds via points 5cm above the target to make sure the gripper doesnt just push the object away
        before being able to grasp it. It additionally, ensures that the robot can safely navigate with object in hand
        around obstacles.
        """
        extended_sequence = []
        last_move = None
        for action in self.action_sequence:
            if action.action_type == ActionType.MOVE:
                last_move = action
                # FIXME: apparently there are plans where no target was given. These will have to be rejected right away
                extended_sequence.append(
                    Action.from_position_and_orientation(
                        target_position=action.target_position
                        + np.array([0.0, 0.0, 0.03]),
                        target_orientation=action.target_orientation,
                    )
                )
            extended_sequence.append(action)
            if (
                action.action_type == ActionType.PICK
                or action.action_type == ActionType.PLACE
            ):
                # We have not moved anywhere, cannot add a via point above pick/place position
                if last_move is None:
                    continue
                # While the via points will not always make sense here, this is a best effort approach
                # of ensuring the EE is always above the pick/place positions before the next move action
                # to avoid colliding with the ground / obstacles
                extended_sequence.append(
                    Action.from_position_and_orientation(
                        target_position=last_move.target_position
                        + np.array([0.0, 0.0, 0.03]),
                        target_orientation=last_move.target_orientation,
                    )
                )
        self.action_sequence = extended_sequence
