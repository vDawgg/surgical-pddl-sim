from enum import StrEnum, auto
from typing import Self

import numpy as np
from isaacsim.core.prims import SingleXFormPrim


class ActionType(StrEnum):
    MOVE = auto()
    PICK = auto()
    PLACE = auto()


class Action:
    def __init__(self, action_type: ActionType, target: SingleXFormPrim | None = None):
        self.action_type = action_type
        target_position, target_pose = (
            target.get_world_pose() if target else (None, None)
        )
        self.target_position: np.ndarray | None = target_position
        self.target_orientation: np.ndarray | None = target_pose


class Plan:
    def __init__(self, action_sequence: list[Action]):
        self.action_sequence = action_sequence

    @classmethod
    def from_actions(cls, action_sequence: list[Action]):
        return cls(action_sequence)

    @classmethod
    def from_pddl(self):
        pass

    # TODO: It would still be nice to implement a cleaner way of doing this
    def add_via_points_to_plan(self):
        """
        This function adds via points 5cm above the target to make sure the gripper doesnt just push the object away
        before being able to grasp it.
        """
        extended_sequence = []
        for action in self.action_sequence:
            extended_sequence.append(action)
            if action.action_type == ActionType.MOVE:
                extended_sequence.append(
                    Action(
                        action_type=ActionType.MOVE,
                        target_position=action.target_position
                        + np.array([0.0, 0.0, 0.05]),
                        target_pose=action.target_orientation,
                    )
                )
        self.action_sequence = extended_sequence
