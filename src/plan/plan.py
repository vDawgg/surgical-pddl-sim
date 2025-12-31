from enum import StrEnum, auto
from typing import Self

import numpy as np
from isaacsim.core.prims import SingleXFormPrim


class ActionType(StrEnum):
    MOVE = auto()
    PICK = auto()
    PLACE = auto()


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
    def from_pddl(self):
        pass

    # TODO: It would still be nice to implement a cleaner way of doing this
    def add_via_points_to_plan(self):
        """
        This function adds via points 5cm above the target to make sure the gripper doesnt just push the object away
        before being able to grasp it. It additionally, ensures that the robot can safely navigate with object in hand
        around obstacles.
        """
        extended_sequence = []
        for i, action in enumerate(self.action_sequence):
            if action.action_type == ActionType.MOVE:
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
                assert self.action_sequence[i - 1].action_type == ActionType.MOVE
                extended_sequence.append(
                    Action.from_position_and_orientation(
                        target_position=self.action_sequence[i - 1].target_position
                        + np.array([0.0, 0.0, 0.03]),
                        target_orientation=self.action_sequence[
                            i - 1
                        ].target_orientation,
                    )
                )
        self.action_sequence = extended_sequence
