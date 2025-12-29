from enum import StrEnum, auto

import numpy as np
from isaacsim.core.api.robots import Robot
from isaacsim.core.utils.types import ArticulationAction

from src.controller.lula_planner import LulaMotionPlanner
from src.base.dvrk_task import GripperPositions


class Action(StrEnum):
    MOVE = auto()
    PICK = auto()
    PLACE = auto()


class LulaController:
    def __init__(
        self,
        name: str,
        robot_articulation: Robot,
        gripper_positions: GripperPositions,
        planner: LulaMotionPlanner,
    ):
        self._name = name
        self._robot_articulation = robot_articulation
        self._dof_names = robot_articulation.dof_names
        self._planner = planner

        self._gripper_indices = [
            self._robot_articulation.get_dof_index("psm_tool_gripper1_joint"),
            self._robot_articulation.get_dof_index("psm_tool_gripper2_joint"),
        ]
        self._gripper_positions = gripper_positions

    def forward(
        self,
        action_type: Action,
        target_position: np.ndarray | None = None,
        target_orientation: np.ndarray | None = None,
    ) -> ArticulationAction:
        joint_positions = self._robot_articulation.get_joint_positions()

        if action_type == Action.MOVE:
            assert target_position is not None
            self._planner.set_end_effector_target(target_position, target_orientation)
            action = self._planner.get_next_articulation_action()
            return action

        # TODO: Make sure the pick and place actions are carried out with the rest of the robot staying
        #       in the same configuration. Currently the joints just give up when these actions are executed
        #       -> This currently only works because gravity is disabled on the robot
        elif action_type == Action.PICK:
            new_joint_positions = joint_positions.copy()
            new_joint_positions[self._gripper_indices] = self._gripper_positions.close
            return ArticulationAction(joint_positions=new_joint_positions)

        elif action_type == Action.PLACE:
            new_joint_positions = joint_positions.copy()
            new_joint_positions[self._gripper_indices] = self._gripper_positions.open
            return ArticulationAction(joint_positions=new_joint_positions)

        return ArticulationAction(joint_positions=joint_positions)
