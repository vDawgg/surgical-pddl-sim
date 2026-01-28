from isaacsim.core.api.robots import Robot
from isaacsim.core.utils.types import ArticulationAction

import numpy as np

from src.base.task import GripperPositions
from src.controller.lula_planner import LulaMotionPlanner
from src.plan.plan import Action, ActionType


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

        # TODO: Tune these
        self._position_threshold = 0.0005
        self._gripper_threshold = 0.0005

    def forward(
        self,
        action: Action,
    ) -> ArticulationAction:
        joint_positions = self._robot_articulation.get_joint_positions()

        if action.action_type == ActionType.MOVE:
            assert action.target_position is not None
            self._planner.set_end_effector_target(
                action.target_position, action.target_orientation
            )
            action = self._planner.get_next_articulation_action()
            return action

        elif action.action_type == ActionType.PICK:
            new_joint_positions = joint_positions.copy()
            new_joint_positions[self._gripper_indices] = self._gripper_positions.close
            return ArticulationAction(joint_positions=new_joint_positions)

        elif action.action_type == ActionType.PLACE:
            new_joint_positions = joint_positions.copy()
            new_joint_positions[self._gripper_indices] = self._gripper_positions.open
            return ArticulationAction(joint_positions=new_joint_positions)

        return ArticulationAction(joint_positions=joint_positions)

    def completed(self, action: Action) -> bool:
        if action.action_type == ActionType.MOVE:
            # Position-based check for MOVE
            current_ee_pos, _ = self._planner.get_end_effector_pose()
            distance = np.sqrt(np.sum(current_ee_pos - action.target_position) ** 2)
            return distance < self._position_threshold
        elif action.action_type in (ActionType.PICK, ActionType.PLACE):
            # Joint-based check for gripper
            current_gripper_pos = self._robot_articulation.get_joint_positions()[
                self._gripper_indices
            ]
            target = (
                self._gripper_positions.close
                if action.action_type == ActionType.PICK
                else self._gripper_positions.open
            )
            return np.allclose(
                current_gripper_pos, target, atol=self._gripper_threshold
            )
