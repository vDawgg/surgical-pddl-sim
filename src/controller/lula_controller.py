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
        robot_name: str | None = None,
    ):
        self._name = name
        self._robot_articulation = robot_articulation
        self._dof_names = robot_articulation.dof_names
        self._planner = planner
        self._robot_name = robot_name
        self._gripper_indices = [
            self._robot_articulation.get_dof_index("psm_tool_gripper1_joint"),
            self._robot_articulation.get_dof_index("psm_tool_gripper2_joint"),
        ]
        self._non_gripper_indices = [
            self._robot_articulation.get_dof_index(name)
            for name in self._dof_names
            if "gripper" not in name
        ]
        self._gripper_positions = gripper_positions
        self._position_threshold = 0.0005
        self._waypoint_threshold = 0.005
        self._gripper_threshold = 0.0005

    def forward(
        self,
        action: Action,
    ) -> ArticulationAction:
        if action.robot_name is not None and self._robot_name is not None:
            if action.robot_name != self._robot_name:
                raise ValueError(
                    f"Action for robot '{action.robot_name}' cannot be applied by '{self._robot_name}'"
                )
        joint_positions = self._robot_articulation.get_joint_positions()
        if action.action_type == ActionType.MOVE:
            assert action.target_position is not None
            current_ee_pos, _ = self._planner.get_end_effector_pose()
            if action.waypoint_targets:
                waypoint_threshold = (
                    action.waypoint_threshold or self._waypoint_threshold
                )
                while action.waypoint_index < len(action.waypoint_targets):
                    waypoint_position, waypoint_orientation = action.waypoint_targets[
                        action.waypoint_index
                    ]
                    distance_to_waypoint = np.linalg.norm(
                        current_ee_pos - waypoint_position
                    )
                    if distance_to_waypoint <= waypoint_threshold:
                        action.waypoint_index += 1
                        print(
                            "Waypoint "
                            f"{action.waypoint_index}/{len(action.waypoint_targets)} "
                            f"at {waypoint_position} "
                            f"reached for {action.target_prim_name or 'move'} "
                            f"at distance {distance_to_waypoint:.4f}"
                        )
                        continue
                    self._planner.set_end_effector_target(
                        waypoint_position, waypoint_orientation
                    )
                    return self._planner.get_next_articulation_action()
            # Target the final position
            self._planner.set_end_effector_target(
                action.target_position, action.target_orientation
            )
            return self._planner.get_next_articulation_action()
        else:
            if action.target_joint_positions is None:
                action.target_joint_positions = np.array(
                    joint_positions[self._non_gripper_indices], copy=True
                )
            new_joint_positions = joint_positions.copy()
            new_joint_positions[self._non_gripper_indices] = (
                action.target_joint_positions
            )
            if action.action_type == ActionType.PICK:
                new_joint_positions[self._gripper_indices] = (
                    self._gripper_positions.close
                )
            elif action.action_type == ActionType.PLACE:
                new_joint_positions[self._gripper_indices] = (
                    self._gripper_positions.open
                )
            return ArticulationAction(joint_positions=new_joint_positions)

    def completed(self, action: Action) -> bool:
        if action.robot_name is not None and self._robot_name is not None:
            if action.robot_name != self._robot_name:
                return False
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
            is_complete = np.allclose(
                current_gripper_pos, target, atol=self._gripper_threshold
            )
            if is_complete:
                action.target_joint_positions = None
            return is_complete
