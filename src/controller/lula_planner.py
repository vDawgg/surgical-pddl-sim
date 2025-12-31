import numpy as np
from isaacsim.robot_motion.motion_generation import RmpFlow, ArticulationMotionPolicy
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.api.robots import Robot

from src.constants import psm_dir


class LulaMotionPlanner:
    def __init__(
        self,
        robot: Robot,
    ):
        self._robot = robot
        self._rmpflow = RmpFlow(
            robot_description_path=str(psm_dir / "robot_description.yaml"),
            urdf_path=str(psm_dir / "psm_col.urdf"),
            rmpflow_config_path=str(psm_dir / "rmpflow_config.yaml"),
            end_effector_frame_name="psm_tool_tip_link",
            maximum_substep_size=0.0033,
        )
        self._rmpflow.set_robot_base_pose(*robot.get_world_pose())
        # TODO: This seems like it will be useful for us
        # self._rmpflow.add_obstacle()
        # self._rmpflow.set_ignore_state_updates(True)
        # self._rmpflow.visualize_collision_spheres()
        self._articulation_policy = ArticulationMotionPolicy(robot, self._rmpflow)

    def set_end_effector_target(
        self, target_position: np.ndarray, target_orientation: np.ndarray | None = None
    ):
        self._rmpflow.set_end_effector_target(
            target_position=target_position, target_orientation=target_orientation
        )

    def get_end_effector_pose(self):
        active_joint_indices = [
            self._robot.get_dof_index(joint)
            for joint in self._rmpflow.get_active_joints()
        ]
        return self._rmpflow.get_end_effector_pose(
            self._robot.get_joint_positions(active_joint_indices)
        )

    def get_next_articulation_action(self) -> ArticulationAction:
        return self._articulation_policy.get_next_articulation_action()

    def reset(self):
        self._rmpflow.reset()
