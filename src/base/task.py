import numpy as np
import isaacsim.core.utils.numpy.rotations as rot_utils
from isaacsim.core.api.robots import Robot
from isaacsim.core.api.scenes import Scene
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.api.tasks import BaseTask
from isaacsim.core.experimental.objects import DistantLight
from isaacsim.sensors.camera import Camera

from src.base.prim import RigidOffsetPrim
from src.constants import psm_dir


class ActuatorConfig:
    def __init__(
        self,
        joints: list[str],
        stiffness: float,
        damping: float,
        effort_limit: float,
        velocity_limit: float,
    ):
        self.joints = joints
        num_joints = len(self.joints)
        self.stiffness = np.repeat(stiffness, num_joints)
        self.damping = np.repeat(damping, num_joints)
        self.effort_limit = np.repeat(effort_limit, num_joints)
        self.velocity_limit = np.repeat(velocity_limit, num_joints)


class GripperPositions:
    def __init__(self, close: np.ndarray, open: np.ndarray | None = None):
        self.open = open or np.array([-0.5, 0.5])
        self.close = close


class GoalConfig:
    """
    Specifies the goal configuration for one object.
    """

    def __init__(self, prim: RigidOffsetPrim, goal_prim: RigidOffsetPrim):
        self.prim = prim
        self.goal_prim = goal_prim

    def is_at_goal(self):
        x_y_pos = self.prim.get_world_position_no_offset()
        target_pos = self.goal_prim.get_world_position_no_offset()
        # TODO: Tune this difference where appropriate
        return np.sqrt(np.sum(x_y_pos - target_pos) ** 2) <= 0.005


class DvrkTask(BaseTask):
    def __init__(self, name: str, gripper_closed_position: np.ndarray):
        super().__init__(name=name)
        self._initial_joint_pos: dict[str, float] = {
            "psm_yaw_joint": 0.01,
            "psm_pitch_end_joint": 0.01,
            "psm_main_insertion_joint": 0.07,
            "psm_tool_roll_joint": 0.01,
            "psm_tool_pitch_joint": 0.01,
            "psm_tool_yaw_joint": 0.01,
            "psm_tool_gripper1_joint": -0.5,
            "psm_tool_gripper2_joint": 0.5,
        }
        self._psm_actuator_cfg = ActuatorConfig(
            joints=[
                "psm_yaw_joint",
                "psm_pitch_end_joint",
                "psm_main_insertion_joint",
                "psm_tool_roll_joint",
                "psm_tool_pitch_joint",
                "psm_tool_yaw_joint",
            ],
            stiffness=800.0,
            damping=40.0,
            effort_limit=12.0,
            velocity_limit=1.0,
        )
        self._gripper_actuator_cfg = ActuatorConfig(
            joints=["psm_tool_gripper1_joint", "psm_tool_gripper2_joint"],
            stiffness=500.0,
            damping=0.1,
            effort_limit=0.1,
            velocity_limit=0.2,
        )
        # This needs to be set on a per-task basis as the ideal closed position will vary depending on the object
        self.gripper_positions = GripperPositions(gripper_closed_position)
        self.action_sequence = None
        self.goal: list[GoalConfig] | None = None

    def set_up_scene(self, scene: Scene) -> None:
        super().set_up_scene(scene)
        self.camera: Camera = scene.add(
            Camera(
                prim_path="/World/camera",
                position=np.array([0.0, 1.0, 0.7]),
                frequency=20,
                resolution=(1024, 1024),
                orientation=rot_utils.euler_angles_to_quats(
                    np.array([0, 35, -90]), degrees=True
                ),
            )
        )
        self.camera.set_focal_length(8)

        scene.add_default_ground_plane()

        # Add a light source to illuminate the scene
        self.light = DistantLight(
            "/World/defaultLight",
            orientations=rot_utils.euler_angles_to_quats(
                np.array([-45, 45, 0]), degrees=True
            ),
            reset_xform_op_properties=True,
        )
        self.light.set_intensities([3000])

        robot_path = psm_dir / "psm_col.usd"
        robot_primt_path = "/World/dVRK"
        add_reference_to_stage(usd_path=str(robot_path), prim_path=robot_primt_path)

        self._dvrk: Robot = scene.add(
            Robot(
                prim_path=robot_primt_path,
                name="dvrk",
                position=np.array([0.0, 0.0, 0.16]),
            )
        )

    def post_reset(self):
        super().post_reset()
        self._dvrk.disable_gravity()

        self._dvrk.set_solver_position_iteration_count(4)
        self._dvrk.set_solver_velocity_iteration_count(0)
        self._dvrk.set_enabled_self_collisions(False)

        art = self._dvrk._articulation_view
        psm_dof_indices = np.array(
            [art.get_dof_index(joint) for joint in self._psm_actuator_cfg.joints],
            dtype=np.int32,
        )
        gripper_dof_indices = np.array(
            [art.get_dof_index(joint) for joint in self._gripper_actuator_cfg.joints],
            dtype=np.int32,
        )

        art.set_max_joint_velocities(
            self._psm_actuator_cfg.velocity_limit,
            joint_indices=psm_dof_indices,
        )
        art.set_max_joint_velocities(
            self._gripper_actuator_cfg.velocity_limit,
            joint_indices=gripper_dof_indices,
        )

        art.set_max_efforts(
            self._psm_actuator_cfg.effort_limit,
            joint_indices=psm_dof_indices,
        )
        art.set_max_efforts(
            self._gripper_actuator_cfg.effort_limit,
            joint_indices=gripper_dof_indices,
        )

        art.set_gains(
            self._psm_actuator_cfg.stiffness,
            self._psm_actuator_cfg.damping,
            joint_indices=psm_dof_indices,
        )
        art.set_gains(
            self._gripper_actuator_cfg.stiffness,
            self._gripper_actuator_cfg.damping,
            joint_indices=gripper_dof_indices,
        )

        self._dvrk.set_joints_default_state(
            positions=list(self._initial_joint_pos.values())
        )
        self._dvrk.post_reset()

    def goal_reached(self):
        return all(goal_config.is_at_goal() for goal_config in self.goal)
