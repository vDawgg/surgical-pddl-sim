import numpy as np
import isaacsim.core.utils.numpy.rotations as rot_utils
from isaacsim.core.prims import SingleRigidPrim


class RigidOffsetPrim(SingleRigidPrim):
    """
    Wrapper around rigid prims to allow for the definition of an offset to the world position.
    This ensures proper placement at grippable positions of the object.
    Optional approach points can be defined in the prim local frame for staged motion.
    Use this for physics-enabled objects that move during simulation (e.g., picked up by gripper).
    """

    def __init__(
        self,
        prim_path,
        name="rigid_prim",
        position=None,
        translation=None,
        orientation=None,
        scale=None,
        visible=None,
        reset_xform_properties=True,
        mass=None,
        density=None,
        linear_velocity=None,
        angular_velocity=None,
        offset: np.ndarray | None = None,
        contact_points: dict[str, np.ndarray] | None = None,
        approach_points: dict[str, np.ndarray] | None = None,
        departure_points: dict[str, np.ndarray] | None = None,
        offset_in_local_frame: bool = True,
    ):
        super().__init__(
            prim_path,
            name,
            position,
            translation,
            orientation,
            scale,
            visible,
            reset_xform_properties,
            mass,
            density,
            linear_velocity,
            angular_velocity,
        )
        self.offset = offset
        self.grasp_points_local = contact_points
        self.approach_points_local = approach_points
        self.departure_points_local = departure_points
        self.approach_offset = np.zeros(3)
        self.departure_offset = np.zeros(3)
        self.offset_in_local_frame = offset_in_local_frame

    def _to_world_vector(self, local_vector: np.ndarray, orientation: np.ndarray):
        if not self.offset_in_local_frame:
            return local_vector
        rotation = rot_utils.quats_to_rot_matrices(orientation)
        return rotation @ local_vector

    def get_world_approach_pose(self):
        position, orientation = self.get_world_pose()
        return position + self._to_world_vector(
            self.approach_offset, orientation
        ), orientation

    def get_world_departure_pose(self):
        position, orientation = self.get_world_pose()
        return position + self._to_world_vector(
            self.departure_offset, orientation
        ), orientation

    def get_world_pose_with_offset(self):
        position, orientation = self.get_world_pose()
        return position + self._to_world_vector(self.offset, orientation), orientation

    def set_active_side(self, side: str):
        self.active_grasp_name = side
        self.active_approach_name = side
        self.offset = self.grasp_points_local[side]
        if self.approach_points_local:
            self.approach_offset = self.approach_points_local[side]
        if self.departure_points_local:
            self.departure_offset = self.departure_points_local[side]
