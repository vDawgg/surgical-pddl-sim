import numpy as np
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.prims import SingleRigidPrim, SingleXFormPrim

from src.base.dvrk_task import DvrkTask
from src.constants import props_dir


class NeedleTransfer(DvrkTask):
    def __init__(self, name):
        super().__init__(name, gripper_closed_position=np.array([-0.05, 0.05]))

    def set_up_scene(self, scene):
        super().set_up_scene(scene)
        needle_path = props_dir / "needle.usd"
        add_reference_to_stage(usd_path=str(needle_path), prim_path="/World/Needle")
        self.needle = scene.add(
            SingleRigidPrim(
                prim_path="/World/Needle",
                name="Needle",
                position=np.array([0.05, 0.05, 0.0]),
                scale=np.array([0.3, 0.3, 0.3]),
            )
        )
        self.goal = scene.add(
            SingleXFormPrim(
                prim_path="/World/Goal",
                name="Goal",
                position=np.array([0.1, 0.1, 0.1]),
                orientation=None,  # TODO: Change this to something more interesting down the line
            )
        )

    def get_observations(self) -> dict[str, np.ndarray]:
        joint_positions = self._dvrk.get_joint_positions()
        needle_position = self.needle.get_world_pose()
        # TODO: Ideally these should be a dataclass
        #       -> For this to be a dataclass, we will need to ensure that the positioning of the objects can be transferred
        #          to multiple scenes. Or we do not care and define one dataclass per scene.
        #       -> This would also mean that we will have to keep the execution loop separate from main so we can distinguish
        #          the execution there
        return {
            "joint_positions": joint_positions,
            "needle": needle_position,
        }
