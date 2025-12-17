from isaacsim.core.api.robots import Robot
from isaacsim.core.api.scenes import Scene
from isaacsim.core.api.tasks import BaseTask
from isaacsim.core.utils.stage import add_reference_to_stage

from standalone.constants import psm_dir


# TODO: We should probably build a standardized base task if we have multiple
class StandardTask(BaseTask):
    def __init__(self, name: str):
        super().__init__(name=name)
        self.task_achieved = False

    def set_up_scene(self, scene: Scene):
        super().set_up_scene(scene)
        scene.add_default_ground_plane()
        print("Added ground plane")

        asset_path = psm_dir / "psm_col.usd"
        print(asset_path)
        add_reference_to_stage(usd_path=str(asset_path), prim_path="/World/dVRK")

        self._dvrk: Robot = scene.add(
            Robot(prim_path="/World/dVRK", name="dvrk", position=(0.0, 0.0, 0.15))
        )
        print("Added robot")

    def get_observations(self):
        joint_positions = self._dvrk.get_joint_positions()
        # NOTE: Ideally these will be enums or dataclasses later on
        return {
            "joint_positions": joint_positions,
        }

    def pre_step(self, time_step_index, simulation_time):
        # NOTE: Here we can set our task achieved to True
        return super().pre_step(time_step_index, simulation_time)

    def post_reset(self):
        # NOTE: Here we should set the dvrk back to its initial position
        #       -> We should make sure that the gripper is in the open position
        return super().post_reset()
