import numpy as np
from isaacsim.core.api.scenes import Scene
from isaacsim.core.utils.stage import add_reference_to_stage

from src.base.prim import RigidOffsetPrim
from src.base.task import GoalConfig, SingleDvrkTask
from src.constants import props_dir
from src.plan.plan import Action, ActionType
from src.tasks.schemas import Problem


class Needle(RigidOffsetPrim):
    def __init__(
        self,
        prim_path,
        name="xform_prim",
        position=None,
        translation=None,
        orientation=None,
        scale=None,
        visible=None,
        reset_xform_properties=True,
    ):
        grasp_points = {
            "default": np.array([0.0, 0.0, -0.001]),
        }
        approach_points = {
            "default": np.array([0.0, 0.0, 0.03]),
        }
        departure_points = approach_points
        super().__init__(
            prim_path,
            name=name,
            position=position,
            translation=translation,
            orientation=orientation,
            scale=scale,
            visible=visible,
            reset_xform_properties=reset_xform_properties,
            contact_points=grasp_points,
            approach_points=approach_points,
            departure_points=departure_points,
        )
        self.set_active_side("default")


class Goal(RigidOffsetPrim):
    def __init__(
        self,
        prim_path,
        name="xform_prim",
        position=None,
        translation=None,
        orientation=None,
        scale=None,
        visible=None,
        reset_xform_properties=True,
    ):
        grasp_points = {"default": np.array([0.0, 0.0, 0.01])}
        approach_points = {
            "default": np.array([0.0, 0.0, 0.03]),
        }
        departure_points = approach_points
        super().__init__(
            prim_path,
            name=name,
            position=position,
            translation=translation,
            orientation=orientation,
            scale=scale,
            visible=visible,
            reset_xform_properties=reset_xform_properties,
            contact_points=grasp_points,
            approach_points=approach_points,
            departure_points=departure_points,
        )
        self.set_active_side("default")


class NeedleSorting(SingleDvrkTask):
    def __init__(self, name, problem):
        super().__init__(name, gripper_closed_position=np.array([-0.125, 0.125]))
        self.problem = problem
        self.last_move_target = None
        self._prims_by_name: dict[str, RigidOffsetPrim] = {}
        self.max_needles_per_color = 5
        self.max_needles = 15
        self.needle_area_start = np.array([-0.05, 0.025, 0.0])
        self.needle_area_end = np.array([0.05, 0.0525, 0.0])
        self.needle_grid_rows = 2
        self.dist_thresh = 0.0125

    def _sample_positions(
        self,
        count: int,
    ) -> list[np.ndarray]:
        cols = int(np.ceil(self.max_needles / self.needle_grid_rows))
        x_values = np.linspace(self.needle_area_start[0], self.needle_area_end[0], cols)
        y_values = np.linspace(
            self.needle_area_start[1],
            self.needle_area_end[1],
            self.needle_grid_rows,
        )
        grid_positions = [
            np.array([x, y, self.needle_area_start[2]], dtype=float)
            for y in y_values
            for x in x_values
        ]
        rng = np.random.default_rng(1000)
        selected_indices = rng.choice(len(grid_positions), size=count, replace=False)
        return [grid_positions[i] for i in selected_indices]

    def _add_goal(
        self,
        scene: Scene,
        goal_path: str,
        prim_path: str,
        name: str,
        position: np.ndarray,
        material,
        prim_name: str,
    ) -> Goal:
        add_reference_to_stage(usd_path=goal_path, prim_path=prim_path)
        goal: Goal = scene.add(
            Goal(
                prim_path=prim_path,
                name=name,
                position=position,
            )
        )
        goal.apply_visual_material(material)
        self._prims_by_name[prim_name] = goal
        return goal

    def _add_needles_per_color(
        self,
        scene: Scene,
        needle_path: str,
        base_name: str,
        prim_prefix: str,
        positions: list[np.ndarray],
        material,
    ) -> list[Needle]:
        needles: list[Needle] = []
        for index, position in enumerate(positions, start=1):
            indexed_name = f"{base_name}_{index}"
            prim_path = f"/World/{prim_prefix}_{index}"
            scene_name = f"{prim_prefix}_{index}"
            add_reference_to_stage(usd_path=needle_path, prim_path=prim_path)
            needle: Needle = scene.add(
                Needle(
                    prim_path=prim_path,
                    name=scene_name,
                    position=position,
                )
            )
            needle.apply_visual_material(material)
            needles.append(needle)
            self._prims_by_name[indexed_name] = needle
        return needles

    def set_up_scene(self, scene: Scene):
        super().set_up_scene(scene)

        needle_path = str(props_dir / "needle.usd")
        goal_path = str(props_dir / "goal.usd")

        if self.problem == Problem.NEEDLE_SORTING_1:
            red_count = 1
            green_count = 1
            blue_count = 1
        elif self.problem == Problem.NEEDLE_SORTING_2:
            red_count = 2
            green_count = 3
            blue_count = 5
        elif self.problem == Problem.NEEDLE_SORTING_3:
            red_count = 5
            green_count = 5
            blue_count = 5

        self._prims_by_name = {}
        total_needle_count = red_count + green_count + blue_count
        shared_positions = self._sample_positions(
            total_needle_count,
        )
        red_positions = shared_positions[:red_count]
        green_positions = shared_positions[red_count : red_count + green_count]
        blue_positions = shared_positions[red_count + green_count :]

        self.red_goal = self._add_goal(
            scene=scene,
            goal_path=goal_path,
            prim_path="/World/Red_Goal",
            name="Red_Goal",
            position=np.array([0.05, -0.05, 0.0]),
            material=self.red,
            prim_name="red_goal",
        )
        self.green_goal = self._add_goal(
            scene=scene,
            goal_path=goal_path,
            prim_path="/World/Green_Goal",
            name="Green_Goal",
            position=np.array([0.0, -0.05, 0.0]),
            material=self.green,
            prim_name="green_goal",
        )
        self.blue_goal = self._add_goal(
            scene=scene,
            goal_path=goal_path,
            prim_path="/World/Blue_Goal",
            name="Blue_Goal",
            position=np.array([-0.05, -0.05, 0.0]),
            material=self.blue,
            prim_name="blue_goal",
        )

        red_needles = self._add_needles_per_color(
            scene=scene,
            needle_path=needle_path,
            base_name="red_needle",
            prim_prefix="Red_Needle",
            positions=red_positions,
            material=self.red,
        )
        green_needles = self._add_needles_per_color(
            scene=scene,
            needle_path=needle_path,
            base_name="green_needle",
            prim_prefix="Green_Needle",
            positions=green_positions,
            material=self.green,
        )
        blue_needles = self._add_needles_per_color(
            scene=scene,
            needle_path=needle_path,
            base_name="blue_needle",
            prim_prefix="Blue_Needle",
            positions=blue_positions,
            material=self.blue,
        )

        self.goal = [
            *[
                GoalConfig(needle, self.red_goal, dist_thresh=self.dist_thresh)
                for needle in red_needles
            ],
            *[
                GoalConfig(needle, self.green_goal, dist_thresh=self.dist_thresh)
                for needle in green_needles
            ],
            *[
                GoalConfig(needle, self.blue_goal, dist_thresh=self.dist_thresh)
                for needle in blue_needles
            ],
        ]

    def post_reset(self):
        super().post_reset()
        self.last_move_target = None

    def prepare_action(self, action: Action, robot_name: str, ee_pos: np.ndarray):
        if action.action_type == ActionType.MOVE:
            prim = self.get_prim(action.target_prim_name)
            action.target_position, action.target_orientation = (
                prim.get_world_pose_with_offset()
            )
            if self.last_move_target is not None:
                # Add departure point of last prim
                action.waypoint_targets.append(
                    self.get_prim(self.last_move_target).get_world_departure_pose()
                )
            action.waypoint_targets.append(prim.get_world_approach_pose())
            self.last_move_target = action.target_prim_name
            action.waypoint_reached = False

    def get_prim(self, prim_name: str) -> RigidOffsetPrim | None:
        if prim_name is None:
            return None
        return self._prims_by_name.get(prim_name.lower())
