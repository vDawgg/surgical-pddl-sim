from argparse import ArgumentParser
from pathlib import Path
from src.tasks import Task

# We need to execute the parsing step here as this will otherwise be ignored due to the sim-app loading
parser = ArgumentParser()
parser.add_argument("--task", choices=[t.value for t in Task], required=True)
parser.add_argument("--plan_path_dir", required=True)
parser.add_argument("--headless", action="store_true")
args = parser.parse_args()

from isaacsim.simulation_app import SimulationApp

simulation_app: SimulationApp = SimulationApp({"headless": args.headless})

# Suppress scientific notation for debug prints of np arrays
import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(suppress=True)

from isaacsim.core.api import World
from isaacsim.core.prims import RigidPrim

from src.controller.lula_controller import LulaController
from src.controller.lula_planner import LulaMotionPlanner
from src.plan.plan import Plan, ParsingException
from src.result.result import Result, Results
from src.tasks import get_task


# TODO: Add a camera for proper view of the workspace and capture image at the start as VLM input
#       and at the end for verfication (for now)
if __name__ == "__main__":
    task_name = args.task
    plan_path_dir = Path(args.plan_path_dir)
    assert plan_path_dir.exists(), "Ensure that the given path exists"

    task = get_task(task_name)

    ## Simulation setup
    world: World = World()
    world.add_task(task)
    world.reset()

    dvrk = task._dvrk
    planner = LulaMotionPlanner(robot=dvrk)
    controller = LulaController(
        name="psm_controller",
        robot_articulation=dvrk,
        gripper_positions=task.gripper_positions,
        planner=planner,
    )

    world.reset()
    task.camera.initialize()
    plan_path_image_dir = plan_path_dir / "images"
    plan_path_image_dir.mkdir(exist_ok=True)

    psm_tool_tip = RigidPrim("/World/dVRK/psm_tool_tip_link", "psm_tool_tip_view")
    psm_tool_tip_pos = psm_tool_tip.get_world_poses()[0][0]

    plans_in_dir = [
        plan_path
        for plan_path in plan_path_dir.iterdir()
        if plan_path.suffix == ".plan"
    ]

    results = []
    for plan_path in plans_in_dir:
        # Ensure stage is rendered before capturing start image
        for _ in range(50):
            world.step(render=True)
        image_start_path = plan_path_image_dir / f"{plan_path.stem}_start.png"
        plt.imsave(image_start_path, np.clip(task.camera.get_rgba() / 255.0, 0, 1))

        try:
            plan = Plan.from_pddl(task, plan_path)
        except ParsingException:
            results.append(
                Result(
                    plan_path.name,
                    image_start_path.name,
                    None,
                    False,
                )
            )
            continue
        plan.add_via_points_to_plan()
        for action in plan.action_sequence:
            print(f"Executing action: {action.action_type}")
            if action.target_position is not None:
                print("Target position:", action.target_position)
                print("Target orientation:", action.target_orientation)
            max_steps = 300
            grace_period = 300
            for step in range(max_steps):
                if controller.completed(action) and step > grace_period:
                    break
                elif controller.completed(action) and grace_period == max_steps:
                    grace_period = step + 20
                robot_action = controller.forward(action)
                dvrk.apply_action(robot_action)
                world.step(render=True)
            print("Current EE position:", planner.get_end_effector_pose()[0])
            print(f"{'-' * 60}")
        print(f"{'=' * 60}")
        print("Goal configuration reached:", task.goal_reached())
        print(f"{'=' * 60}")

        image_end_path = plan_path_image_dir / f"{plan_path.stem}_end.png"
        plt.imsave(image_end_path, np.clip(task.camera.get_rgba() / 255.0, 0, 1))
        results.append(
            Result(
                plan_path.name,
                image_start_path.name,
                image_end_path.name,
                task.goal_reached(),
            )
        )
        world.reset()

    print(f"{'=' * 60}")
    print("Simulation completed successfully!")
    print(f"{'=' * 60}")
    Results(results).save_results(plan_path_dir)
    simulation_app.close()
