from argparse import ArgumentParser
from pathlib import Path
from src.tasks import Task
from src.tasks.schemas import Problem

# We need to execute the parsing step here as this will otherwise be ignored due to the sim-app loading
parser = ArgumentParser()
parser.add_argument("--task", choices=[t.value for t in Task], required=True)
parser.add_argument("--problem", choices=[t.value for t in Problem], required=True)
parser.add_argument("--pipeline_results", required=True)
parser.add_argument("--write-results", action="store_true")
parser.add_argument("--headless", action="store_true")
args = parser.parse_args()

from isaacsim.simulation_app import SimulationApp

simulation_app: SimulationApp = SimulationApp({"headless": args.headless})

# Suppress scientific notation for debug prints of np arrays
import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(suppress=True)

import polars as pl
from isaacsim.core.api import World

from src.base.task import Arms, DualDvrkTask, SingleDvrkTask
from src.constants import results_dir
from src.controller.lula_controller import LulaController
from src.controller.lula_planner import LulaMotionPlanner
from src.plan.plan import Plan, ParsingException
from src.result.result import Result, Results
from src.tasks import get_task


if __name__ == "__main__":
    task_name = args.task
    problem = args.problem
    results_file = Path(args.pipeline_results)
    assert results_file.exists(), "Ensure that the given results file exists"
    pipeline_results = pl.read_csv(results_file)
    write_results = args.write_results

    task = get_task(task_name, problem)

    ## Simulation setup
    world: World = World(physics_dt=1.0 / 240.0, rendering_dt=1.0 / 60.0)
    world.add_task(task)
    world.reset()

    is_dual_task = isinstance(task, DualDvrkTask)
    if is_dual_task:
        robots = {
            Arms.LEFT_ARM: task._left_dvrk,
            Arms.RIGHT_ARM: task._right_dvrk,
        }
        planners = {
            name: LulaMotionPlanner(robot=robot) for name, robot in robots.items()
        }
        controllers = {
            name: LulaController(
                name=f"psm_controller_{name}",
                robot_articulation=robot,
                gripper_positions=task.gripper_positions,
                planner=planners[name],
                robot_name=name,
            )
            for name, robot in robots.items()
        }
        default_robot_name = Arms.LEFT_ARM
    else:
        assert isinstance(task, SingleDvrkTask)
        robots = {"dvrk": task._dvrk}
        planners = {"dvrk": LulaMotionPlanner(robot=task._dvrk)}
        controllers = {
            "dvrk": LulaController(
                name="psm_controller",
                robot_articulation=task._dvrk,
                gripper_positions=task.gripper_positions,
                planner=planners["dvrk"],
                robot_name="dvrk",
            )
        }
        default_robot_name = "dvrk"

    world.reset()
    task.camera.initialize()
    task.rotate_camera(45)
    curr_results_dir = results_dir / results_file.stem
    curr_results_dir.mkdir(exist_ok=True)
    curr_results_image_dir = curr_results_dir / "images"
    curr_results_image_dir.mkdir(exist_ok=True)

    results = []
    for plan_path in pipeline_results["plan_file"]:
        if plan_path is None:
            results.append(
                Result(
                    None,
                    None,
                    False,
                )
            )
            continue
        plan_path = Path(plan_path)
        # Ensure stage is rendered before capturing start image
        for _ in range(50):
            world.step(render=True)
        image_start_path = curr_results_image_dir / f"{plan_path.stem}_start.png"
        plt.imsave(image_start_path, np.clip(task.camera.get_rgba() / 255.0, 0, 1))

        try:
            plan = Plan.from_pddl(task, plan_path)
        except ParsingException as e:
            print(e)
            results.append(
                Result(
                    image_start_path.name,
                    None,
                    False,
                )
            )
            continue
        for action in plan.action_sequence:
            robot_name = action.robot_name or default_robot_name
            ee_pos = planners[robot_name].get_end_effector_pose()[0]
            task.prepare_action(action, robot_name, ee_pos)
            if robot_name not in controllers:
                raise ValueError(
                    "Unknown robot requested by action. "
                    f"robot_name={robot_name}, available={list(controllers.keys())}"
                )

            controller = controllers[robot_name]
            planner = planners[robot_name]
            dvrk = robots[robot_name]

            print(f"Executing action: {action.action_type} on {robot_name}")
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

        image_end_path = curr_results_image_dir / f"{plan_path.stem}_end.png"
        plt.imsave(image_end_path, np.clip(task.camera.get_rgba() / 255.0, 0, 1))
        results.append(
            Result(
                image_start_path.name,
                image_end_path.name,
                task.goal_reached(),
            )
        )
        world.reset()

    print(f"{'=' * 60}")
    print("Simulation completed successfully!")
    print(f"{'=' * 60}")
    if write_results:
        Results(results).save_results(pipeline_results, results_file)
    simulation_app.close()
