from isaacsim.simulation_app import SimulationApp

simulation_app: SimulationApp = SimulationApp({"headless": False})

from isaacsim.core.api import World
from isaacsim.core.api.robots import Robot
from isaacsim.core.prims import RigidPrim

from src.controller.lula_controller import LulaController
from src.controller.physics_step_wrapper import PhysicsStepWrapper
from src.controller.lula_planner import LulaMotionPlanner
from src.tasks.needle_transfer import NeedleTransfer
from src.plan.plan import Action, ActionType, Plan


if __name__ == "__main__":
    ## Simulation setup
    world: World = World()
    task = NeedleTransfer(name="needle_transfer_task")

    world.add_task(task)

    world.reset()

    dvrk: Robot = world.scene.get_object("dvrk")
    planner = LulaMotionPlanner(robot=dvrk)
    controller = LulaController(
        name="psm_controller",
        robot_articulation=dvrk,
        gripper_positions=task.gripper_positions,
        planner=planner,
    )
    wrapper = PhysicsStepWrapper(controller=controller, robot=dvrk, world=world)

    obs = task.get_observations()

    plan = Plan.from_actions(
        [
            Action(ActionType.MOVE, task.needle),
            Action(ActionType.PICK),
            Action(ActionType.MOVE, task.goal),
            Action(ActionType.PLACE),
        ]
    )
    world.reset()

    psm_tool_tip = RigidPrim("/World/dVRK/psm_tool_tip_link", "psm_tool_tip_view")
    psm_tool_tip_pos = psm_tool_tip.get_world_poses()[0][0]

    # TODO: This should be a command-line argument
    num_trials = 2
    for trial in range(num_trials):
        print(f"{'=' * 60}")
        print(f"TRIAL {trial + 1}/{num_trials}")
        print(f"{'=' * 60}")

        for action in plan.action_sequence:
            print(f"Executing action: {action.action_type}")
            if action.target_position is not None:
                print("Target position:", action.target_position)
                print("Target orientation:", action.target_orientation)

            max_steps = 300
            # TODO: We need to find general stop-criteria that we can communicate via the task and/or for each step
            #       to ensure a cleaner execution-flow
            #       -> Ideally this will be added to the controller
            for step in range(max_steps):
                wrapper.step(action)
        world.reset()

    print(f"{'=' * 60}")
    print("Simulation completed successfully!")
    print(f"{'=' * 60}")
    simulation_app.close()
