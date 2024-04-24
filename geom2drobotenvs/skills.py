"""Skills that might be useful in certain environments."""

from relational_structs.structs import ParameterizedOption, State, Object, OptionMemory, Action
from relational_structs.spaces import ObjectSequenceSpace
from geom2drobotenvs.object_types import CRVRobotType, RectangleType
from geom2drobotenvs.structs import SE2Pose, MultiBody2D
from geom2drobotenvs.utils import get_se2_pose, CRVRobotActionSpace, state_has_collision
from tomsutils.motion_planning import BiRRT
from tomsutils.utils import get_signed_angle_distance
from gym.spaces import Space
from typing import Dict, Sequence, Iterable, Optional
import numpy as np


def create_rectangle_vaccum_pick_option(action_space: Space) -> ParameterizedOption:
    """Use motion planning to get to a pre-pick pose. Then extend the arm and
    turn on the vacuum."""

    name = "RectangleVacuumPick"
    params_space = ObjectSequenceSpace([CRVRobotType, RectangleType])

    def policy(state: State, params: Sequence[Object], memory: OptionMemory) -> Action:
        import ipdb; ipdb.set_trace()


    def initiable(state: State, params: Sequence[Object], memory: OptionMemory) -> bool:
        # Set up motion planning in SE(2) for the robot.

        # TODO move this all into utils

        # Always use the same random seed.
        rng = np.random.default_rng(0)
        
        # Sensible defaults. May want to move these elsewhere in the future, but
        # ideally one would not need to tune hyperparameters to use this code.
        num_attempts = 10
        num_iters = 100
        smooth_amt = 50

        # Use the object positions in the state to create a rough room boundary.
        x_lb, x_ub, y_lb, y_ub = np.inf, -np.inf, np.inf, -np.inf
        for obj in state:
            pose = get_se2_pose(state, obj)
            x_lb = min(x_lb, pose.x)
            x_ub = max(x_ub, pose.x)
            y_lb = min(y_lb, pose.y)
            y_ub = max(y_ub, pose.y)

        # Create a static version of the state so that the geoms only need to be
        # instantiated once during motion planning (except for the robot).
        robot: Optional[Object] = None
        static_state = state.copy()
        for o in static_state:
            if o.is_instance(CRVRobotType):
                robot = o
                continue
            static_state.set(o, "static", 1.0)
        assert robot is not None
        static_object_body_cache: Dict[Object, MultiBody2D] = {}

        # Set up the RRT methods.
        def sample_fn(_: SE2Pose) -> SE2Pose:
            """Sample a robot pose."""
            x = rng.uniform(x_lb, x_ub)
            y = rng.uniform(y_lb, y_ub)
            theta = rng.uniform(-np.pi, np.pi)
            return SE2Pose(x, y, theta)
        
        def extend_fn(pt1: SE2Pose, pt2: SE2Pose) -> Iterable[SE2Pose]:
            """Interpolate between the two poses."""
            # Make sure that we obey the bounds on actions.
            dx = pt2.x - pt1.x
            dy = pt2.y - pt1.y
            dtheta = get_signed_angle_distance(pt2.theta, pt1.theta)
            assert isinstance(action_space, CRVRobotActionSpace)
            abs_x = action_space.high[0] if dx > 0 else action_space.low[0]
            abs_y = action_space.high[1] if dy > 0 else action_space.low[1]
            abs_theta = action_space.high[2] if dtheta > 0 else action_space.low[2]
            x_num_steps = dx / abs_x
            assert x_num_steps >= 0
            y_num_steps = dy / abs_y
            assert y_num_steps >= 0
            theta_num_steps = dtheta / abs_theta
            assert theta_num_steps >= 0
            num_steps = max(x_num_steps, y_num_steps, theta_num_steps)
            x_interp = np.linspace(pt1.x, pt2.x, num=num_steps, endpoint=True)
            y_interp = np.linspace(pt1.y, pt2.y, num=num_steps, endpoint=True)
            theta_interp = np.linspace(pt1.theta, pt2.theta, num=num_steps, endpoint=True)
            for x, y, theta in zip(x_interp, y_interp, theta_interp):
                yield SE2Pose(x, y, theta)

        def collision_fn(pt: SE2Pose) -> bool:
            """Check for collisions if the robot were at this pose."""
            # Update the static state with the robot's new hypothetical pose.
            static_state.set(robot, "x", pt.x)
            static_state.set(robot, "y", pt.y)
            static_state.set(robot, "theta", pt.theta)
            return state_has_collision(static_state, static_object_body_cache)
        
        def distance_fn(pt1: SE2Pose, pt2: SE2Pose) -> float:
            """Return a distance between the two points."""
            dx = pt2.x - pt1.x
            dy = pt2.y - pt1.y
            dtheta = get_signed_angle_distance(pt2.theta, pt1.theta)
            return np.sqrt(dx**2 + dy**2) + abs(dtheta)


        birrt = BiRRT(sample_fn, extend_fn, collision_fn, distance_fn, rng, num_attempts,
                    num_iters, smooth_amt)


        import ipdb; ipdb.set_trace()


    def terminal(state: State, params: Sequence[Object], memory: OptionMemory) -> bool:
        import ipdb; ipdb.set_trace()

    return ParameterizedOption(name, params_space, policy, initiable, terminal)
