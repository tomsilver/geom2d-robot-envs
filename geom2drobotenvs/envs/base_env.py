"""Base class for Geom2D robot environments."""

import abc
from typing import ClassVar, Dict, Optional, Tuple

import gym
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from relational_structs.spaces import ObjectCentricStateSpace
from relational_structs.structs import Array, Object, State
from tomsutils.utils import fig2data, wrap_angle

from geom2drobotenvs.object_types import CRVRobotType, RectangleType
from geom2drobotenvs.structs import MultiBody2D, SE2Pose
from geom2drobotenvs.utils import (
    CRVRobotActionSpace,
    get_suctioned_objects,
    object_to_multibody2d,
    state_has_collision,
)


class Geom2DRobotEnv(gym.Env):
    """Base class for Geom2D robot environments.

    NOTE: this implementation currently assumes we are using CRVRobotType.
    If we add other robot types in the future, we will need to refactor a bit.
    """

    # Only RGB rendering is implemented.
    render_mode = "rgb_array"
    metadata = {"render_modes": [render_mode]}
    _render_dpi: int = 150

    # The world is oriented like a standard X/Y coordinate frame.
    # Subclasses may override.
    _world_min_x: ClassVar[float] = 0.0
    _world_max_x: ClassVar[float] = 10.0
    _world_min_y: ClassVar[float] = 0.0
    _world_max_y: ClassVar[float] = 10.0

    # Arm length for the robot.
    _max_robot_arm_joint: ClassVar[float] = 3.0

    def __init__(self) -> None:
        self._types = {RectangleType, CRVRobotType}
        self.observation_space = ObjectCentricStateSpace(self._types)
        self.action_space = CRVRobotActionSpace()

        # Initialized by reset().
        self._current_state: Optional[State] = None
        self._static_object_body_cache: Dict[Object, MultiBody2D] = {}

        super().__init__()

    @abc.abstractmethod
    def _sample_initial_state(self) -> State:
        """Use self.np_random to sample an initial state."""

    def _get_obs(self) -> State:
        assert self._current_state is not None, "Need to call reset()"
        return self._current_state.copy()

    def _get_info(self) -> Dict:
        return {}  # no extra info provided right now

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[State, Dict]:
        super().reset(seed=seed)

        # Need to flush the cache in case static objects move.
        self._static_object_body_cache = {}

        # For testing purposes only, the options may specify an initial scene.
        if options is not None and "init_state" in options:
            self._current_state = options["init_state"].copy()

        # Otherwise, set up the initial scene here.
        else:
            self._current_state = self._sample_initial_state()

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action: Array) -> Tuple[State, float, bool, bool, Dict]:
        assert self.action_space.contains(action)
        dx, dy, dtheta, darm, vac = action
        assert self._current_state is not None, "Need to call reset()"
        state = self._current_state.copy()
        robots = [o for o in state if o.is_instance(CRVRobotType)]
        assert len(robots) == 1, "Multi-robot not yet supported"
        robot = robots[0]

        # NOTE: xy clipping is not needed because world boundaries are assumed
        # handled by collision detection with walls.
        new_x = state.get(robot, "x") + dx
        new_y = state.get(robot, "y") + dy
        new_theta = wrap_angle(state.get(robot, "theta") + dtheta)
        min_arm = state.get(robot, "base_radius")
        max_arm = self._max_robot_arm_joint
        new_arm = np.clip(state.get(robot, "arm_joint") + darm, min_arm, max_arm)
        state.set(robot, "x", new_x)
        state.set(robot, "y", new_y)
        state.set(robot, "arm_joint", new_arm)
        state.set(robot, "theta", new_theta)
        state.set(robot, "vacuum", vac)

        # Update the state of any objects that are currently suctioned.
        world_to_robot = SE2Pose(new_x, new_y, new_theta)
        for obj, robot_to_obj in get_suctioned_objects(self._current_state, robot):
            world_to_obj = world_to_robot * robot_to_obj
            state.set(obj, "x", world_to_obj.x)
            state.set(obj, "y", world_to_obj.y)
            state.set(obj, "theta", world_to_obj.theta)

        # Check for collisions, and only update the state if none exist.
        if not state_has_collision(state, self._static_object_body_cache):
            self._current_state = state

        # NOTE: add goals in the future.
        terminated = False
        truncated = False  # No maximum horizon, by default
        reward = 1 if terminated else 0  # Binary sparse rewards
        observation = self._get_obs()
        info = self._get_info()
        return observation, reward, terminated, truncated, info

    def render(self) -> NDArray[np.uint8]:
        assert self.render_mode == "rgb_array"
        return self._render_frame()

    def _render_frame(self) -> NDArray[np.uint8]:
        figsize = (
            self._world_max_x - self._world_min_x,
            self._world_max_y - self._world_min_y,
        )
        fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=self._render_dpi)

        assert self._current_state is not None, "Need to call reset()"
        state = self._current_state

        # Sort objects by ascending z order, with the robot first.
        def _render_order(obj: Object) -> int:
            if obj.is_instance(CRVRobotType):
                return -1
            return int(state.get(obj, "z_order"))

        for obj in sorted(state, key=_render_order):
            body = object_to_multibody2d(obj, state, self._static_object_body_cache)
            body.plot(ax)

        pad_x = (self._world_max_x - self._world_min_x) / 25
        pad_y = (self._world_max_y - self._world_min_y) / 25
        ax.set_xlim(self._world_min_x - pad_x, self._world_max_x + pad_x)
        ax.set_ylim(self._world_min_y - pad_y, self._world_max_y + pad_y)
        ax.axis("off")
        plt.tight_layout()
        img = fig2data(fig)
        plt.clf()
        return img