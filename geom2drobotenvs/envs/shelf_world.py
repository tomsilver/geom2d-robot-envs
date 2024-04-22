"""Shelf world environment."""

from typing import ClassVar, Dict, Optional, Tuple

import gym
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from relational_structs.spaces import ObjectCentricStateSpace
from relational_structs.structs import Array, Object, State
from relational_structs.utils import create_state_from_dict
from tomsutils.utils import fig2data, wrap_angle

from geom2drobotenvs.object_types import CRVRobotType, RectangleType
from geom2drobotenvs.structs import Body2D, ZOrder
from geom2drobotenvs.utils import (
    CRVRobotActionSpace,
    create_walls_from_world_boundaries,
    object_to_body2d,
    state_has_collision,
)


class ShelfWorldEnv(gym.Env):
    """Shelf world environment."""

    # Only RGB rendering is implemented.
    render_mode = "rgb_array"
    metadata = {"render_modes": [render_mode]}
    _render_dpi: int = 150

    # The world is oriented like a standard X/Y coordinate frame.
    _world_min_x: ClassVar[float] = 0.0
    _world_max_x: ClassVar[float] = 10.0
    _world_min_y: ClassVar[float] = 0.0
    _world_max_y: ClassVar[float] = 10.0

    _robot_base_radius: ClassVar[float] = 0.36
    _max_robot_arm_joint: ClassVar[float] = 2.0

    def __init__(self) -> None:
        self._types = {RectangleType, CRVRobotType}
        self.observation_space = ObjectCentricStateSpace(self._types)
        self.action_space = CRVRobotActionSpace()

        # Initialized by reset().
        self._current_state: Optional[State] = None
        self._static_object_body_cache: Dict[Object, Body2D] = {}

        super().__init__()

    def _get_obs(self) -> State:
        assert self._current_state is not None, "Need to call reset()"
        return self._current_state.copy()

    def _get_info(self) -> Dict:
        return {}  # no extra info provided

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[State, Dict]:
        super().reset(seed=seed)

        # Need to flush the cache in case static objects move.
        self._static_object_body_cache = {}

        # For testing purposes only, the options may specify an initial scene.
        if options is not None and "init_state" in options:
            self._current_state = options["init_state"].copy()

        # Otherwise, set up the initial scene here.
        else:
            # Coming soon: randomization.
            init_state_dict: Dict[Object, Dict[str, float]] = {}
            robot = CRVRobotType("robot")
            init_state_dict[robot] = {
                "x": (self._world_min_x + self._world_max_x) / 2.0,  # center of room
                "y": (self._world_min_y + self._world_max_y) / 2.0,
                "theta": 0.0,  # facing right
                "base_radius": self._robot_base_radius,
                "arm_joint": self._robot_base_radius,  # arm is fully retracted
                "vacuum": 0.0,  # vacuum is off
            }
            right_table = RectangleType("right_table")
            right_table_width = (self._world_max_x - self._world_min_x) / 10.0
            right_table_height = (self._world_max_y - self._world_min_y) / 3.0
            right_table_right_pad = right_table_width / 2
            init_state_dict[right_table] = {
                # Origin is bottom left hand corner.
                "x": self._world_max_x - (right_table_width + right_table_right_pad),
                "y": (self._world_min_y + self._world_max_y - right_table_height) / 2.0,
                "width": right_table_width,
                "height": right_table_height,
                "theta": 0.0,
                "static": True,  # table can't move
                "color_r": 0.4,  # gray
                "color_g": 0.4,
                "color_b": 0.4,
                "z_order": ZOrder.FLOOR.value,
            }
            assert isinstance(self.action_space, CRVRobotActionSpace)
            min_dx, min_dy = self.action_space.low[:2]
            max_dx, max_dy = self.action_space.high[:2]
            wall_state_dict = create_walls_from_world_boundaries(
                self._world_min_x,
                self._world_max_x,
                self._world_min_y,
                self._world_max_y,
                min_dx,
                max_dx,
                min_dy,
                max_dy,
            )
            init_state_dict.update(wall_state_dict)
            # Finalize state.
            self._current_state = create_state_from_dict(init_state_dict)

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action: Array) -> Tuple[State, float, bool, bool, Dict]:
        # NOTE: this should be abstracted out in the future.
        assert self.action_space.contains(action)
        dx, dy, dtheta, darm, vac = action
        assert self._current_state is not None, "Need to call reset()"
        state = self._current_state.copy()
        robot = next(o for o in state if o.is_instance(CRVRobotType))

        # NOTE: xy clipping is not needed because world boundaries are handled
        # by collision detection with walls.
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

        # Check for collisions, and only update the state if none exist.
        if not state_has_collision(state, self._static_object_body_cache):
            self._current_state = state

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
        # NOTE: this should be abstracted out in the future.
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
            body = object_to_body2d(obj, state, self._static_object_body_cache)
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
