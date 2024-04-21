"""Shelf world environment."""

from typing import ClassVar, Dict, Optional, List

import gym
import matplotlib.pyplot as plt
import numpy as np
from relational_structs.spaces import ObjectCentricStateSpace
from relational_structs.structs import Object, State
from relational_structs.utils import create_state_from_dict

from geom2drobotenvs.object_types import CRVRobotType, RectangleType
from geom2drobotenvs.utils import CRVRobotActionSpace, object_to_geom2d_list

from tomsgeoms2d.structs import Geom2D
from tomsutils.utils import fig2data, wrap_angle


class ShelfWorldEnv(gym.Env):
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

    def __init__(self):
        self._types = {RectangleType, CRVRobotType}
        self.observation_space = ObjectCentricStateSpace(self._types)
        self.action_space = CRVRobotActionSpace()

        # Initialized by reset().
        self._current_state: Optional[State] = None
        self._static_object_geom_cache: Dict[Object, List[Geom2D]] = {}

        super().__init__()

    def _get_obs(self):
        return self._current_state.copy()

    def _get_info(self):
        return {}  # no extra info provided

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Need to flush the cache in case static objects move.
        self._static_object_geom_cache = {}

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
            "y": (self._world_min_y + self._world_max_y) / 2.0,
            "width": right_table_width,
            "height": right_table_height,
            "theta": 0.0,
            "static": True,  # table can't move
            "color_r": 0.4,  # gray
            "color_g": 0.4,
            "color_b": 0.4,
        }
        # Finalize state.
        self._current_state = create_state_from_dict(init_state_dict)

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action):
        # NOTE: this should be abstracted out in the future.
        assert self.action_space.contains(action)
        dx, dy, dtheta, darm, vac = action
        state = self._current_state.copy()
        robot = next(o for o in state if o.is_instance(CRVRobotType))
        new_x = np.clip(state.get(robot, "x") + dx, self._world_min_x, self._world_max_x)
        new_y = np.clip(state.get(robot, "y") + dy, self._world_min_y, self._world_max_y)
        new_theta = wrap_angle(state.get(robot, "theta") + dtheta)
        min_arm = state.get(robot, "base_radius")
        max_arm = self._max_robot_arm_joint
        new_arm = np.clip(state.get(robot, "arm_joint") + darm, min_arm, max_arm)
        state.set(robot, "x", new_x)
        state.set(robot, "y", new_y)
        state.set(robot, "arm_joint", new_arm)
        state.set(robot, "theta", new_theta)
        state.set(robot, "vacuum", vac)
        self._current_state = state
        terminated = False
        truncated = False  # No maximum horizon, by default
        reward = 1 if terminated else 0  # Binary sparse rewards
        observation = self._get_obs()
        info = self._get_info()
        return observation, reward, terminated, truncated, info

    def render(self):
        assert self.render_mode == "rgb_array"
        return self._render_frame()

    def _render_frame(self):
        # NOTE: this should be abstracted out in the future.
        figsize = (self._world_max_x - self._world_min_x, self._world_max_y - self._world_min_y)
        fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=self._render_dpi)

        for obj in self._current_state:
            geoms = object_to_geom2d_list(obj, self._current_state, self._static_object_geom_cache)
            if obj.is_instance(CRVRobotType):
                color = (0.5, 0.6, 0.7)
            else:
                color = (
                    self._current_state.get(obj, "color_r"),
                    self._current_state.get(obj, "color_g"),
                    self._current_state.get(obj, "color_b"),
                )
            for geom in geoms:
                geom.plot(ax,
                          facecolor=color, edgecolor="black")

        ax.set_xlim(self._world_min_x, self._world_max_x)
        ax.set_ylim(self._world_min_y, self._world_max_y)
        ax.axis("off")
        plt.tight_layout()
        img = fig2data(fig)
        plt.close()
        return img

    def close(self):
        import ipdb

        ipdb.set_trace()
