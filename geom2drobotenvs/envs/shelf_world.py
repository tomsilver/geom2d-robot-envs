from typing import ClassVar, Dict, Optional

import gym
import numpy as np
from relational_structs.spaces import ObjectCentricStateSpace
from relational_structs.structs import Object, State
from relational_structs.utils import create_state_from_dict

from geom2drobotenvs.object_types import CRVRobotType, RectangleType
from geom2drobotenvs.utils import CRVRobotActionSpace


class ShelfWorldEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # The world is oriented like a standard X/Y coordinate frame.
    _world_min_x: ClassVar[float] = 0.0
    _world_max_x: ClassVar[float] = 10.0
    _world_min_y: ClassVar[float] = 0.0
    _world_max_y: ClassVar[float] = 10.0

    def __init__(self):
        self._types = {RectangleType, CRVRobotType}
        self.observation_space = ObjectCentricStateSpace(self._types)
        self.action_space = CRVRobotActionSpace()

        # Initialized by reset().
        self._current_state: Optional[State] = None

    def _get_obs(self):
        return self._current_state.copy()

    def _get_info(self):
        return {}  # no extra info provided

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Coming soon: randomization.
        init_state_dict: Dict[Object, Dict[str, float]] = {}
        robot = CRVRobotType("robot")
        init_state_dict[robot] = {
            "x": (self._world_min_x + self._world_max_x) / 2.0,  # center of room
            "y": (self._world_min_y + self._world_max_y) / 2.0,
            "theta": 0.0,  # facing right
            "base_radius": 1.0,
            "arm_joint": 1.0,  # arm is fully retracted
            "vacuum_on": 0.0,  # vacuum is off
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
            "is_static": True,  # table can't move
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
        import ipdb

        ipdb.set_trace()
        terminated = ...
        truncated = False  # No maximum horizon, by default
        reward = 1 if terminated else 0  # Binary sparse rewards
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, truncated, info

    def render(self):
        assert self.render_mode == "rgb_array"
        return self._render_frame()

    def _render_frame(self):
        import ipdb

        ipdb.set_trace()

    def close(self):
        import ipdb

        ipdb.set_trace()
