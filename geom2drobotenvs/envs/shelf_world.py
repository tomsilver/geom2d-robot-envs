import gym
from relational_structs.spaces import ObjectCentricStateSpace
from geom2drobotenvs.object_types import RectangleType, CRVRobotType
from geom2drobotenvs.utils import CRVRobotActionSpace
import numpy as np


class ShelfWorldEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self):
        self._types = {RectangleType, CRVRobotType}
        self.observation_space = ObjectCentricStateSpace(self._types)
        self.action_space = CRVRobotActionSpace()

    def _get_obs(self):
        import ipdb; ipdb.set_trace()

    def _get_info(self):
        import ipdb; ipdb.set_trace()

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random.
        super().reset(seed=seed)

        import ipdb; ipdb.set_trace()

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action):
        import ipdb; ipdb.set_trace()
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
        import ipdb; ipdb.set_trace()

    def close(self):
        import ipdb; ipdb.set_trace()
