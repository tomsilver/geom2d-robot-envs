from gym.envs.registration import register

register(
    id="geom2drobotenvs/ShelfWorld-v0",
    entry_point="geom2drobotenvs.envs:ShelfWorldEnv",
)
