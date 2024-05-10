"""Register environments for use with gym.make()."""

from gymnasium.envs.registration import register

register(
    id="geom2drobotenvs/ThreeTables-v0",
    entry_point="geom2drobotenvs.envs.three_table_env:ThreeTableEnv",
)
