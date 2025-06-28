"""Register environments for use with gym.make()."""

from gymnasium.envs.registration import register


def register_all_environments() -> None:
    """Register all environments in this repository."""

    register(
        id="geom2drobotenvs/ThreeTables-v0",
        entry_point="geom2drobotenvs.envs.three_table_env:ThreeTableEnv",
    )

    register(
        id="geom2drobotenvs/Obstruction2D-v0",
        entry_point="geom2drobotenvs.envs.obstruction_2d_env:Obstruction2DEnv",
    )
