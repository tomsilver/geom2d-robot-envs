"""Register environments for use with gym.make()."""

import matplotlib
from gymnasium.envs.registration import register

# We require the Agg backend to ensure consistent image shapes between systems.
# Previously we did not have this, and different users with the macosx had different
# rendered image shapes because they had different monitor sizes.
matplotlib.use("Agg")


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
