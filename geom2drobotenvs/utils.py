"""Utilities."""

import numpy as np
from gym.spaces import Box


class CRVRobotActionSpace(Box):
    """An action space for a CRV robot.

    Actions are bounded relative movements of the base and the arm, as
    well as an absolute setting for the vacuum.
    """

    def __init__(
        self,
        min_dx: float = -1e-1,
        max_dx: float = 1e-1,
        min_dy: float = -1e-1,
        max_dy: float = 1e-1,
        min_dtheta: float = -1.0,
        max_dtheta: float = 1.0,
        min_darm: float = -1e-1,
        max_darm: float = 1e-1,
        min_vac: float = 0.0,
        max_vac: float = 1.0,
    ) -> None:
        low = np.array([min_dx, min_dy, min_dtheta, min_darm, min_vac])
        high = np.array([max_dx, max_dy, max_dtheta, max_darm, max_vac])
        return super().__init__(low, high)
