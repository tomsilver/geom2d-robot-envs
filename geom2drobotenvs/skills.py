"""Skills that might be useful in certain environments."""

from relational_structs.structs import ParameterizedOption, State, Object, OptionMemory, Action
from relational_structs.spaces import ObjectSequenceSpace
from geom2drobotenvs.object_types import CRVRobotType, RectangleType
from typing import Sequence


# Move to the center of the world, then directly towards the target rectangle
# until it is within reach. Then extend the arm and turn on the vacuum.
name = "CartesianRectangleVacuumPick"
params_space = ObjectSequenceSpace([CRVRobotType, RectangleType])

def _policy(state: State, params: Sequence[Object], memory: OptionMemory) -> Action:
    import ipdb; ipdb.set_trace()


def _initiable(state: State, params: Sequence[Object], memory: OptionMemory) -> bool:
    import ipdb; ipdb.set_trace()


def _terminal(state: State, params: Sequence[Object], memory: OptionMemory) -> bool:
    import ipdb; ipdb.set_trace()

CartesianRectangleVacuumPick = ParameterizedOption(name, params_space, _policy, _initiable, _terminal)
