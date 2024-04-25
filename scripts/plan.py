"""Plan and execute in the ThreeTableEnv()."""

import argparse
from typing import Sequence, Set

import gym
from gym.wrappers.record_video import RecordVideo
from relational_structs import (
    LiftedOperator,
    Object,
    PDDLDomain,
    PDDLProblem,
    Predicate,
    State,
)
from relational_structs.utils import abstract

# Needed to register environments for gym.make().
import geom2drobotenvs  # pylint: disable=unused-import
from geom2drobotenvs.object_types import CRVRobotType, RectangleType
from geom2drobotenvs.structs import ZOrder


def _create_predicates() -> Set[Predicate]:
    predicates: Set[Predicate] = set()

    # On.
    def _on_holds(state: State, objs: Sequence[Object]) -> bool:
        target, table = objs
        import ipdb

        ipdb.set_trace()

    On = Predicate("On", [RectangleType, RectangleType], _on_holds)
    predicates.add(On)

    # ClearToPick.
    def _clear_to_pick_holds(state: State, objs: Sequence[Object]) -> bool:
        (target,) = objs
        import ipdb

        ipdb.set_trace()

    ClearToPick = Predicate("ClearToPick", [RectangleType], _clear_to_pick_holds)
    predicates.add(ClearToPick)

    # HandEmpty.
    def _hand_empty_holds(state: State, objs: Sequence[Object]) -> bool:
        (robot,) = objs
        import ipdb

        ipdb.set_trace()

    HandEmpty = Predicate("HandEmpty", [CRVRobotType], _hand_empty_holds)
    predicates.add(HandEmpty)

    # Holding.
    def _holding_holds(state: State, objs: Sequence[Object]) -> bool:
        robot, obj = objs
        import ipdb

        ipdb.set_trace()

    Holding = Predicate("Holding", [CRVRobotType, RectangleType], _holding_holds)
    predicates.add(Holding)

    return predicates


def _create_operators(predicates: Set[Predicate]) -> Set[LiftedOperator]:
    operators: Set[LiftedOperator] = set()
    pred_name_to_pred = {p.name: p for p in predicates}

    On = pred_name_to_pred["On"]
    HandEmpty = pred_name_to_pred["HandEmpty"]
    ClearToPick = pred_name_to_pred["ClearToPick"]
    Holding = pred_name_to_pred["Holding"]

    # Pick.
    robot = CRVRobotType("?robot")
    target = RectangleType("?target")
    table = RectangleType("?table")
    preconditions = {
        On([target, table]),
        ClearToPick([target]),
        HandEmpty([robot]),
    }
    add_effects = {
        Holding([target, robot]),
    }
    delete_effects = {
        On([target, table]),
        ClearToPick([target]),
        HandEmpty([robot]),
    }
    Pick = LiftedOperator(
        "Pick", [robot, target, table], preconditions, add_effects, delete_effects
    )
    operators.add(Pick)

    # Place.
    robot = CRVRobotType("?robot")
    held = RectangleType("?held")
    table = RectangleType("?table")
    preconditions = {
        Holding([target, robot]),
    }
    add_effects = {
        On([held, table]),
        ClearToPick([held]),
        HandEmpty([robot]),
    }
    delete_effects = {
        Holding([target, robot]),
    }
    Place = LiftedOperator(
        "Place", [robot, held, table], preconditions, add_effects, delete_effects
    )
    operators.add(Place)

    return operators


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("env", type=str)
    parser.add_argument("--seed", required=False, type=int, default=0)
    parser.add_argument("--outdir", required=False, type=str, default="plan_videos")
    args = parser.parse_args()
    env = gym.make(args.env)
    env = RecordVideo(env, args.outdir)
    obs, _ = env.reset(seed=args.seed)
    assert isinstance(obs, State)
    env.action_space.seed(args.seed)

    # Get the relevant objects.
    blocks = [
        o for o in obs if o.is_instance(RectangleType) and obs.get(o, "static") < 0.5
    ]
    blocks = sorted(blocks, key=lambda b: obs.get(b, "width") * obs.get(b, "height"))
    tables = [
        o
        for o in obs
        if o.is_instance(RectangleType)
        and obs.get(o, "static") > 0.5
        and int(obs.get(o, "z_order")) == ZOrder.FLOOR.value
    ]
    bx = obs.get(blocks[0], "x")
    by = obs.get(blocks[0], "y")
    dist = lambda t: (bx - obs.get(t, "x")) ** 2 + (by - obs.get(t, "y")) ** 2
    table = max(tables, key=dist)

    # Construct a PDDL domain and problem.
    predicates = _create_predicates()
    operators = _create_operators(predicates)
    types = {o.type for o in obs}
    domain = PDDLDomain("three-tables", operators, predicates, types)

    objects = set(obs)
    init_atoms = abstract(obs, predicates)
    pred_name_to_pred = {p.name: p for p in predicates}
    On = pred_name_to_pred["On"]
    goal = {On([block, table]) for block in blocks}
    problem = PDDLProblem(domain.name, "problem0", objects, init_atoms, goal)

    import ipdb

    ipdb.set_trace()

    for _ in range(args.steps):
        action = env.action_space.sample()
        _, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            print("WARNING: terminating early.")
            break
    env.close()


if __name__ == "__main__":
    _main()
