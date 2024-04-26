"""Plan and execute in the ThreeTableEnv()."""

import argparse
from typing import Dict, Sequence, Set

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
from tomsgeoms2d.structs import LineSegment, Rectangle
from tomsgeoms2d.utils import geom2ds_intersect
from tomsutils.pddl_planning import run_pddl_planner

# Needed to register environments for gym.make().
import geom2drobotenvs  # pylint: disable=unused-import
from geom2drobotenvs.concepts import is_inside, is_movable_rectangle
from geom2drobotenvs.object_types import CRVRobotType, Geom2DType, RectangleType
from geom2drobotenvs.structs import MultiBody2D, SE2Pose, ZOrder
from geom2drobotenvs.utils import (
    get_se2_pose,
    get_suctioned_objects,
    object_to_multibody2d,
    z_orders_may_collide,
)


def _create_predicates(
    static_object_cache: Dict[Object, MultiBody2D]
) -> Set[Predicate]:
    predicates: Set[Predicate] = set()

    # Helper function.
    def _get_table_reference_frame(state: State, table: Object) -> SE2Pose:
        import ipdb

        ipdb.set_trace()

    # OnTable.
    def _on_table_holds(state: State, objs: Sequence[Object]) -> bool:
        target, table = objs
        return is_inside(state, target, table, static_object_cache)

    OnTable = Predicate("OnTable", [RectangleType, RectangleType], _on_table_holds)
    predicates.add(OnTable)

    # InFrontOnTable.
    def _in_front_on_table(state: State, objs: Sequence[Object]) -> bool:
        obj1, obj2, table = objs
        world_to_table = _get_table_reference_frame(state, table)
        world_to_obj1 = get_se2_pose(state, obj1)
        world_to_obj2 = get_se2_pose(state, obj2)
        table_to_obj1 = world_to_table.inverse * world_to_obj1
        table_to_obj2 = world_to_table.inverse * world_to_obj2
        import ipdb

        ipdb.set_trace()

    InFrontOnTable = Predicate(
        "InFrontOnTable",
        [RectangleType, RectangleType, RectangleType],
        _in_front_on_table,
    )
    predicates.add(InFrontOnTable)

    # ClearToPick.
    def _clear_to_pick_holds(state: State, objs: Sequence[Object]) -> bool:
        if not _on_table_holds(state, objs):
            return False
        target, table = objs
        if not is_movable_rectangle(state, target):
            return False
        # This is difficult to define in general... so we'll define it in a
        # hacky way... draw a line from the object to each side of the table
        # that it's on. If that line doesn't intersect anything, we're clear.
        table_mb = object_to_multibody2d(table, state, static_object_cache)
        assert len(table_mb.bodies) == 1
        table_rect = table_mb.bodies[0].geom
        assert isinstance(table_rect, Rectangle)
        target_mb = object_to_multibody2d(target, state, static_object_cache)
        assert len(target_mb.bodies) == 1
        target_rect = target_mb.bodies[0].geom
        assert isinstance(target_rect, Rectangle)
        target_x, target_y = target_rect.center
        target_z_order = ZOrder(int(state.get(target, "z_order")))
        obstacles = set(state) - {target, table}
        for (x1, y1), (x2, y2) in zip(
            table_rect.vertices, table_rect.vertices[1:] + [table_rect.vertices[0]]
        ):
            side_x = (x1 + x2) / 2
            side_y = (y1 + y2) / 2
            line_geom = LineSegment(target_x, target_y, side_x, side_y)
            line_is_clear = True
            for obstacle in obstacles:
                if not line_is_clear:
                    break
                obstacle_multibody = object_to_multibody2d(
                    obstacle, state, static_object_cache
                )
                for obstacle_body in obstacle_multibody.bodies:
                    if not z_orders_may_collide(target_z_order, obstacle_body.z_order):
                        continue
                    if geom2ds_intersect(line_geom, obstacle_body.geom):
                        line_is_clear = False
                        break
            if line_is_clear:
                return True
        return False

    ClearToPick = Predicate(
        "ClearToPick", [RectangleType, RectangleType], _clear_to_pick_holds
    )
    predicates.add(ClearToPick)

    # HandEmpty.
    def _hand_empty_holds(state: State, objs: Sequence[Object]) -> bool:
        (robot,) = objs
        return not get_suctioned_objects(state, robot)

    HandEmpty = Predicate("HandEmpty", [CRVRobotType], _hand_empty_holds)
    predicates.add(HandEmpty)

    # Holding.
    def _holding_holds(state: State, objs: Sequence[Object]) -> bool:
        obj, robot = objs
        held_objs = {o for o, _ in get_suctioned_objects(state, robot)}
        return obj in held_objs

    Holding = Predicate("Holding", [RectangleType, CRVRobotType], _holding_holds)
    predicates.add(Holding)

    return predicates


def _create_operators(predicates: Set[Predicate]) -> Set[LiftedOperator]:
    operators: Set[LiftedOperator] = set()
    pred_name_to_pred = {p.name: p for p in predicates}

    OnTable = pred_name_to_pred["OnTable"]
    HandEmpty = pred_name_to_pred["HandEmpty"]
    ClearToPick = pred_name_to_pred["ClearToPick"]
    Holding = pred_name_to_pred["Holding"]
    InFrontOnTable = pred_name_to_pred["InFrontOnTable"]

    # PickFromInFront.
    robot = CRVRobotType("?robot")
    target = RectangleType("?target")
    behind = RectangleType("?behind")
    table = RectangleType("?table")
    preconditions = {
        OnTable([target, table]),
        ClearToPick([target, table]),
        HandEmpty([robot]),
        InFrontOnTable([target, behind, table]),
    }
    add_effects = {
        Holding([target, robot]),
        ClearToPick([behind, table]),
    }
    delete_effects = {
        OnTable([target, table]),
        ClearToPick([target, table]),
        HandEmpty([robot]),
    }
    PickFromInFront = LiftedOperator(
        "PickFromInFront",
        [robot, target, table],
        preconditions,
        add_effects,
        delete_effects,
    )
    operators.add(PickFromInFront)

    # Place.
    robot = CRVRobotType("?robot")
    held = RectangleType("?held")
    table = RectangleType("?table")
    preconditions = {
        Holding([held, robot]),
    }
    add_effects = {
        OnTable([held, table]),
        ClearToPick([held, table]),
        HandEmpty([robot]),
    }
    delete_effects = {
        Holding([held, robot]),
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
    blocks = [o for o in obs if is_movable_rectangle(obs, o)]
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
    static_object_cache: Dict[Object, MultiBody2D] = {}
    predicates = _create_predicates(static_object_cache)
    operators = _create_operators(predicates)
    types = {Geom2DType, CRVRobotType, RectangleType}
    domain = PDDLDomain("three-tables", operators, predicates, types)

    objects = set(obs)
    init_atoms = abstract(obs, predicates)
    pred_name_to_pred = {p.name: p for p in predicates}
    OnTable = pred_name_to_pred["OnTable"]
    goal = {OnTable([block, table]) for block in blocks}
    problem = PDDLProblem(domain.name, "problem0", objects, init_atoms, goal)

    action_strs = run_pddl_planner(str(domain), str(problem))
    assert action_strs is not None
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
