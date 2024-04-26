"""Plan and execute in the ThreeShelfEnv()."""

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
    rectangle_object_to_geom,
    z_orders_may_collide,
)


def _create_predicates(
    static_object_cache: Dict[Object, MultiBody2D]
) -> Set[Predicate]:
    predicates: Set[Predicate] = set()

    # Helper functions.
    def _get_immovable_objects_on_object(state: State, obj: Object) -> Set[Object]:
        immovable_objs: Set[Object] = set()
        surface = rectangle_object_to_geom(state, obj, static_object_cache)
        for other_obj in state.get_objects(RectangleType):
            if other_obj == obj:
                continue
            if state.get(other_obj, "static") < 0.5:
                continue
            rect = rectangle_object_to_geom(state, other_obj, static_object_cache)
            x, y = rect.center
            if surface.contains_point(x, y):
                immovable_objs.add(other_obj)
        return immovable_objs

    def _get_shelf_reference_frame(state: State, shelf: Object) -> SE2Pose:
        import ipdb

        ipdb.set_trace()

    # IsBlock.
    def _is_block_holds(state: State, objs: Sequence[Object]) -> bool:
        (block,) = objs
        return is_movable_rectangle(state, block)

    IsBlock = Predicate("IsBlock", [RectangleType], _is_block_holds)
    predicates.add(IsBlock)

    # IsShelf.
    def _is_shelf_holds(state: State, objs: Sequence[Object]) -> bool:
        # This is very naive -- doesn't even check the pose or shape of the
        # static objects, just assumes 3 static objects == is shelf.
        (shelf,) = objs
        static_objs = _get_immovable_objects_on_object(state, shelf)
        return len(static_objs) == 3

    IsShelf = Predicate("IsShelf", [RectangleType], _is_shelf_holds)
    predicates.add(IsShelf)

    # OnShelf.
    def _on_shelf_holds(state: State, objs: Sequence[Object]) -> bool:
        target, shelf = objs
        return is_inside(state, target, shelf, static_object_cache)

    OnShelf = Predicate("OnShelf", [RectangleType, RectangleType], _on_shelf_holds)
    predicates.add(OnShelf)

    # InFrontOnShelf.
    def _in_front_on_shelf(state: State, objs: Sequence[Object]) -> bool:
        obj1, obj2, shelf = objs
        if not _is_shelf_holds(state, [shelf]):
            return False
        world_to_shelf = _get_shelf_reference_frame(state, shelf)
        world_to_obj1 = get_se2_pose(state, obj1)
        world_to_obj2 = get_se2_pose(state, obj2)
        shelf_to_obj1 = world_to_shelf.inverse * world_to_obj1
        shelf_to_obj2 = world_to_shelf.inverse * world_to_obj2
        import ipdb

        ipdb.set_trace()

    InFrontOnShelf = Predicate(
        "InFrontOnShelf",
        [RectangleType, RectangleType, RectangleType],
        _in_front_on_shelf,
    )
    predicates.add(InFrontOnShelf)

    # ClearToPick.
    def _clear_to_pick_holds(state: State, objs: Sequence[Object]) -> bool:
        if not _on_shelf_holds(state, objs):
            return False
        target, shelf = objs
        if not is_movable_rectangle(state, target):
            return False
        # This is difficult to define in general... so we'll define it in a
        # hacky way... draw a line from the object to each side of the shelf
        # that it's on. If that line doesn't intersect anything, we're clear.
        shelf_mb = object_to_multibody2d(shelf, state, static_object_cache)
        assert len(shelf_mb.bodies) == 1
        shelf_rect = shelf_mb.bodies[0].geom
        assert isinstance(shelf_rect, Rectangle)
        target_mb = object_to_multibody2d(target, state, static_object_cache)
        assert len(target_mb.bodies) == 1
        target_rect = target_mb.bodies[0].geom
        assert isinstance(target_rect, Rectangle)
        target_x, target_y = target_rect.center
        target_z_order = ZOrder(int(state.get(target, "z_order")))
        obstacles = set(state) - {target, shelf}
        for (x1, y1), (x2, y2) in zip(
            shelf_rect.vertices, shelf_rect.vertices[1:] + [shelf_rect.vertices[0]]
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

    IsBlock = pred_name_to_pred["IsBlock"]
    IsShelf = pred_name_to_pred["IsShelf"]
    OnShelf = pred_name_to_pred["OnShelf"]
    HandEmpty = pred_name_to_pred["HandEmpty"]
    ClearToPick = pred_name_to_pred["ClearToPick"]
    Holding = pred_name_to_pred["Holding"]
    InFrontOnShelf = pred_name_to_pred["InFrontOnShelf"]

    # PickFromInFront.
    robot = CRVRobotType("?robot")
    target = RectangleType("?target")
    behind = RectangleType("?behind")
    shelf = RectangleType("?shelf")
    preconditions = {
        IsShelf([shelf]),
        IsBlock([target]),
        IsBlock([behind]),
        OnShelf([target, shelf]),
        ClearToPick([target, shelf]),
        HandEmpty([robot]),
        InFrontOnShelf([target, behind, shelf]),
    }
    add_effects = {
        Holding([target, robot]),
        ClearToPick([behind, shelf]),
    }
    delete_effects = {
        OnShelf([target, shelf]),
        ClearToPick([target, shelf]),
        HandEmpty([robot]),
    }
    PickFromInFront = LiftedOperator(
        "PickFromInFront",
        [robot, target, behind, shelf],
        preconditions,
        add_effects,
        delete_effects,
    )
    operators.add(PickFromInFront)

    # PickGeneric.
    robot = CRVRobotType("?robot")
    target = RectangleType("?target")
    shelf = RectangleType("?shelf")
    preconditions = {
        IsShelf([shelf]),
        IsBlock([target]),
        OnShelf([target, shelf]),
        ClearToPick([target, shelf]),
        HandEmpty([robot]),
    }
    add_effects = {
        Holding([target, robot]),
    }
    delete_effects = {
        OnShelf([target, shelf]),
        ClearToPick([target, shelf]),
        HandEmpty([robot]),
    }
    PickGeneric = LiftedOperator(
        "PickGeneric",
        [robot, target, shelf],
        preconditions,
        add_effects,
        delete_effects,
    )
    operators.add(PickGeneric)

    # PlaceInFront.
    robot = CRVRobotType("?robot")
    held = RectangleType("?held")
    behind = RectangleType("?behind")
    shelf = RectangleType("?shelf")
    preconditions = {
        IsBlock([held]),
        IsShelf([shelf]),
        Holding([held, robot]),
        ClearToPick([behind, shelf]),
    }
    add_effects = {
        OnShelf([held, shelf]),
        ClearToPick([held, shelf]),
        HandEmpty([robot]),
        InFrontOnShelf([held, behind, shelf]),
    }
    delete_effects = {
        Holding([held, robot]),
        ClearToPick([behind, shelf]),
    }
    PlaceInFront = LiftedOperator(
        "PlaceInFront",
        [robot, held, behind, shelf],
        preconditions,
        add_effects,
        delete_effects,
    )
    operators.add(PlaceInFront)

    # PlaceGeneric.
    robot = CRVRobotType("?robot")
    held = RectangleType("?held")
    shelf = RectangleType("?shelf")
    preconditions = {
        IsBlock([held]),
        IsShelf([shelf]),
        Holding([held, robot]),
    }
    add_effects = {
        OnShelf([held, shelf]),
        ClearToPick([held, shelf]),
        HandEmpty([robot]),
    }
    delete_effects = {
        Holding([held, robot]),
    }
    PlaceGeneric = LiftedOperator(
        "PlaceGeneric", [robot, held, shelf], preconditions, add_effects, delete_effects
    )
    operators.add(PlaceGeneric)

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
    shelfs = [
        o
        for o in obs
        if o.is_instance(RectangleType)
        and obs.get(o, "static") > 0.5
        and int(obs.get(o, "z_order")) == ZOrder.FLOOR.value
    ]
    bx = obs.get(blocks[0], "x")
    by = obs.get(blocks[0], "y")
    dist = lambda t: (bx - obs.get(t, "x")) ** 2 + (by - obs.get(t, "y")) ** 2
    shelf = max(shelfs, key=dist)

    # Construct a PDDL domain and problem.
    static_object_cache: Dict[Object, MultiBody2D] = {}
    predicates = _create_predicates(static_object_cache)
    operators = _create_operators(predicates)
    types = {Geom2DType, CRVRobotType, RectangleType}
    domain = PDDLDomain("three-shelfs", operators, predicates, types)

    objects = set(obs)
    init_atoms = abstract(obs, predicates)
    pred_name_to_pred = {p.name: p for p in predicates}
    OnShelf = pred_name_to_pred["OnShelf"]
    goal = {OnShelf([block, shelf]) for block in blocks}
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
