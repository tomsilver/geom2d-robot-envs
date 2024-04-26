"""Plan and execute in the ThreeShelfEnv()."""

import argparse
from typing import Dict, Sequence, Set, Tuple

import gym
from gym.wrappers.record_video import RecordVideo
from relational_structs import (
    GroundOperator,
    LiftedOperator,
    Object,
    Option,
    PDDLDomain,
    PDDLProblem,
    Predicate,
    State,
)
from relational_structs.utils import abstract, parse_pddl_plan
from tomsgeoms2d.structs import LineSegment, Rectangle
from tomsgeoms2d.utils import geom2ds_intersect
from tomsutils.pddl_planning import run_pddl_planner

# Needed to register environments for gym.make().
import geom2drobotenvs  # pylint: disable=unused-import
from geom2drobotenvs.concepts import is_inside, is_movable_rectangle
from geom2drobotenvs.object_types import CRVRobotType, Geom2DType, RectangleType
from geom2drobotenvs.skills import (
    create_rectangle_vaccum_pick_option,
    create_rectangle_vaccum_table_place_option,
)
from geom2drobotenvs.structs import MultiBody2D, ZOrder
from geom2drobotenvs.utils import (
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
    def _get_objects_on_object(state: State, obj: Object) -> Set[Object]:
        ret_objs: Set[Object] = set()
        surface = rectangle_object_to_geom(state, obj, static_object_cache)
        for other_obj in state.get_objects(RectangleType):
            if other_obj == obj:
                continue
            rect = rectangle_object_to_geom(state, other_obj, static_object_cache)
            x, y = rect.center
            if surface.contains_point(x, y):
                ret_objs.add(other_obj)
        return ret_objs

    def _get_shelf_empty_side_center(
        state: State, shelf: Object
    ) -> Tuple[float, float]:
        objs_on_top = _get_objects_on_object(state, shelf)
        walls = {o for o in objs_on_top if state.get(o, "static") > 0.5}
        assert len(walls) == 3
        wall_rects = {
            rectangle_object_to_geom(state, w, static_object_cache) for w in walls
        }
        example_wall_rect = next(iter(wall_rects))
        wall_thickness = min(example_wall_rect.height, example_wall_rect.width)
        pad = wall_thickness / 2
        shelf_rect = rectangle_object_to_geom(state, shelf, static_object_cache)
        height_scale = (shelf_rect.height - pad) / shelf_rect.height
        width_scale = (shelf_rect.width - pad) / shelf_rect.width
        inner_shelf_rect = shelf_rect.scale_about_center(
            width_scale=width_scale, height_scale=height_scale
        )
        for v1, v2 in zip(
            inner_shelf_rect.vertices,
            inner_shelf_rect.vertices[1:] + [inner_shelf_rect.vertices[0]],
        ):
            cx = (v1[0] + v2[0]) / 2
            cy = (v1[1] + v2[1]) / 2
            contained_in_wall = False
            for wall_rect in wall_rects:
                if wall_rect.contains_point(cx, cy):
                    contained_in_wall = True
                    break

            if not contained_in_wall:
                return (cx, cy)
        raise ValueError("There is no empty side on the shelf.")

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
        objs_on_top = _get_objects_on_object(state, shelf)
        static_objs = {o for o in objs_on_top if state.get(o, "static") > 0.5}
        return len(static_objs) == 3

    IsShelf = Predicate("IsShelf", [RectangleType], _is_shelf_holds)
    predicates.add(IsShelf)

    # OnShelf.
    def _on_shelf_holds(state: State, objs: Sequence[Object]) -> bool:
        target, shelf = objs
        if not _is_block_holds(state, [target]):
            return False
        if not _is_shelf_holds(state, [shelf]):
            return False
        return is_inside(state, target, shelf, static_object_cache)

    OnShelf = Predicate("OnShelf", [RectangleType, RectangleType], _on_shelf_holds)
    predicates.add(OnShelf)

    # InFrontOnShelf.
    def _in_front_on_shelf(state: State, objs: Sequence[Object]) -> bool:
        # First draw a line from the behind object to the empty side of the
        # shelf and check if the front object intersects that line. If so,
        # draw a line from the behind object to the front object and check if
        # any other objects intersect that line.
        front_obj, behind_obj, shelf = objs
        if front_obj == behind_obj:
            return False
        if not _is_shelf_holds(state, [shelf]):
            return False
        if not _is_block_holds(state, [front_obj]):
            return False
        if not _is_block_holds(state, [behind_obj]):
            return False
        if not _on_shelf_holds(state, [front_obj, shelf]):
            return False
        if not _on_shelf_holds(state, [behind_obj, shelf]):
            return False
        empty_x, empty_y = _get_shelf_empty_side_center(state, shelf)
        front_rect = rectangle_object_to_geom(state, front_obj, static_object_cache)
        behind_rect = rectangle_object_to_geom(state, behind_obj, static_object_cache)
        behind_to_empty = LineSegment(
            behind_rect.center[0], behind_rect.center[1], empty_x, empty_y
        )
        if not geom2ds_intersect(front_rect, behind_to_empty):
            return False
        behind_to_front = LineSegment(
            behind_rect.center[0],
            behind_rect.center[1],
            front_rect.center[0],
            front_rect.center[1],
        )
        for other_obj in state.get_objects(RectangleType):
            if other_obj in {front_obj, behind_obj}:
                continue
            if not _is_block_holds(state, [other_obj]):
                continue
            other_rect = rectangle_object_to_geom(state, other_obj, static_object_cache)
            if geom2ds_intersect(other_rect, behind_to_front):
                return False
        return True

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


def _ground_op_to_option(ground_op: GroundOperator, action_space: gym.Space) -> Option:
    if ground_op.name == "PickFromInFront":
        param_option = create_rectangle_vaccum_pick_option(action_space)
        robot, target, _, _ = ground_op.parameters
        return param_option.ground([robot, target])

    if ground_op.name == "PickGeneric":
        param_option = create_rectangle_vaccum_pick_option(action_space)
        robot, target, _ = ground_op.parameters
        return param_option.ground([robot, target])

    if ground_op.name == "PlaceInFront":
        param_option = create_rectangle_vaccum_table_place_option(action_space)
        robot, target, _, shelf = ground_op.parameters
        return param_option.ground([robot, target, shelf])

    if ground_op.name == "PlaceGeneric":
        param_option = create_rectangle_vaccum_table_place_option(action_space)
        robot, target, shelf = ground_op.parameters
        return param_option.ground([robot, target, shelf])

    raise NotImplementedError


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

    ground_op_strs = run_pddl_planner(str(domain), str(problem))
    ground_op_plan = parse_pddl_plan(ground_op_strs, domain, problem)
    option_plan = [_ground_op_to_option(o, env.action_space) for o in ground_op_plan]

    for option in option_plan:
        print("Starting option", option)
        assert option.initiable(obs)
        for _ in range(100):
            action = option.policy(obs)
            obs, _, terminated, truncated, _ = env.step(action)
            assert not terminated or truncated
            if option.terminal(obs):
                break
        else:
            assert False, "Option did not terminate"

    env.close()


if __name__ == "__main__":
    _main()
