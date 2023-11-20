import sys
from lux.kit import obs_to_game_state, EnvConfig, GameState
from lux.utils import *
from lux.pathfinding import get_shortest_loop
from lux.forward_sim import stop_movement_collisions
from lux.config import resource_ids

import numpy as np
from numpy.linalg import norm

from lux.states import (
    Plan,
    ResourcePlan,
    TransmitPlan,
    UnitState,
    UnitStateEnum,
    FactoryState,
    EarlyStepState,
    UnitMission,
    UnitRole,
    FactoryRole,
    FactoryId,
    UnitId,
)

from lux.factory import Factory
from lux.unit import Unit


class Agent:
    def __init__(self, player: str, env_cfg: EnvConfig) -> None:
        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"

        np.random.seed(0)

        self.env_cfg: EnvConfig = env_cfg
        EarlyStepState.MAP_SIZE = env_cfg.map_size
        self.early_step_state: EarlyStepState = EarlyStepState()

        self.unit_states: dict[UnitId, UnitState] = {}
        self.factory_states: dict[FactoryId, FactoryState] = {}
        self.move_cost_map = -1, None

    def _num_factories(self, game_state: GameState) -> int:
        factories = game_state.factories
        n_0 = len(factories["player_0"]) if "player_0" in factories else 0
        n_1 = len(factories["player_1"]) if "player_1" in factories else 0

        return n_0 + n_1

    def early_setup(self, step: int, obs, remainingOverageTime: int = 60):
        if step == 0:
            # bid 0 to not waste plans bidding and declare as the default faction
            # you can bid -n to prefer going second or n to prefer going first in placement
            return dict(faction="AlphaStrike", bid=0)
        else:
            game_state = obs_to_game_state(step, self.env_cfg, obs)
            # factory placement period
            if self.early_step_state.factory_score is None:
                map_size = self.env_cfg.map_size
                es_state = self.early_step_state
                rubble_score = conv2d(game_state.board.rubble, average_kernel(5), n=2)
                normalized_rubble_score = (rubble_score / np.max(rubble_score)) * 0.5

                ice_tile_locations = np.argwhere(game_state.board.ice == 1)
                ore_tile_locations = np.argwhere(game_state.board.ore == 1)
                all_locations = np.stack(
                    np.indices((map_size, map_size)), axis=-1
                ).reshape(-1, 2)
                ice_distances = taxi_distances(all_locations, ice_tile_locations)
                ice_distances = np.min(ice_distances, axis=-1)
                ice_distances = ice_distances.reshape(map_size, map_size)

                ore_distances = taxi_distances(all_locations, ore_tile_locations)
                ore_distances = np.min(ore_distances, axis=-1)
                ore_distances = ore_distances.reshape(map_size, map_size)

                resource_score = (
                    np.clip(ice_distances - 2, a_min=0, a_max=None)
                    + np.clip(ore_distances - 2, a_min=0, a_max=None) * 0.3
                )
                es_state.factory_score = resource_score + normalized_rubble_score
                es_state.rubble_score = rubble_score
                es_state.resource_score = resource_score

            # how many factories you have left to place
            factories_to_place = game_state.teams[self.player].factories_to_place
            my_turn_to_place = my_turn_to_place_factory(
                game_state.teams[self.player].place_first, step
            )
            # Debugging : Only place two factories
            # if len(game_state.factories[self.player]) >= 2:
            #     return {}

            new_factory_id = f"factory_{self._num_factories(game_state)}"
            spawn_action = {}
            if factories_to_place > 0 and my_turn_to_place:
                es_state = self.early_step_state
                map_size = self.env_cfg.map_size

                # place main factory
                if es_state.latest_main_factory is None:
                    # Build factory at the position with the lowest factory_score
                    factory_score = (
                        es_state.factory_score
                        + (obs["board"]["valid_spawns_mask"] == 0) * 1e9
                    )
                    spawn_loc = np.unravel_index(
                        np.argmin(factory_score), factory_score.shape
                    )

                    # Debug message
                    # factory_score_spawn = factory_score[spawn_loc[0], spawn_loc[1]]
                    # rubble_score_spawn = es_state.rubble_score[
                    #     spawn_loc[0], spawn_loc[1]
                    # ]

                    # print(
                    #     f"{self.player} placed {new_factory_id} at {spawn_loc}, factory score: {factory_score_spawn}, rubble score: {rubble_score_spawn}",
                    #     file=sys.stderr,
                    # )

                    plans = dict(water=150, metal=240)
                    if factories_to_place == 1:
                        plans = dict(water=150, metal=150)

                    es_state.latest_main_factory = new_factory_id
                    self.factory_states[new_factory_id] = FactoryState(
                        role=FactoryRole.MAIN
                    )
                    spawn_action = dict(spawn=spawn_loc, **plans)
                # place sub factory
                else:
                    all_locations = np.stack(
                        np.indices((map_size, map_size)), axis=-1
                    ).reshape(-1, 2)
                    main_factory_id = es_state.latest_main_factory
                    factory_pos = game_state.factories[self.player][main_factory_id].pos
                    factory_distance = taxi_distances(
                        all_locations, factory_pos
                    ).reshape(map_size, map_size)
                    rubble_score = es_state.rubble_score
                    normalized_rubble_score = (
                        rubble_score / np.max(rubble_score)
                    ) * 0.5
                    score = factory_distance + normalized_rubble_score
                    score += (obs["board"]["valid_spawns_mask"] == 0) * 1e9

                    ice_map = game_state.board.ice
                    ice_locs = np.argwhere(ice_map == 1)
                    ore_map = game_state.board.ore
                    ore_locs = np.argwhere(ore_map == 1)
                    main_factory_tiles = get_factory_tiles(factory_pos)
                    ice_tile_distances = taxi_distances(ice_locs, main_factory_tiles)
                    min_ice_tile_distances = np.min(ice_tile_distances)
                    arg_ice_candidates = np.argwhere(
                        ice_tile_distances == min_ice_tile_distances
                    )
                    ice_loc_candidates = set(
                        map(tuple, ice_locs[arg_ice_candidates[:, 0]])
                    )
                    ore_tile_distances = taxi_distances(ore_locs, main_factory_tiles)
                    min_ore_tile_distances = np.min(ore_tile_distances)
                    arg_ore_candidates = np.argwhere(
                        ore_tile_distances == min_ore_tile_distances
                    )
                    ore_loc_candidates = set(
                        map(tuple, ore_locs[arg_ore_candidates[:, 0]])
                    )
                    cost_map = self.get_move_cost_map(game_state)

                    plans = []
                    spawn_loc = None
                    for _ in range(10):
                        spawn_loc = np.unravel_index(
                            np.argmin(score, axis=None), score.shape
                        )
                        spawn_factory_locs = get_factory_tiles(spawn_loc)
                        for ice_loc in ice_loc_candidates:
                            for ore_loc in ore_loc_candidates:
                                empty_factory_locs = get_factory_tiles(factory_pos)
                                empty_factory_locs = remove_loc(
                                    empty_factory_locs, factory_pos
                                )

                                f2f_plan = self._find_factory_to_factory_route(
                                    cost_map,
                                    spawn_factory_locs,
                                    empty_factory_locs,
                                    [ice_loc, ore_loc],
                                )
                                if f2f_plan is None:
                                    continue
                                if f2f_plan.max_route_robots > 6:
                                    continue
                                closest_factory_loc = f2f_plan.destination
                                empty_factory_locs = remove_loc(
                                    empty_factory_locs, closest_factory_loc
                                )

                                best_f2i_plan = None
                                for empty_factory_loc in empty_factory_locs:
                                    f2i_plan = self._find_factory_to_route(
                                        cost_map,
                                        [empty_factory_loc],
                                        [ice_loc],
                                        [
                                            ore_loc,
                                            *f2f_plan.route.path,
                                        ],
                                    )
                                    if f2i_plan is None:
                                        continue
                                    if (
                                        best_f2i_plan is None
                                        or f2i_plan.route.cost
                                        < best_f2i_plan.route.cost
                                    ):
                                        best_f2i_plan = f2i_plan
                                if best_f2i_plan is None:
                                    continue
                                f2i_plan = best_f2i_plan
                                closest_ice_factory_tile = f2i_plan.source
                                empty_factory_locs = remove_loc(
                                    empty_factory_locs, closest_ice_factory_tile
                                )

                                best_f2o_plan = None
                                for empty_factory_loc in empty_factory_locs:
                                    f2o_plan = self._find_factory_to_route(
                                        cost_map,
                                        [empty_factory_loc],
                                        [ore_loc],
                                        [*f2f_plan.route.path, *f2i_plan.route.path],
                                    )
                                    if f2o_plan is None:
                                        continue
                                    if (
                                        best_f2o_plan is None
                                        or f2o_plan.route.cost
                                        < best_f2o_plan.route.cost
                                    ):
                                        best_f2o_plan = f2o_plan
                                if best_f2o_plan is None:
                                    continue
                                f2o_plan = best_f2o_plan
                                plans.append(
                                    (
                                        f2i_plan.route.cost
                                        + f2o_plan.route.cost
                                        + f2f_plan.route.cost,
                                        dict(ice=f2i_plan, ore=f2o_plan),
                                        dict(factory_to_factory=f2f_plan),
                                    )
                                )
                                break
                        if len(plans) == 0:
                            score[spawn_loc[0], spawn_loc[1]] = 1e9
                            continue
                        else:
                            break
                    if len(plans) == 0:
                        raise ValueError("No factory placement found")
                    _, main_factory_plans, sub_factory_plans = min(
                        plans, key=lambda x: x[0]
                    )
                    self.factory_states[
                        es_state.latest_main_factory
                    ].sub_factory = new_factory_id
                    self.factory_states[new_factory_id] = FactoryState(
                        role=FactoryRole.SUB,
                        main_factory=main_factory_id,
                        plans=sub_factory_plans,
                        ban_list=f2f_plan.route.path,
                    )
                    self.factory_states[main_factory_id].plans = main_factory_plans
                    self.factory_states[main_factory_id].ban_list = [
                        *f2f_plan.route.path,
                        *f2i_plan.route.path,
                        *f2o_plan.route.path,
                    ]
                    es_state.latest_main_factory = None
                    es_state.sub_factory_map[main_factory_id] = new_factory_id

                    spawn_action = {"spawn": spawn_loc, "water": 150, "metal": 60}

                    # Debug message
                    # print(
                    #     f"{self.player} placed {new_factory_id} at {spawn_loc}, main: {main_factory_id}",
                    #     file=sys.stderr,
                    # )
                    # print(
                    #     f"f2f: {sub_factory_plans['factory_to_factory'].route.path}",
                    #     file=sys.stderr,
                    # )
                    # print(
                    #     f"ice: {main_factory_plans['ice'].route.path}, ore: {main_factory_plans['ore'].route.path}",
                    #     file=sys.stderr,
                    # )

            return spawn_action

    def _get_factory_misc(self, factories):
        factory_centers, factory_units = [], []
        factory_ids = []
        for factory_id, factory in factories.items():
            factory_centers += [factory.pos]
            factory_units += [factory]
            factory_ids += [factory_id]

        return factory_centers, factory_units, factory_ids

    def handle_robot_actions(
        self, game_state, factory: Factory, unit: Unit, actions, factory_pickup_robots
    ):
        unit_id = unit.unit_id
        factory_inf_distance = norm(factory.pos - unit.pos, ord=np.inf)
        can_pickup_from_factory = factory_inf_distance <= 1
        can_transfer_to_factory = factory_inf_distance <= 2
        resource_type = unit.state.resource_type
        resource_plan = None
        if resource_type is not None:
            resource_plan = factory.state.plans[resource_type]
            if resource_type == "factory_to_factory":
                resource_id = resource_ids["water"]
            else:
                resource_id = resource_ids[resource_type]

        route = unit.state.following_route
        route_step = None

        if route is not None:
            if tuple(unit.pos) not in route.path:
                unit.state.state = UnitStateEnum.MOVING_TO_START
            else:
                route_step = route.path.index(tuple(unit.pos))

        def move_to(target, avoid=True) -> bool:
            if np.all(target == unit.pos):
                return True
            else:
                route_to_target = get_shortest_loop(
                    self.get_move_cost_map(game_state),
                    unit.pos,
                    target,
                    ban_list=factory.state.ban_list if avoid else [],
                )
                if route_to_target is None:
                    raise ValueError("No route found")
                direction = direction_to(unit.pos, route_to_target.path[1])
                move_cost = unit.move_cost(game_state, direction)
                if (
                    move_cost is not None
                    and unit.power >= move_cost + unit.action_queue_cost(game_state)
                ):
                    actions[unit_id] = [unit.move(direction, repeat=0, n=1)]
                return False

        def follow_route_to(target) -> bool:
            target_step = route.path.index(tuple(target))
            if target_step > route_step:
                direction = direction_to(unit.pos, route.path[route_step + 1])
            elif target_step < route_step:
                direction = direction_to(unit.pos, route.path[route_step - 1])
            else:
                return True
            move_cost = unit.move_cost(game_state, direction)
            if (
                move_cost is not None
                and unit.power >= move_cost + unit.action_queue_cost(game_state)
            ):
                actions[unit_id] = [unit.move(direction)]
            return False

        for _ in range(len(UnitStateEnum)):
            if unit.state.state == UnitStateEnum.INITIAL:
                if route is not None:
                    # if unit is not on route, state is already MOVING_TO_START
                    # unit is always on route
                    unit.state.state = UnitStateEnum.MOVING_TO_TARGET
                    continue
                break
            if unit.state.state == UnitStateEnum.MOVING_TO_START:
                arrived = move_to(route.path[0])
                if arrived:
                    unit.state.state = UnitStateEnum.PICKING_RESOURCE
                    continue
                break
            if unit.state.state == UnitStateEnum.PICKING_RESOURCE:
                if can_pickup_from_factory:
                    target_power = (
                        unit.unit_cfg.INIT_POWER
                    )  # TODO calculate target power
                    if unit.power < target_power:
                        if target_power - unit.power <= factory.power:
                            pickup_power = target_power - unit.power
                            actions[unit_id] = [unit.pickup(4, pickup_power)]
                    else:
                        unit.state.state = UnitStateEnum.MOVING_TO_TARGET
                        continue
                else:
                    unit.state.state = UnitStateEnum.MOVING_TO_FACTORY
                    continue
                break
            if unit.state.state == UnitStateEnum.MOVING_TO_TARGET:
                arrived = follow_route_to(unit.state.target_pos)
                if arrived:
                    unit.state.state = UnitStateEnum.PERFORMING_ROLE
                    continue
                break
            if unit.state.state == UnitStateEnum.PERFORMING_ROLE:
                if unit.state.role.is_stationary:
                    odd = (game_state.env_steps % 2) ^ (route_step % 2)
                    if odd:
                        if unit.state.role.is_miner:
                            if (
                                unit.power
                                >= unit.dig_cost(game_state)
                                + unit.action_queue_cost(game_state) * 2
                            ):
                                actions[unit_id] = [unit.dig(repeat=0, n=1)]
                        elif (
                            resource_type == "factory_to_factory"
                            and route_step == len(route) - 1
                        ):
                            if factory.state.main_factory is not None:
                                charge_threshold = (
                                    unit.action_queue_cost(game_state) * 2
                                    + unit.unit_cfg.INIT_POWER
                                )
                                charge_to = int(unit.unit_cfg.INIT_POWER * 1.5)
                                if unit.power < charge_threshold:
                                    transfer_power = charge_to - unit.power
                                    if factory.power >= transfer_power:
                                        if unit.power >= unit.action_queue_cost(
                                            game_state
                                        ):
                                            actions[unit_id] = [
                                                unit.pickup(4, transfer_power)
                                            ]
                                elif (
                                    unit.power >= unit.action_queue_cost(game_state) * 2
                                ):
                                    actions[unit_id] = [unit.pickup(resource_id, 2)]
                        else:
                            min_power = (
                                unit.action_queue_cost(game_state) * 2
                                + unit.unit_cfg.INIT_POWER
                            )
                            if unit.power > min_power:
                                transfer_power = unit.power - min_power
                                direction = direction_to(
                                    unit.pos, route.path[route_step + 1]
                                )
                                actions[unit_id] = [
                                    unit.transfer(direction, 4, transfer_power)
                                ]
                    else:
                        resource_threshold = 6  # TODO calculate resource threshold
                        if unit.unit_type == "HEAVY":
                            resource_threshold *= 10

                        if (
                            unit.state.role.is_miner
                            and unit.cargo.from_id(resource_id) < resource_threshold
                        ):
                            # extra mining!!
                            if (
                                unit.power
                                >= unit.dig_cost(game_state)
                                + unit.action_queue_cost(game_state) * 2
                            ):
                                actions[unit_id] = [unit.dig(repeat=0, n=1)]
                        elif route_step == 0:
                            pickup_amount = 50  # TODO calculate pickup amount
                            if unit.power >= unit.action_queue_cost(game_state) * 2:
                                actions[unit_id] = [unit.pickup(4, pickup_amount)]
                        else:
                            direction = direction_to(
                                unit.pos, route.path[route_step - 1]
                            )
                            if unit.cargo.from_id(resource_id) > 0:
                                if unit.power >= unit.action_queue_cost(game_state):
                                    actions[unit_id] = [
                                        unit.transfer(
                                            direction,
                                            resource_id,
                                            unit.cargo.from_id(resource_id),
                                        )
                                    ]
                else:
                    if resource_type in ["ice", "ore"]:
                        resource_threshold = 6  # TODO calculate resource threshold
                        if unit.unit_type == "HEAVY":
                            resource_threshold *= 10
                    else:
                        resource_threshold = 2
                    if unit.cargo.from_id(resource_id) < resource_threshold:
                        if unit.state.role.is_miner:
                            if (
                                unit.power
                                >= unit.dig_cost(game_state)
                                + unit.action_queue_cost(game_state) * 2
                            ):
                                actions[unit_id] = [unit.dig(repeat=0, n=1)]
                        elif (
                            resource_type == "factory_to_factory"
                            and route_step == len(route) - 1
                        ):
                            if unit.power >= unit.action_queue_cost(game_state):
                                actions[unit_id] = [unit.pickup(resource_id, 2)]
                        else:
                            # only have to transfer power
                            min_power = (
                                unit.action_queue_cost(game_state) * 2
                                + unit.unit_cfg.INIT_POWER
                            )
                            if unit.power > min_power:
                                transfer_power = unit.power - min_power
                                direction = direction_to(
                                    unit.pos, route.path[route_step + 1]
                                )
                                actions[unit_id] = [
                                    unit.transfer(direction, 4, transfer_power)
                                ]
                    else:
                        unit.state.state = UnitStateEnum.MOVING_TO_FACTORY
                        continue
                break
            if unit.state.state == UnitStateEnum.MOVING_TO_FACTORY:
                arrived = follow_route_to(route.start)
                if arrived:
                    unit.state.state = UnitStateEnum.DROPPING_RESOURCE
                    continue
                break
            if unit.state.state == UnitStateEnum.DROPPING_RESOURCE:
                if can_transfer_to_factory:
                    if unit.cargo.from_id(resource_id) > 0:
                        if unit.power >= unit.action_queue_cost(game_state):
                            actions[unit_id] = [
                                unit.transfer(
                                    0, resource_id, unit.cargo.from_id(resource_id)
                                )
                            ]
                    else:
                        unit.state.state = UnitStateEnum.PICKING_RESOURCE
                        continue
                else:
                    unit.state.state = UnitStateEnum.MOVING_TO_FACTORY
                    continue
                break
        else:
            raise ValueError("Unit State Machine failed to break")
        return actions

    def _assign_role_to_unit(self, unit, factory):
        mission = unit.state.mission
        mission_robots = factory.state.robot_missions[mission]
        pipe_robots = []
        if mission == UnitMission.PIPE_MINE_ICE:
            pipe_robots = factory.state.robot_missions[UnitMission.PIPE_FACTORY_TO_ICE]
        elif mission == UnitMission.PIPE_MINE_ORE:
            pipe_robots = factory.state.robot_missions[UnitMission.PIPE_FACTORY_TO_ORE]

        if mission in [UnitMission.PIPE_MINE_ICE, UnitMission.PIPE_MINE_ORE]:
            if len(pipe_robots) > 0:
                unit.state.set_role(UnitRole.STATIONARY_MINER)
            else:
                unit.state.set_role(UnitRole.MINER_TRANSPORTER)
            unit.state.set_target_pos(
                factory.state.plans[mission.resource_type].destination
            )
        elif mission in [
            UnitMission.PIPE_FACTORY_TO_ICE,
            UnitMission.PIPE_FACTORY_TO_ORE,
        ]:
            route = None
            miner_robots = None
            if mission == UnitMission.PIPE_FACTORY_TO_ICE:
                route = factory.state.plans["ice"].route
                miner_robots = factory.state.robot_missions[UnitMission.PIPE_MINE_ICE]
            elif mission == UnitMission.PIPE_FACTORY_TO_ORE:
                route = factory.state.plans["ore"].route
                miner_robots = factory.state.robot_missions[UnitMission.PIPE_MINE_ORE]
            miner_robot_id = miner_robots[0]  # There is only one miner robot
            miner_robot_state = self.unit_states[miner_robot_id]
            miner_robot_state.set_role(UnitRole.STATIONARY_MINER)
            miner_robot_state.set_target_pos(route.path[-1])
            pipe_is_full = len(mission_robots) + 1 == len(route)
            for i, robot_id in enumerate(mission_robots):
                robot_state = self.unit_states[robot_id]
                if i == len(mission_robots) - 1 and not pipe_is_full:
                    robot_state.set_role(UnitRole.TRANSPORTER)
                else:
                    robot_state.set_role(UnitRole.STATIONARY_TRANSPORTER)
                robot_state.set_target_pos(route.path[-2 - i])

        elif mission == UnitMission.PIPE_FACTORY_TO_FACTORY:
            route = factory.state.plans["factory_to_factory"].route
            pipe_is_full = len(mission_robots) == len(route)
            for i, robot_id in enumerate(mission_robots):
                robot_state = self.unit_states[robot_id]
                if i == len(mission_robots) - 1 and not pipe_is_full:
                    robot_state.set_role(UnitRole.TRANSPORTER)
                else:
                    robot_state.set_role(UnitRole.STATIONARY_TRANSPORTER)
                robot_state.set_target_pos(route.path[-1 - i])

        elif mission == UnitMission.DIG_RUBBLE:
            unit.state.set_role(UnitRole.RUBBLE_DIGGER)
        else:
            raise ValueError("Invalid mission")

    def _add_unit_to_factory(self, unit, factory: Factory):
        mission = None
        if factory.state.role == FactoryRole.SUB:
            if (
                len(factory.state.robot_missions[UnitMission.PIPE_FACTORY_TO_FACTORY])
                < factory.state.plans["factory_to_factory"].max_route_robots
            ):
                mission = UnitMission.PIPE_FACTORY_TO_FACTORY
            else:
                mission = UnitMission.DIG_RUBBLE
        else:
            # TODO check if we can override miner (if unit.unit_type == "HEAVY")
            if len(factory.state.robot_missions[UnitMission.PIPE_MINE_ICE]) == 0:
                mission = UnitMission.PIPE_MINE_ICE
            elif len(factory.state.robot_missions[UnitMission.PIPE_MINE_ORE]) == 0:
                mission = UnitMission.PIPE_MINE_ORE
            elif (
                len(factory.state.robot_missions[UnitMission.PIPE_FACTORY_TO_ICE]) + 1
                < factory.state.plans["ice"].max_route_robots
            ):
                mission = UnitMission.PIPE_FACTORY_TO_ICE
            elif (
                len(factory.state.robot_missions[UnitMission.PIPE_FACTORY_TO_ORE]) + 1
                < factory.state.plans["ore"].max_route_robots
            ):
                mission = UnitMission.PIPE_FACTORY_TO_ORE
            else:
                mission = UnitMission.DIG_RUBBLE

        unit.state.mission = mission
        factory.state.robot_missions[mission] += [unit.unit_id]
        unit.state.resource_type = mission.resource_type
        if mission.resource_type is not None:
            unit.state.following_route = factory.state.plans[
                mission.resource_type
            ].route
        if mission == UnitMission.PIPE_FACTORY_TO_FACTORY:
            unit.state.following_route = factory.state.plans["factory_to_factory"].route

        self._assign_role_to_unit(unit, factory)

    def _fetch_states(self, game_state: GameState):
        units = game_state.units[self.player]
        factories = game_state.factories[self.player]

        for unit_id, unit in units.items():
            if unit_id in self.unit_states:
                unit.state = self.unit_states[unit_id]

        for factory_id, factory in factories.items():
            if factory_id in self.factory_states:
                factory.state = self.factory_states[factory_id]

    def get_move_cost_map(self, game_state: GameState):
        turn, cost_map = self.move_cost_map
        if game_state.env_steps != turn:
            cost_map = (
                game_state.board.rubble
                * self.env_cfg.ROBOTS["LIGHT"].RUBBLE_MOVEMENT_COST
                + self.env_cfg.ROBOTS["LIGHT"].MOVE_COST
            )
            self.move_cost_map = game_state.env_steps, cost_map
        return cost_map

    def _find_route(self, cost_map, source_locs, destination_locs, ban_list=[]):
        if len(source_locs) > 1 or len(destination_locs) > 1:
            distances = taxi_distances(destination_locs, source_locs)
            argmin_sub_factory, argmin_factory = np.unravel_index(
                np.argmin(distances), distances.shape
            )
            closest_destination_loc = destination_locs[argmin_sub_factory]
            closest_source_loc = source_locs[argmin_factory]
        else:
            closest_destination_loc = destination_locs[0]
            closest_source_loc = source_locs[0]

        route = get_shortest_loop(
            cost_map,
            closest_source_loc,
            closest_destination_loc,
            ban_list=ban_list,
        )
        if route is None:
            return None
        return Plan(
            destination=closest_destination_loc,
            source=closest_source_loc,
            route=route,
            max_route_robots=len(route),
        )

    def _find_factory_to_factory_route(
        self, cost_map, empty_factory_locs, sub_factory_locs, ban_list=[]
    ):
        route = self._find_route(
            cost_map, empty_factory_locs, sub_factory_locs, ban_list
        )
        if route is None:
            return None
        return TransmitPlan.from_plan(route)

    def _find_factory_to_route(
        self, cost_map, empty_factory_locs, resource_locs, ban_list=[]
    ):
        route = self._find_route(cost_map, empty_factory_locs, resource_locs, ban_list)
        if route is None:
            return None
        return ResourcePlan.from_plan(route)

    def _register_factories(self, game_state):
        """
        initialize self.factory_states
        """

        ice_map = game_state.board.ice
        ice_locs = np.argwhere(ice_map == 1)
        ore_map = game_state.board.ore
        ore_locs = np.argwhere(ore_map == 1)

        factories = game_state.factories[self.player]
        cost_map = self.get_move_cost_map(game_state)

        for factory_id, factory in factories.items():
            factory_state = self.factory_states[factory_id]
            if factory_state.plans is not None:
                continue

            empty_factory_locs = get_factory_tiles(factory.pos)

            ice_tile_distances = taxi_distances(ice_locs, empty_factory_locs)
            argmin_ice_tile, argmin_factory_tile = np.unravel_index(
                np.argmin(ice_tile_distances), ice_tile_distances.shape
            )
            closest_ice_tile = ice_locs[argmin_ice_tile]
            closest_ice_factory_tile = empty_factory_locs[argmin_factory_tile]

            empty_indices = np.any(
                empty_factory_locs != closest_ice_factory_tile, axis=-1
            ).nonzero()
            empty_factory_locs = empty_factory_locs[empty_indices]

            ore_tile_distances = taxi_distances(ore_locs, empty_factory_locs)
            argmin_ore_tile, argmin_factory_tile = np.unravel_index(
                np.argmin(ore_tile_distances), ore_tile_distances.shape
            )
            closest_ore_tile = ore_locs[argmin_ore_tile]
            closest_ore_factory_tile = empty_factory_locs[argmin_factory_tile]

            ice_route = get_shortest_loop(
                cost_map,
                closest_ice_factory_tile,
                closest_ice_tile,
                ban_list=[closest_ore_tile, closest_ore_factory_tile],
            )
            if ice_route is None:
                raise ValueError("No ice route found")

            ore_route = get_shortest_loop(
                cost_map,
                closest_ore_factory_tile,
                closest_ore_tile,
                ban_list=ice_route.path,
            )
            if ore_route is None:
                raise ValueError("No ore route found")

            factory_state.plans = dict(
                ice=ResourcePlan(
                    destination=closest_ice_tile,
                    source=closest_ice_factory_tile,
                    route=ice_route,
                    max_route_robots=len(ice_route),
                ),
                ore=ResourcePlan(
                    destination=closest_ore_tile,
                    source=closest_ore_factory_tile,
                    route=ore_route,
                    max_route_robots=len(ore_route),
                ),
            )
            factory_state.ban_list = [*ice_route.path, *ore_route.path]
            # Debug
            # print(
            #     f"{factory_id}: ice: {ice_route.path}, ore: {ore_route.path}",
            #     file=sys.stderr,
            # )

            factory.state = self.factory_states[factory_id]

    def _unregister_factories(self, game_state):
        # Remove robots from factories if factory is destroyed
        factory_ids_to_destroy = []
        factory_ids = game_state.factories[self.player].keys()
        for factory_id, factory_state in self.factory_states.items():
            if factory_id not in factory_ids:
                factory_ids_to_destroy.append(factory_id)
                for role, role_robots in factory_state.robot_missions.items():
                    for unit_id in role_robots:
                        del self.unit_states[unit_id]
                        game_state.units[self.player][unit_id].state = None
        for factory_id in factory_ids_to_destroy:
            if self.factory_states[factory_id].role == FactoryRole.MAIN:
                sub_factory_id = self.factory_states[factory_id].sub_factory
                if sub_factory_id is not None and sub_factory_id in factory_ids:
                    self.factory_states[sub_factory_id].main_factory = None
                    # TODO reassign sub factory as main factory
            else:
                main_factory_id = self.factory_states[factory_id].main_factory
                if main_factory_id in factory_ids:
                    self.factory_states[main_factory_id].sub_factory = None
            del self.factory_states[factory_id]
        return game_state

    def _register_units(self, units, factories, factory_centers, factory_ids):
        # Register robots to factories if not registered
        for unit_id, unit in units.items():
            if unit_id not in self.unit_states:
                self.unit_states[unit_id] = UnitState()
                unit.state = self.unit_states[unit_id]
                factory_distances = taxi_dist(factory_centers, unit.pos)
                factory_id = factory_ids[np.argmin(factory_distances)]
                if np.any(factories[factory_id].pos != unit.pos):
                    # Unit reassigned to factory
                    if factories[factory_id].state.role == FactoryRole.SUB:
                        main_factory_id = factories[factory_id].state.main_factory
                        if (
                            main_factory_id is not None
                            and main_factory_id in factory_ids
                        ):
                            factory_id = main_factory_id
                unit.state.owner = factory_id
                self._add_unit_to_factory(unit, factories[factory_id])

    def _unregister_units(self, units, factories):
        unit_ids_to_destroy = []
        for unit_id, unit_state in self.unit_states.items():
            if unit_id not in units:
                unit_ids_to_destroy.append(unit_id)
                factory_id = unit_state.owner
                factory_state = self.factory_states[factory_id]
                mission = unit_state.mission
                factory_state.robot_missions[mission].remove(unit_id)
                if len(factory_state.robot_missions[mission]) > 0:
                    unit_id = factory_state.robot_missions[mission][-1]
                    unit = units[unit_id]
                    self._assign_role_to_unit(unit, factories[factory_id])
                else:
                    ...  # TODO more sophisticated role reassignment

        for unit_id in unit_ids_to_destroy:
            del self.unit_states[unit_id]

    def act(self, step: int, obs, remainingOverageTime: int = 60):
        actions = dict()
        game_state = obs_to_game_state(step, self.env_cfg, obs)
        cycle = game_state.real_env_steps // self.env_cfg.CYCLE_LENGTH
        turn_in_cycle = game_state.real_env_steps % self.env_cfg.CYCLE_LENGTH
        is_day = turn_in_cycle < self.env_cfg.DAY_LENGTH
        is_night = not is_day
        remaining_days = 0 if is_night else (self.env_cfg.DAY_LENGTH - turn_in_cycle)
        remaining_nights = 0 if is_day else (self.env_cfg.CYCLE_LENGTH - turn_in_cycle)

        if game_state.real_env_steps == 0:
            self._register_factories(game_state)
        self._fetch_states(game_state)

        factories = game_state.factories[self.player]

        factory_centers, factory_units, factory_ids = self._get_factory_misc(factories)
        units = game_state.units[self.player]

        self._unregister_factories(game_state)
        self._unregister_units(units, factories)
        self._register_units(units, factories, factory_centers, factory_ids)

        for factory_id, factory in factories.items():
            # handle action of robots bound to factories
            factory_pickup_robots = 0

            for mission, mission_robot_ids in factory.state.robot_missions.items():
                for unit_id in mission_robot_ids:
                    unit = units[unit_id]
                    factory_inf_distance = norm(factory.pos - unit.pos, ord=np.inf)
                    can_pickup_from_factory = factory_inf_distance <= 1
                    if (
                        can_pickup_from_factory
                        and self.unit_states[unit_id].state
                        > UnitStateEnum.MOVING_TO_START
                    ):
                        if unit.unit_type == "LIGHT":
                            factory_pickup_robots += 1
                        else:
                            factory_pickup_robots += 10
            for mission, mission_robot_ids in factory.state.robot_missions.items():
                for unit_id in mission_robot_ids:
                    unit = units[unit_id]
                    actions = self.handle_robot_actions(
                        game_state, factory, unit, actions, factory_pickup_robots
                    )

            # handle factory actions
            if factory.state.role == FactoryRole.MAIN:
                remaining_steps = (
                    self.env_cfg.max_episode_length - game_state.real_env_steps
                )
                water_cost = factory.water_cost(game_state)
                spreads = remaining_steps / self.env_cfg.MIN_LICHEN_TO_SPREAD
                multiple = (
                    spreads * (spreads + 1) * (2 * spreads + 1) / 6
                ) * self.env_cfg.MIN_LICHEN_TO_SPREAD
                estimated_water_cost = water_cost * multiple
                if (estimated_water_cost + remaining_steps) <= factory.cargo.water:
                    actions[factory_id] = factory.water()
                elif remaining_steps == 1 and water_cost < factory.cargo.water:
                    actions[factory_id] = factory.water()
                else:  # or build robots
                    robot_ids = sum(factory.state.robot_missions.values(), start=[])
                    light_robots = [
                        robot_id
                        for robot_id in robot_ids
                        if units[robot_id].unit_type == "LIGHT"
                    ]
                    heavy_robots = [
                        robot_id
                        for robot_id in robot_ids
                        if units[robot_id].unit_type == "HEAVY"
                    ]
                    required_transmitters = sum(
                        map(
                            lambda plan: plan.max_route_robots,
                            factory.state.plans.values(),
                        )
                    )
                    if len(robot_ids) < 2:
                        if factory.can_build_heavy(
                            game_state
                        ):  # TODO check if we can build one more light if len(robots) == 0
                            actions[factory_id] = factory.build_heavy()
                        elif factory.can_build_light(game_state):
                            actions[factory_id] = factory.build_light()
                        else:
                            ...  # Doomed
                    elif len(robot_ids) < required_transmitters:
                        if factory.can_build_light(game_state):
                            actions[factory_id] = factory.build_light()
                    else:
                        ...
                        # if factory.can_build_heavy(game_state):
                        #     actions[factory_id] = factory.build_heavy()

            elif factory.state.role == FactoryRole.SUB:
                required_transmitters = factory.state.plans[
                    "factory_to_factory"
                ].max_route_robots
                robot_ids = sum(factory.state.robot_missions.values(), start=[])
                if len(robot_ids) < required_transmitters:
                    if factory.can_build_light(game_state):
                        actions[factory_id] = factory.build_light()
            else:
                raise ValueError(f"Invalid factory role {factory.state.role}")

        actions = stop_movement_collisions(
            obs, game_state, self.env_cfg, self.player, actions, self.unit_states
        )
        return actions


def main(env, agents, steps, seed):
    # reset our env
    obs, _ = env.reset(seed=seed)
    np.random.seed(0)

    step = 0
    # Note that as the environment has two phases, we also keep track a value called
    # `real_env_steps` in the environment state. The first phase ends once `real_env_steps` is 0 and used below

    # iterate until phase 1 ends
    while env.state.real_env_steps < 0:
        if step >= steps:
            break
        actions = {}
        for player in env.agents:
            o = obs[player]
            a = agents[player].early_setup(step, o)
            actions[player] = a
        step += 1
        obs, rewards, terminations, truncations, infos = env.step(actions)
        dones = {k: terminations[k] or truncations[k] for k in terminations}
    done = False
    while not done:
        if step >= steps:
            break
        actions = {}
        for player in env.agents:
            o = obs[player]
            a = agents[player].act(step, o)
            actions[player] = a
        step += 1
        obs, rewards, terminations, truncations, infos = env.step(actions)
        dones = {k: terminations[k] or truncations[k] for k in terminations}
        done = dones["player_0"] and dones["player_1"]


if __name__ == "__main__":
    from luxai_s2.env import LuxAI_S2

    env = LuxAI_S2()  # create the environment object
    agents = {
        player: Agent(player, env.state.env_cfg) for player in ["player_0", "player_1"]
    }
    main(env, agents, 100, 101)
