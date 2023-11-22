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
from lux.water_costs import water_costs


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
        self.enemy_factory_tiles = {}
        self.enemy_factory_tile_ban_list = []
        self.global_ban_list = []

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
            water_left = game_state.teams[self.player].water
            if water_left == 0:
                factories_to_place = 0
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
                                closest_ore_factory_tile = f2o_plan.source
                                empty_factory_locs = remove_loc(
                                    empty_factory_locs, closest_ore_factory_tile
                                )
                                plans.append(
                                    (
                                        f2i_plan.route.cost
                                        + f2o_plan.route.cost
                                        + f2f_plan.route.cost,
                                        dict(ice=f2i_plan, ore=f2o_plan),
                                        dict(factory_to_factory=f2f_plan),
                                        empty_factory_locs
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
                    _, main_factory_plans, sub_factory_plans, empty_factory_locs = min(
                        plans, key=lambda x: x[0]
                    )
                    self.factory_states[
                        es_state.latest_main_factory
                    ].sub_factory = new_factory_id
                    self.factory_states[new_factory_id] = FactoryState(
                        role=FactoryRole.SUB,
                        main_factory=main_factory_id,
                        plans=sub_factory_plans,
                        ban_list=f2f_plan.route.path.copy(),
                        empty_factory_locs = remove_loc(get_factory_tiles(spawn_loc), sub_factory_plans['factory_to_factory'].route.start)
                    )
                    self.factory_states[main_factory_id].plans = main_factory_plans
                    self.factory_states[main_factory_id].ban_list = [
                        *main_factory_plans['ice'].route.path,
                        *main_factory_plans['ore'].route.path,
                    ]
                    self.factory_states[main_factory_id].empty_factory_locs = empty_factory_locs
                    es_state.latest_main_factory = None
                    es_state.sub_factory_map[main_factory_id] = new_factory_id
                    res = dict(water=150, metal=60)
                    if factories_to_place % 2 == 0:
                        more_res = self.env_cfg.ROBOTS["HEAVY"].METAL_COST + sub_factory_plans['factory_to_factory'].max_route_robots
                        if (more_res - 60) + 300 * ((factories_to_place - 1) // 2) <= game_state.teams[self.player].metal:
                            res = dict(water=150, metal=more_res)
                    if factories_to_place == 2:  # Last main-sub factory pair
                        res = dict(water=game_state.teams[self.player].water, metal=game_state.teams[self.player].metal)
                    spawn_action = {"spawn": spawn_loc, **res}

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
        self, game_state: GameState, factory: Factory, unit: Unit, actions, factory_pickup_robots
    ):
        if unit.state.mission == UnitMission.NONE:
            if len(unit.action_queue) > 0:
                return actions
            if game_state.board.rubble[unit.pos[0], unit.pos[1]] > 0:
                if unit.power >= unit.unit_cfg.DIG_COST + unit.action_queue_cost(game_state):
                    time_to_dig = (unit.power - unit.action_queue_cost(game_state)) // unit.unit_cfg.DIG_COST
                    time_to_dig = min(time_to_dig, int(np.ceil(game_state.board.rubble[unit.pos[0], unit.pos[1]] / unit.unit_cfg.DIG_RUBBLE_REMOVED)))
                    actions[unit.unit_id] = [unit.dig(repeat=0, n=time_to_dig)]
                else:
                    ... # TODO: Handle recharging
            return actions
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
                if factory.state.main_factory is not None:
                    main_factory = game_state.factories[self.player][factory.state.main_factory]
                    if all([
                        len(main_factory.state.robot_missions[UnitMission.PIPE_FACTORY_TO_ICE]) + 1 == len(main_factory.state.plans["ice"].route),
                        len(main_factory.state.robot_missions[UnitMission.PIPE_FACTORY_TO_ORE]) + 1 == len(main_factory.state.plans["ore"].route),
                        factory.cargo.water > 50,
                    ]):
                        if len(factory.state.robot_missions[UnitMission.DIG_RUBBLE]) < factory.state.MAX_DIGGER:
                            resource_id = resource_ids["metal"]
                        elif main_factory.power > 500:
                            resource_id = resource_ids["power"]
                    
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
                unit.state.route_cache = None
                return True
            else:
                if unit.state.route_cache is not None and unit.state.route_cache.end == tuple(target):
                    route_to_target = unit.state.route_cache
                    if tuple(unit.pos) in route_to_target.path:
                        route_step = route_to_target.path.index(tuple(unit.pos))
                        unit.state.route_cache.path = route_to_target.path[route_step:]
                        unit.state.route_cache.start = tuple(unit.pos)
                    else:
                        unit.state.route_cache = None
                else:
                    unit.state.route_cache = None
                if unit.state.route_cache is None:
                    if unit.state.role == UnitRole.RUBBLE_DIGGER:
                        ban_list = self.global_ban_list.copy()
                    else:
                        ban_list = factory.state.ban_list.copy()
                        if factory.state.main_factory is not None:
                            ban_list += game_state.factories[self.player][factory.state.main_factory].state.ban_list
                        if factory.state.sub_factory is not None:
                            ban_list += game_state.factories[self.player][factory.state.sub_factory].state.ban_list
                        ban_list += self.enemy_factory_tile_ban_list
                    route_to_target = get_shortest_loop(
                        self.get_move_cost_map(game_state),
                        unit.pos,
                        target,
                        ban_list=ban_list if avoid else [],
                    )
                    if route_to_target is None:
                        raise ValueError("No route found")
                    unit.state.route_cache = route_to_target
                else:
                    route_to_target = unit.state.route_cache
                direction = direction_to(unit.pos, route_to_target.path[1])
                move_cost = unit.move_cost(game_state, direction)
                if (
                    move_cost is not None
                    and unit.power >= move_cost + unit.action_queue_cost(game_state)
                ):
                    actions[unit_id] = [unit.move(direction, repeat=0, n=1)]
                elif move_cost is None:
                    unit.state.route_cache = None
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

        def find_next_rubble_to_mine():
            ban_list = self.global_ban_list.copy()
            # ban_list = factory.state.ban_list.copy()
            # if factory.state.main_factory is not None:
            #     ban_list += game_state.factories[self.player][factory.state.main_factory].state.ban_list
            rubble_map = game_state.board.rubble.copy()
            for ban_loc in ban_list:
                rubble_map[ban_loc[0], ban_loc[1]] = 0
            for robot_id in factory.state.robot_missions[UnitMission.DIG_RUBBLE]:
                unit_state = self.unit_states[robot_id]
                if unit_state.state in [
                    UnitStateEnum.MOVING_TO_RUBBLE,
                    UnitStateEnum.DIGGING_RUBBLE,
                ]:
                    if unit_state.target_pos is not None:
                        rubble_map[unit_state.target_pos] = 0
            rubble_locs = np.argwhere(rubble_map > 0)
            distances = taxi_dist(factory.pos, rubble_locs)
            return rubble_locs[np.argmin(distances)]

        for _ in range(len(UnitStateEnum)):
            if unit.state.state == UnitStateEnum.INITIAL:
                if unit.state.role == UnitRole.RUBBLE_DIGGER:
                    unit.state.state = UnitStateEnum.MOVING_TO_RUBBLE
                    continue
                if route is not None:
                    # if unit is not on route, state is already MOVING_TO_START
                    # unit is always on route
                    unit.state.state = UnitStateEnum.MOVING_TO_TARGET
                    continue
                break
            if unit.state.state == UnitStateEnum.MOVING_TO_START:
                arrived = move_to(route.path[0])
                if arrived:
                    if unit.state.role.is_stationary:
                        unit.state.state = UnitStateEnum.MOVING_TO_TARGET
                    else:
                        unit.state.state = UnitStateEnum.PICKING_RESOURCE
                    continue
                break
            if unit.state.state == UnitStateEnum.PICKING_RESOURCE:
                if can_pickup_from_factory:
                    target_power = 150
                    if factory.state.main_factory is not None and len(
                        factory.state.robot_missions[
                            UnitMission.PIPE_FACTORY_TO_FACTORY
                        ]
                    ) == len(route):
                        # Has to build a digger robot, so transfer less power
                        main_factory = game_state.factories[self.player][
                                factory.state.main_factory
                            ]
                        if main_factory.state.ore_disabled and main_factory.power > 500:
                            target_power = 0
                    
                    if unit.power < target_power:
                        if target_power - unit.power + unit.action_queue_cost(game_state)<= factory.power:
                            pickup_power = target_power - unit.power + unit.action_queue_cost(game_state)
                            actions[unit_id] = [unit.pickup(4, min(unit.unit_cfg.BATTERY_CAPACITY, pickup_power))]
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
                    unit.state.waiting_for = 0
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
                                charge_threshold = unit.action_queue_cost(game_state) * 5
                                charge_to = int(unit.unit_cfg.INIT_POWER * 1.5)
                                if unit.power < charge_threshold:
                                    transfer_power = charge_to - unit.power + unit.action_queue_cost(game_state)
                                    if factory.power >= transfer_power:
                                        if unit.power >= unit.action_queue_cost(
                                            game_state
                                        ):
                                            actions[unit_id] = [
                                                unit.pickup(4, min(unit.unit_cfg.BATTERY_CAPACITY, transfer_power))
                                            ]
                                elif (
                                    unit.power >= unit.action_queue_cost(game_state) * 2
                                ):
                                    main_factory = factory.state.main_factory
                                    main_factory_cargo = game_state.factories[
                                        self.player
                                    ][main_factory].cargo
                                    if resource_id == resource_ids["water"]:
                                        pickup_amount = main_factory_cargo.water - min(10, game_state.remaining_steps)
                                        pickup_amount = min(pickup_amount, self.env_cfg.ROBOTS["LIGHT"].CARGO_SPACE)
                                        pickup_amount = min(pickup_amount, unit.unit_cfg.CARGO_SPACE - unit.cargo.water)
                                        if pickup_amount > 0:
                                            actions[unit_id] = [unit.pickup(resource_id, pickup_amount)]
                                    elif resource_id == resource_ids["metal"]:
                                        if main_factory_cargo.metal > 0:
                                            actions[unit_id] = [unit.pickup(resource_id, min(main_factory_cargo.metal, self.env_cfg.ROBOTS["LIGHT"].CARGO_SPACE))]
                                    elif resource_id == resource_ids["power"]:
                                        if game_state.factories[self.player][main_factory].power > 0:
                                            actions[unit_id] = [unit.pickup(resource_id, min(game_state.factories[self.player][main_factory].power, self.env_cfg.ROBOTS["LIGHT"].BATTERY_CAPACITY))]
                        else:
                            if resource_id != resource_ids["power"]:    
                                min_power = unit.action_queue_cost(game_state) * 4
                                pipe_full = game_state.units[self.player][factory.state.robot_missions[unit.state.mission][-1]].state.role.is_stationary
                                if game_state.board.rubble[unit.pos[0], unit.pos[1]] > 0 and pipe_full:
                                    min_power += unit.unit_cfg.DIG_COST  # for rubble mining
                                elif factory.state.main_factory is not None and pipe_full:
                                    main_factory = game_state.factories[self.player][
                                            factory.state.main_factory
                                        ]
                                    if main_factory.power > 600 and route_step == len(route) - 2:
                                        min_power = unit.unit_cfg.BATTERY_CAPACITY
                                if unit.power > min_power:
                                    transfer_power = unit.power - min_power
                                    direction = direction_to(
                                        unit.pos, route.path[route_step + 1]
                                    )
                                    actions[unit_id] = [
                                        unit.transfer(direction, 4, transfer_power)
                                    ]
                    else:  # EVEN CASE
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
                            pickup_amount = 2 * 50  # TODO calculate pickup amount
                            if factory.state.main_factory is not None and len(
                                factory.state.robot_missions[
                                    UnitMission.PIPE_FACTORY_TO_FACTORY
                                ]
                            ) == len(route):
                                # Has to build a digger robot, so transfer less power
                                main_factory = game_state.factories[self.player][
                                        factory.state.main_factory
                                    ]
                                if main_factory.power > 600:
                                     # cycle = game_state.real_env_steps // self.env_cfg.CYCLE_LENGTH
                                    turn_in_cycle = game_state.real_env_steps % self.env_cfg.CYCLE_LENGTH
                                    is_day = turn_in_cycle < self.env_cfg.DAY_LENGTH
                                    is_night = not is_day
                                    remaining_days = 0 if is_night else (self.env_cfg.DAY_LENGTH - turn_in_cycle)
                                    if remaining_days > (len(route) - 1):
                                        pickup_amount = 0
                                    else:
                                        pickup_amount = (len(route) - 1) * unit.action_queue_cost(game_state) * 2 - 1
                                else:
                                    pickup_amount = max(
                                        0, factory.power - 300
                                    )
                            elif (
                                factory.state.role == FactoryRole.MAIN
                                and len(
                                    factory.state.robot_missions[
                                        UnitMission.PIPE_FACTORY_TO_ICE
                                    ]
                                )
                                == len(factory.state.plans["ice"].route)
                                and len(
                                    factory.state.robot_missions[
                                        UnitMission.PIPE_FACTORY_TO_ORE
                                    ]
                                )
                                == len(factory.state.plans["ore"].route)
                            ):
                                if factory.state.ore_disabled:
                                    pickup_amount = factory.power
                                else:
                                    pickup_amount = max(
                                        0, factory.power // 2
                                    )
                            pickup_amount = min(pickup_amount, unit.unit_cfg.BATTERY_CAPACITY - unit.power)
                            if pickup_amount > 0:
                                if unit.power >= unit.action_queue_cost(game_state) * 2:
                                    actions[unit_id] = [unit.pickup(4, min(unit.unit_cfg.BATTERY_CAPACITY, pickup_amount))]
                        else:
                            direction = direction_to(
                                unit.pos, route.path[route_step - 1]
                            )
                            if resource_id == resource_ids["power"]:
                                min_power = unit.action_queue_cost(game_state) * 4
                                pipe_full = game_state.units[self.player][factory.state.robot_missions[unit.state.mission][-1]].state.role.is_stationary
                                if game_state.board.rubble[unit.pos[0], unit.pos[1]] > 0 and pipe_full:
                                    min_power += unit.unit_cfg.DIG_COST  # for rubble mining
                                if unit.power > min_power:
                                    transfer_power = unit.power - min_power
                                    actions[unit_id] = [
                                        unit.transfer(direction, 4, transfer_power)
                                    ]
                            elif unit.cargo.from_id(resource_id) > 0:
                                if unit.power >= unit.action_queue_cost(game_state):
                                    actions[unit_id] = [
                                        unit.transfer(
                                            direction,
                                            resource_id,
                                            unit.cargo.from_id(resource_id),
                                        )
                                    ]
                            if game_state.board.rubble[unit.pos[0], unit.pos[1]] > 0:  # Rubble Mining!!
                                pipe_full = game_state.units[self.player][factory.state.robot_missions[unit.state.mission][-1]].state.role.is_stationary
                                max_digger_achieved = False
                                if factory.state.role == FactoryRole.MAIN:
                                    if factory.state.sub_factory is not None:
                                        max_digger_achieved = len(game_state.factories[self.player][factory.state.sub_factory].state.robot_missions[UnitMission.DIG_RUBBLE]) >= game_state.factories[self.player][factory.state.sub_factory].state.MAX_DIGGER
                                elif factory.state.role == FactoryRole.SUB:
                                    max_digger_achieved = len(factory.state.robot_missions[UnitMission.DIG_RUBBLE]) >= factory.state.MAX_DIGGER
                                if pipe_full and max_digger_achieved:
                                    if unit.power >= unit.unit_cfg.DIG_COST + unit.action_queue_cost(game_state):
                                        actions[unit_id] = [unit.dig(repeat=0, n=1)]

                else:
                    if resource_type in ["ice", "ore"]:
                        resource_threshold = 6  # TODO calculate resource threshold
                        if unit.unit_type == "HEAVY":
                            resource_threshold *= 10
                    else:
                        resource_threshold = 2
                    min_power = (
                                unit.action_queue_cost(game_state) * 2
                                + unit.unit_cfg.INIT_POWER
                    )  # TODO Consider leading units' powers (especially the miner!)
                    if unit.power <= min_power + unit.action_queue_cost(game_state) * 3:
                        unit.state.waiting_for += 1
                        if unit.state.waiting_for > 4:
                            unit.state.state = UnitStateEnum.MOVING_TO_FACTORY
                            continue 
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
                            if unit.power > min_power + unit.action_queue_cost(game_state):
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
            if unit.state.state == UnitStateEnum.MOVING_TO_RUBBLE:
                if unit.state.target_pos is None or game_state.board.rubble[unit.state.target_pos[0], unit.state.target_pos[1]] == 0:
                    unit.state.target_pos = find_next_rubble_to_mine()
                arrived = move_to(unit.state.target_pos)
                if arrived:
                    unit.state.state = UnitStateEnum.DIGGING_RUBBLE
                    continue
                break
            if unit.state.state == UnitStateEnum.DIGGING_RUBBLE:
                if game_state.board.rubble[unit.pos[0], unit.pos[1]] > 0:
                    if unit.power >= unit.dig_cost(game_state) + unit.action_queue_cost(game_state) * 10:
                        actions[unit_id] = [unit.dig(repeat=0, n=1)]
                    else:
                        unit.state.state = UnitStateEnum.RUBBLE_MOVING_TO_FACTORY
                        continue
                else:
                    unit.state.target_pos = None
                    unit.state.state = UnitStateEnum.MOVING_TO_RUBBLE
                    continue
                break
            if unit.state.state == UnitStateEnum.RUBBLE_MOVING_TO_FACTORY:
                target = factory.pos
                if factory.state.empty_factory_locs is not None:
                    dists = taxi_dist(unit.pos, factory.state.empty_factory_locs)
                    target = factory.state.empty_factory_locs[np.argmin(dists)]
                arrived = move_to(target)
                if arrived:
                    unit.state.state = UnitStateEnum.RUBBLE_RECHARGING
                    continue
                break
            if unit.state.state == UnitStateEnum.RUBBLE_RECHARGING:
                if can_pickup_from_factory:
                    target_power = unit.unit_cfg.BATTERY_CAPACITY
                    if unit.power < target_power:
                        if target_power - unit.power <= factory.power:
                            pickup_power = target_power - unit.power + unit.action_queue_cost(game_state)
                            actions[unit_id] = [unit.pickup(4, min(unit.unit_cfg.BATTERY_CAPACITY, pickup_power))]
                    else:
                        unit.state.state = UnitStateEnum.MOVING_TO_RUBBLE
                        continue
                    break
                else:
                    unit.state.state = UnitStateEnum.RUBBLE_MOVING_TO_FACTORY
                    continue
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
            if len(miner_robots) == 0:
                # miner robot died :(
                miner_robot_id = mission_robots.pop(0)
            else:
                miner_robot_id = miner_robots[0]  # There is only one miner robot
            miner_robot_state = self.unit_states[miner_robot_id]
            if len(mission_robots) > 0:
                miner_robot_state.set_role(UnitRole.STATIONARY_MINER)
            else:
                miner_robot_state.set_role(UnitRole.MINER_TRANSPORTER)
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
        elif mission == UnitMission.NONE:
            unit.state.set_role(UnitRole.NONE)
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
            elif not factory.state.ore_disabled and len(factory.state.robot_missions[UnitMission.PIPE_MINE_ORE]) == 0:
                mission = UnitMission.PIPE_MINE_ORE
            elif (
                len(factory.state.robot_missions[UnitMission.PIPE_FACTORY_TO_ICE]) + 1
                < factory.state.plans["ice"].max_route_robots
            ):
                mission = UnitMission.PIPE_FACTORY_TO_ICE
            elif not factory.state.ore_disabled and (
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
            # unit_poses = [unit.pos for unit in game_state.units[self.opp_player].values()]
            # factory_poses = sum([list(get_factory_tiles(factory.pos)) for factory in game_state.factories[self.opp_player].values()], start=[])
            # for x, y in unit_poses + factory_poses:
            #     cost_map[x][y] = -1

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
        
        for factory_id, factory in game_state.factories[self.opp_player].items():
            self.enemy_factory_tiles[factory_id] = list(map(tuple, get_factory_tiles(factory.pos)))
            self.enemy_factory_tile_ban_list += self.enemy_factory_tiles[factory_id]

        for factory_id, factory in factories.items():
            factory_state = self.factory_states[factory_id]
            for plan in factory_state.plans.values():
                self.global_ban_list += plan.route.path

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
        enemy_factory_ids_to_destroy = []
        for factory_id in self.enemy_factory_tiles:
            if factory_id not in game_state.factories[self.opp_player]:
                enemy_factory_ids_to_destroy.append(factory_id)
                for tile in self.enemy_factory_tiles[factory_id]:
                    self.enemy_factory_tile_ban_list.remove(tile)
        for factory_id in enemy_factory_ids_to_destroy:
            del self.enemy_factory_tiles[factory_id]
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
        missions_to_reassign = []
        for unit_id, unit_state in self.unit_states.items():
            if unit_id not in units:
                unit_ids_to_destroy.append(unit_id)
                factory_id = unit_state.owner
                mission = unit_state.mission
                missions_to_reassign.append((factory_id, mission))
                factory_state = self.factory_states[factory_id]
                factory_state.robot_missions[mission].remove(unit_id)

        for unit_id in unit_ids_to_destroy:
            if self.unit_states[unit_id].role == UnitRole.RUBBLE_DIGGER:
                factory_state.MAX_DIGGER = max(1, factory_state.MAX_DIGGER - 1)
            del self.unit_states[unit_id]

        for factory_id, mission in missions_to_reassign:
            factory_state = self.factory_states[factory_id]
            if len(factory_state.robot_missions[mission]) > 0:
                unit_id = factory_state.robot_missions[mission][-1]
                unit = units[unit_id]
                self._assign_role_to_unit(unit, factories[factory_id])
            else:
                ...  # TODO more sophisticated role reassignment

    def act(self, step: int, obs, remainingOverageTime: int = 60):
        actions = dict()
        game_state = obs_to_game_state(step, self.env_cfg, obs)
        # cycle = game_state.real_env_steps // self.env_cfg.CYCLE_LENGTH
        # turn_in_cycle = game_state.real_env_steps % self.env_cfg.CYCLE_LENGTH
        # is_day = turn_in_cycle < self.env_cfg.DAY_LENGTH
        # is_night = not is_day
        # remaining_days = 0 if is_night else (self.env_cfg.DAY_LENGTH - turn_in_cycle)
        # remaining_nights = 0 if is_day else (self.env_cfg.CYCLE_LENGTH - turn_in_cycle)

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
            if factory.state.sub_factory is not None and len(factories[factory.state.sub_factory].state.robot_missions[UnitMission.DIG_RUBBLE]) == factory.state.MAX_DIGGER:
                if not factory.state.ore_disabled:
                    factory.state.ore_disabled = True
                    robots_to_reassign = factory.state.robot_missions[UnitMission.PIPE_MINE_ORE] + factory.state.robot_missions[UnitMission.PIPE_FACTORY_TO_ORE]
                    for pos in factory.state.plans["ore"].route.path:
                        factory.state.ban_list.remove(pos)
                        self.global_ban_list.remove(pos)
                    for robot_id in robots_to_reassign:
                        unit = units[robot_id]
                        factory.state.robot_missions[unit.state.mission].remove(unit.unit_id)
                        unit.state.mission = UnitMission.NONE
                        factory.state.robot_missions[unit.state.mission].append(unit.unit_id)
                        # if unit.unit_type == "HEAVY":
                        #     factory.state.ban_list.append(unit.pos)  # NOTE unit may move

            if factory.state.main_factory is not None and len(factory.state.robot_missions[UnitMission.DIG_RUBBLE]) < factory.state.MAX_DIGGER:
                if remaining_steps < 100 and len(factory.state.robot_missions[UnitMission.DIG_RUBBLE]) > 0:
                    factory.state.MAX_DIGGER = len(factory.state.robot_missions[UnitMission.DIG_RUBBLE])
            factory_pickup_robots = 0
            for mission, mission_robot_ids in factory.state.robot_missions.items():
                for unit_id in mission_robot_ids:
                    unit = units[unit_id]
                    actions = self.handle_robot_actions(
                        game_state, factory, unit, actions, factory_pickup_robots
                    )

            remaining_steps = game_state.remaining_steps

            # handle factory actions
            if factory.state.role == FactoryRole.MAIN:
                water_cost = factory.water_cost(game_state)
                # if remaining_steps < self.env_cfg.MIN_LICHEN_TO_SPREAD:
                #     if (water_cost + 1) * remaining_steps < factory.cargo.water:
                #         actions[factory_id] = factory.water()
                # elif remaining_steps == 1 and water_cost + 1 < factory.cargo.water:
                #     actions[factory_id] = factory.water()
                # else:  # or build robots
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
                if factory.state.ore_disabled:
                    required_transmitters -= factory.state.plans['ore'].max_route_robots
                if len(robot_ids) < (2 - factory.state.ore_disabled):
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
                water_income = 0
                unit_waters = 0
                unit_count = 0
                for unit_id in factory.state.robot_missions[UnitMission.PIPE_FACTORY_TO_FACTORY]:
                    unit = units[unit_id]
                    unit_waters += unit.cargo.water
                    unit_count += 1
                if unit_count > 0:
                    water_income = unit_waters / unit_count
                factory.state.update_average_water_income(water_income)
            
                required_transmitters = factory.state.plans[
                    "factory_to_factory"
                ].max_route_robots
                if len(factory.state.robot_missions[UnitMission.PIPE_FACTORY_TO_FACTORY]) < required_transmitters:
                    if factory.can_build_light(game_state):
                        actions[factory_id] = factory.build_light()
                elif len(
                    factory.state.robot_missions[UnitMission.DIG_RUBBLE]
                ) < factory.state.MAX_DIGGER and factory.can_build_heavy(game_state):  # TODO build more heavy units
                    actions[factory_id] = factory.build_heavy()
                elif factory.water_cost(game_state) > 0:
                    if factory.water_cost(game_state) + min(25, remaining_steps) <= factory.cargo.water:
                        actions[factory_id] = factory.water()
                    elif factory.cargo.water == 1:
                        print(f"Factory {factory_id} is going to die!!", file=sys.stderr)
                    else:
                        print(f"Factory {factory_id} does not have enough water", file=sys.stderr)
                elif remaining_steps <= len(water_costs):
                    water_loss = 2  # include main factory
                    if water_costs[remaining_steps - 1] + (water_loss - factory.state.average_water_income) * remaining_steps <= factory.cargo.water:
                        actions[factory_id] = factory.water()
            else:
                raise ValueError(f"Invalid factory role {factory.state.role}")

        action_was_updated = True
        for _ in range(10):
            if not action_was_updated:
                break
            action_was_updated, actions = stop_movement_collisions(
                obs, game_state, self.env_cfg, self.player, actions, self.unit_states, ban_list=self.global_ban_list
            )
        if actions is None:
            raise ValueError("Invalid actions")
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
    main(env, agents, 1000, 832321049)#167985129)
