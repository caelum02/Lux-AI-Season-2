import sys
from lux.kit import obs_to_game_state, EnvConfig, GameState
from lux.utils import *
from lux.pathfinding import get_shortest_loop
from lux.forward_sim import stop_movement_collisions
import numpy as np
from numpy.linalg import norm

from lux.states import (
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

    def _num_factories(self, game_state: GameState) -> int:
        factories = game_state.factories
        n_0 = len(factories["player_0"]) if "player_0" in factories else 0
        n_1 = len(factories["player_1"]) if "player_1" in factories else 0

        return n_0 + n_1

    def early_setup(self, step: int, obs, remainingOverageTime: int = 60):
        if step == 0:
            # bid 0 to not waste resources bidding and declare as the default faction
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

                # TODO check for ice, ore routes
                # distances[x][y] is the distance to the nearest ice tile
                es_state.factory_score = resource_score + normalized_rubble_score
                es_state.rubble_score = rubble_score
                es_state.resource_score = resource_score

            # how many factories you have left to place
            factories_to_place = game_state.teams[self.player].factories_to_place
            my_turn_to_place = my_turn_to_place_factory(
                game_state.teams[self.player].place_first, step
            )

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
                    factory_score_spawn = factory_score[spawn_loc[0], spawn_loc[1]]
                    rubble_score_spawn = es_state.rubble_score[
                        spawn_loc[0], spawn_loc[1]
                    ]

                    print(
                        f"{self.player} placed factory_{new_factory_id} at {spawn_loc}, factory score: {factory_score_spawn}, rubble score: {rubble_score_spawn}",
                        file=sys.stderr,
                    )

                    resources = dict(water=150, metal=240)
                    if factories_to_place == 1:
                        resources = dict(water=150, metal=150)

                    es_state.latest_main_factory = new_factory_id
                    self.factory_states[new_factory_id] = FactoryState(
                        role=FactoryRole.MAIN
                    )
                    spawn_action = dict(spawn=spawn_loc, **resources)
                # place sub factory
                else:
                    all_locations = np.stack(
                        np.indices((map_size, map_size)), axis=-1
                    ).reshape(-1, 2)
                    _id = es_state.latest_main_factory
                    factory_pos = game_state.factories[self.player][_id].pos
                    factory_distance = taxi_distances(
                        all_locations, factory_pos
                    ).reshape(map_size, map_size)
                    score = factory_distance + es_state.rubble_score * 0.1
                    score += (obs["board"]["valid_spawns_mask"] == 0) * 1e9
                    spawn_loc = np.unravel_index(
                        np.argmin(score, axis=None), score.shape
                    )

                    main_factory_id = es_state.latest_main_factory
                    self.factory_states[
                        es_state.latest_main_factory
                    ].sub_factory = new_factory_id
                    self.factory_states[new_factory_id] = FactoryState(
                        role=FactoryRole.SUB, main_factory=main_factory_id
                    )

                    es_state.latest_main_factory = None
                    es_state.sub_factory_map[main_factory_id] = new_factory_id

                    spawn_action = {"spawn": spawn_loc, "water": 150, "metal": 60}

                    # Debug message
                    print(
                        f"{self.player} placed factory_{new_factory_id} at {spawn_loc}, main: {main_factory_id}",
                        file=sys.stderr,
                    )

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
        self, game_state, factory, unit, actions, factory_pickup_robots
    ):
        unit_id = unit.unit_id
        factory_inf_distance = norm(factory.pos - unit.pos, ord=np.inf)
        can_pickup_from_factory = factory_inf_distance <= 1
        can_transfer_to_factory = factory_inf_distance <= 2
        resource_type = unit.state.resource_type
        resource_plan = None
        if resource_type is not None:
            resource_plan = factory.state.resources[resource_type]
            res_threshold = resource_plan.resource_threshold_light
            if unit.unit_type == "HEAVY":
                res_threshold *= 10

        if unit.state.role is None:
            raise ValueError("Unit role is None")
        if unit.state.role.is_stationary:
            start_pos = unit.state.idle_pos
        elif unit.state.role.is_transporter:
            start_pos = resource_plan.resource_factory_pos
        elif unit.state.role == UnitRole.RUBBLE_DIGGER:
            ...  # TODO
        else:
            raise ValueError("Invalid unit role")

        is_resource_tick = game_state.real_env_step % 2 == 0
        is_power_transfer_tick = not is_resource_tick

        def next_state(current, mission, role):
            if role.is_stationary:
                if current == UnitStateEnum.INITIAL:
                    if role.is_miner:
                        return UnitStateEnum.DIGGING
                    else:
                        return UnitStateEnum.TRANSFERING_RESOURCE
                elif current == UnitStateEnum.DIGGING:
                    return UnitStateEnum.TRANSFERING_RESOURCE
                elif current == UnitStateEnum.TRANSFERING_RESOURCE:
                    return UnitStateEnum.DIGGING

        for _ in range(len(UnitStateEnum)):
            if unit.state.role.is_stationary:
                if np.any(start_pos != unit.pos):
                    unit.state.state = UnitStateEnum.INITIAL
                    continue
                if unit.state.state == UnitStateEnum.INITIAL:
                    target = start_pos
                    if np.all(target == unit.pos):
                        unit.state.state = next_state(
                            unit.state.state, unit.state.mission, unit.state.role
                        )
                    else:
                        direction = direction_to(
                            unit.pos, target
                        )  # TODO better pathfinding
                        move_cost = unit.move_cost(game_state, direction)
                        if (
                            move_cost is not None
                            and unit.power
                            >= move_cost + unit.action_queue_cost(game_state)
                        ):
                            actions[unit_id] = [unit.move(direction, repeat=0, n=1)]
                        break
                elif unit.state.state == UnitStateEnum.DIGGING:
                    if getattr(unit.cargo, resource_type) < res_threshold:
                        if unit.power >= unit.dig_cost(
                            game_state
                        ) + unit.action_queue_cost(game_state):
                            actions[unit_id] = [unit.dig(repeat=0, n=1)]
                        break
                    else:
                        unit.state.state = next_state(
                            unit.state.state, unit.state.mission, unit.state.role
                        )
                elif unit.state.state == UnitStateEnum.TRANSFERING_RESOURCE:
                    if is_resource_tick:
                        route = resource_plan.resource_route
                        try:
                            route_step = route.path.index(tuple(unit.pos))
                        except ValueError:
                            # not in route
                            unit.state.state = UnitStateEnum.INITIAL
                            continue
                        if route_step == 0:
                            pickup_power = 100
                            if unit.power < pickup_power:
                                power = unit.power - pickup_power
                                if factory.power >= power:
                                    if unit.power >= unit.action_queue_cost(game_state):
                                        actions[unit_id] = [
                                            unit.pickup(4, power, repeat=0, n=1)
                                        ]
                            break
                        else:
                            before_in_route = route.path[route_step - 1]
                        direction = direction_to(unit.pos, before_in_route)
                        if getattr(unit.cargo, resource_type) > 0:
                            if unit.power >= unit.action_queue_cost(game_state):
                                resource_id = 0 if resource_type == "ice" else 1
                                actions[unit_id] = [
                                    unit.transfer(
                                        direction,
                                        resource_type,
                                        getattr(unit.cargo, resource_type),
                                        repeat=0,
                                        n=1,
                                    )
                                ]
                        elif unit.state.role.is_miner:
                            unit.state.state = next_state(
                                unit.state.state, unit.state.mission, unit.state.role
                            )
                        break
                    elif is_power_transfer_tick:
                        route = resource_plan.resource_route
                        try:
                            route_step = route.path.index(tuple(unit.pos))
                        except ValueError:
                            # not in route
                            unit.state.state = UnitStateEnum.INITIAL
                            continue
                        if route_step == len(route.path) - 1:
                            break
                        else:
                            next_in_route = route.path[route_step + 1]
                            direction = direction_to(unit.pos, next_in_route)
                            min_remaining_power = 4 * unit.action_queue_cost(game_state)
                            if unit.power >= min_remaining_power:
                                power = unit.power - min_remaining_power
                                actions[unit_id] = [
                                    unit.transfer(direction, 4, power, repeat=0, n=1)
                                ]
                continue
            if unit.state.state == UnitStateEnum.INITIAL:
                target = start_pos
                if np.all(target == unit.pos):
                    unit.state.state = next_state(
                        unit.state.state, unit.state.mission, unit.state.role
                    )
                else:
                    direction = direction_to(
                        unit.pos, target
                    )  # TODO better pathfinding
                    move_cost = unit.move_cost(game_state, direction)
                    if (
                        move_cost is not None
                        and unit.power >= move_cost + unit.action_queue_cost(game_state)
                    ):
                        actions[unit_id] = [unit.move(direction, repeat=0, n=1)]
                    break

            if unit.state.state == UnitStateEnum.MOVING_TO_RESOURCE:
                target = resource_plan.resource_pos
                if np.all(target == unit.pos):
                    unit.state.state = UnitStateEnum.DIGGING
                else:
                    route = resource_plan.resource_route
                    try:
                        route_step = route.path.index(tuple(unit.pos))
                    except ValueError:
                        # not in route
                        unit.state.state = UnitStateEnum.INITIAL
                        continue
                    next_in_route = route.path[route_step + 1]
                    direction = direction_to(unit.pos, next_in_route)
                    move_cost = unit.move_cost(game_state, direction)
                    if (
                        move_cost is not None
                        and unit.power >= move_cost + unit.action_queue_cost(game_state)
                    ):
                        actions[unit_id] = [unit.move(direction, repeat=0, n=1)]
                    break
            if unit.state.state == UnitStateEnum.DIGGING:
                if getattr(unit.cargo, resource_type) < res_threshold:
                    if unit.power >= unit.dig_cost(game_state) + unit.action_queue_cost(
                        game_state
                    ):
                        actions[unit_id] = [unit.dig(repeat=0, n=1)]
                    break
                else:
                    unit.state.state = UnitStateEnum.MOVING_TO_FACTORY
            if unit.state.state == UnitStateEnum.MOVING_TO_FACTORY:
                target = resource_plan.resource_factory_pos
                if np.all(target == unit.pos):
                    unit.state.state = UnitStateEnum.DROPPING_RESOURCE
                else:
                    route = resource_plan.resource_route
                    try:
                        route_step = route.path.index(tuple(unit.pos))
                    except ValueError:
                        # not in route
                        unit.state.state = UnitStateEnum.INITIAL
                        continue
                    next_in_route = route.path[route_step + 1]
                    direction = direction_to(unit.pos, next_in_route)
                    move_cost = unit.move_cost(game_state, direction)
                    if (
                        move_cost is not None
                        and unit.power >= move_cost + unit.action_queue_cost(game_state)
                    ):
                        actions[unit_id] = [unit.move(direction, repeat=0, n=1)]
                    break
            if unit.state.state == UnitStateEnum.DROPPING_RESOURCE:
                if getattr(unit.cargo, resource_type) > 0:
                    if unit.power >= unit.action_queue_cost(game_state):
                        direction = 0
                        resource_id = 0 if resource_type == "ice" else 1
                        actions[unit_id] = [
                            unit.transfer(
                                direction,
                                resource_id,
                                getattr(unit.cargo, resource_type),
                                repeat=0,
                                n=1,
                            )
                        ]
                    break
                else:
                    unit.state.state = UnitStateEnum.RECHARGING
                    if unit.unit_type == "LIGHT":
                        factory_pickup_robots += 1
                    else:
                        factory_pickup_robots += 10
            if unit.state.state == UnitStateEnum.RECHARGING:
                target_power = unit.unit_cfg.INIT_POWER
                robots_multiple = (
                    factory_pickup_robots
                    if unit.unit_type == "LIGHT"
                    else factory_pickup_robots / 10
                )
                if unit.power < target_power:
                    if (
                        target_power - unit.power
                    ) * robots_multiple <= factory.power:  # NOTE naive setup. need to check all needs
                        power = target_power - unit.power
                        actions[unit_id] = [unit.pickup(4, power, repeat=0, n=1)]
                        break
                    else:
                        if len(unit.action_queue) == 0:
                            actions[
                                unit_id
                            ] = (
                                []
                            )  # may need to move(0) if there is existing action queue
                            del actions[unit_id]  # Save action queue update cost
                        else:
                            actions[unit_id] = unit.move(0, repeat=0, n=1)
                        break
                else:
                    unit.state.state = UnitStateEnum.MOVING_TO_RESOURCE
        else:
            print("Unit state machine failed to break", file=sys.stderr)
            raise ValueError()
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
                unit.state.role = UnitRole.STATIONARY_MINER
            else:
                unit.state.role = UnitRole.MINER_TRANSPORTER
        elif mission in [
            UnitMission.PIPE_FACTORY_TO_ICE,
            UnitMission.PIPE_FACTORY_TO_ORE,
        ]:
            route = None
            miner_robots = None
            if mission == UnitMission.PIPE_FACTORY_TO_ICE:
                route = factory.state.resources["ice"].resource_route
                miner_robots = factory.state.robot_missions[UnitMission.PIPE_MINE_ICE]
            elif mission == UnitMission.PIPE_FACTORY_TO_ORE:
                route = factory.state.resources["ore"].resource_route
                miner_robots = factory.state.robot_missions[UnitMission.PIPE_MINE_ORE]
            miner_robot_id = miner_robots[0]  # There is only one miner robot
            miner_robot_state = self.unit_states[miner_robot_id]
            miner_robot_state.role = UnitRole.STATIONARY_MINER
            miner_robot_state.idle_pos = route.path[-1]
            for i, robot_id in enumerate(mission_robots[:-1]):
                robot_state = self.unit_states[robot_id]
                robot_state.role = UnitRole.STATIONARY_TRANSPORTER
                robot_state.idle_pos = route.path[-2 - i]
            if len(mission_robots) + 1 == len(route):
                # Full pipe
                robot_state = self.unit_states[robot_id]
                robot_state.role = UnitRole.STATIONARY_TRANSPORTER
                robot_state.idle_pos = route.path[0]
            else:
                robot_state.role = UnitRole.TRANSPORTER

        elif mission == UnitMission.PIPE_FACTORY_TO_FACTORY:
            raise NotImplementedError()

        elif mission == UnitMission.DIG_RUBBLE:
            unit.state.role = UnitRole.RUBBLE_DIGGER
        else:
            raise ValueError("Invalid mission")

    def _add_unit_to_factory(self, unit, factory):
        mission = None
        if factory.role == FactoryRole.SUB:
            mission = UnitMission.PIPE_FACTORY_TO_FACTORY
        else:
            if factory.state.robot_missions[UnitMission.PIPE_MINE_ICE] == 0:
                mission = UnitMission.PIPE_MINE_ICE
            elif factory.state.robot_missions[UnitMission.PIPE_MINE_ORE] == 0:
                mission = UnitMission.PIPE_MINE_ORE
            elif (
                len(factory.state.robot_missions[UnitMission.PIPE_FACTORY_TO_ICE]) + 1
                < factory.state.resources["ice"].max_resource_robots
            ):
                mission = UnitMission.PIPE_FACTORY_TO_ICE
                need_more_ice_robot = len(factory.state.robot_missions)
            elif (
                len(factory.state.robot_missions[UnitMission.PIPE_FACTORY_TO_ORE]) + 1
                < factory.state.resources["ore"].max_resource_robots
            ):
                mission = UnitMission.PIPE_FACTORY_TO_ORE
            else:
                mission = UnitMission.DIG_RUBBLE

        unit.state.mission = mission
        factory.state.robot_missions[mission] += [unit.unit_id]
        unit.state.resource_type = mission.resource_type
        if mission.resource_type is not None:
            unit.state.following_route = factory.state.resources[
                mission.resource_type
            ].resource_route
        if mission == UnitMission.PIPE_FACTORY_TO_FACTORY:
            unit.state.following_route = None  # TODO factory-factory route

        self._assign_role_to_unit(unit, factory)

    def _fetch_states(self, game_state: GameState):
        units = game_state.units[self.player]
        factories = game_state.factories[self.player]

        for unit_id, unit in units.items():
            if unit_id in self.unit_states:
                unit.state = unit.state = self.unit_states[unit_id]

        for factory_id, factory in factories.items():
            if factory_id in self.factory_states:
                factory.state = self.factory_states[factory_id]

    def get_move_cost_map(self, game_state):
        return (
            game_state.board.rubble * self.env_cfg.ROBOTS["LIGHT"].RUBBLE_MOVEMENT_COST
            + self.env_cfg.ROBOTS["LIGHT"].MOVE_COST
        )

    def _register_factories(self, game_state):
        """
        initialize self.factory_states
        """

        ice_map = game_state.board.ice
        ice_locs = np.argwhere(ice_map == 1)
        ore_map = game_state.board.ore
        ore_locs = np.argwhere(ore_map == 1)

        factories = game_state.factories[self.player]

        for factory_id, factory in factories.items():
            factory_state = self.factory_states[factory_id]
            cost_map = self.get_move_cost_map(game_state)
            empty_factory_locs = get_factory_tiles(factory.pos)

            # Sub-factory is handled with its main factory
            if factory_state.role == FactoryRole.SUB:
                continue

            factory_to_factory_path = []

            # Handle sub-factory if exists
            if factory_state.sub_factory is not None:
                sub_factory_id = factory_state.sub_factory
                sub_factory = factories[sub_factory_id]
                sub_factory_locs = get_factory_tiles(sub_factory.pos)

                distances = taxi_distances(sub_factory_locs, empty_factory_locs)
                argmin_sub_factory, argmin_factory = np.unravel_index(
                    np.argmin(distances), distances.shape
                )
                closest_sub_factory_loc = sub_factory_locs[argmin_sub_factory]
                closest_factory_loc = empty_factory_locs[argmin_factory]

                empty_indices = np.any(
                    empty_factory_locs != closest_factory_loc, axis=-1
                ).nonzero()
                empty_factory_locs = empty_factory_locs[empty_indices]

                factory_to_factory_route = get_shortest_loop(
                    cost_map,
                    closest_sub_factory_loc,
                    closest_factory_loc,
                    ban_list=[*ice_locs, *ore_locs],
                )
                if factory_to_factory_route is None:
                    raise ValueError("No factory to factory route found")
                factory_to_factory_path = factory_to_factory_route.path
                self.factory_states[sub_factory_id].plans = {
                    "factory_to_factory": TransmitPlan(
                        target_pos=closest_factory_loc,
                        source_pos=closest_sub_factory_loc,
                        transmit_route=factory_to_factory_route,
                        max_transmit_robots=len(factory_to_factory_route),
                    )
                }

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
                ban_list=[closest_ore_tile, closest_ore_factory_tile]
                + factory_to_factory_path,
            )
            if ice_route is None:
                raise ValueError("No ice route found")

            ore_route = get_shortest_loop(
                cost_map,
                closest_ore_factory_tile,
                closest_ore_tile,
                ban_list=ice_route.path + factory_to_factory_path,
            )
            if ore_route is None:
                raise ValueError("No ore route found")

            self.factory_states[factory_id].plans = dict(
                ice=ResourcePlan(
                    destination=closest_ice_tile,
                    source=closest_ice_factory_tile,
                    route=ice_route,
                    max_route_robots=len(ice_route),
                    resource_threshold_light=8,
                ),
                ore=ResourcePlan(
                    destination=closest_ore_tile,
                    source=closest_ore_factory_tile,
                    route=ore_route,
                    max_route_robots=len(ore_route),
                    resource_threshold_light=8,
                ),
            )

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
        for factory_id in factory_ids_to_destroy:
            del self.factory_states[factory_id]
        return game_state

    def _register_units(self, units, factories, factory_centers, factory_ids):
        # Register robots to factories if not registered
        for unit_id, unit in units.items():
            if unit_id not in self.unit_states:
                self.unit_states[unit_id] = unit.state

                factory_distances = taxi_dist(factory_centers, unit.pos)
                factory_id = factory_ids[np.argmin(factory_distances)]
                factory_state = self.factory_states[factory_id]
                unit.state.owner = factory_id

                self._add_unit_to_factory(unit, factories[factory_id])

    def _unregister_units(self, units, factories):
        unit_ids_to_destroy = []
        for unit_id, unit_state in self.unit_states:
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

        cycle = step // self.env_cfg.CYCLE_LENGTH
        turn_in_cycle = step % self.env_cfg.CYCLE_LENGTH
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
                        and self.unit_states[unit_id].state == UnitStateEnum.RECHARGING
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
                    robots = sum(factory.state.robot_missions.values(), start=[])
                    light_robots = [
                        robot for robot in robots if robot.unit_type == "LIGHT"
                    ]
                    heavy_robots = [
                        robot for robot in robots if robot.unit_type == "HEAVY"
                    ]
                    required_transmitters = sum(map(lambda plan: plan.max_route_robots - 1, factory.state.plans.values()))
                    if len(heavy_robots) < 2:
                        if factory.can_build_heavy(game_state):
                            actions[factory_id] = factory.build_heavy()
                    elif len(light_robots) < required_transmitters:
                        if factory.can_build_light(game_state):
                            actions[factory_id] = factory.build_light()
                    else:
                        if factory.can_build_heavy(game_state):
                            actions[factory_id] = factory.build_heavy()

            elif factory.state.role == FactoryRole.SUB:
                required_transmitters = factory.state.plans[
                    "factory_to_factory"
                ].max_transmit_robots
                robots = sum(factory.state.robot_missions.values(), start=[])
                if len(robots) < required_transmitters:
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
