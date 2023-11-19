import sys
from typing import Literal
from lux.kit import obs_to_game_state, GameState, EnvConfig
from lux.utils import direction_to, get_factory_tiles, my_turn_to_place_factory, taxi_dist
from lux.pathfinding import get_shortest_loop
from lux.forward_sim import stop_movement_collisions
import numpy as np
from numpy.linalg import norm

from lux.states import ResourcePlan, UnitState, UnitStateEnum, FactoryState
from action_enum import RESOURCE_T

class Agent():
    def __init__(self, player: str, env_cfg: EnvConfig) -> None:
        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        np.random.seed(0)
        self.env_cfg: EnvConfig = env_cfg
        self.factory_score = None
        self.rubble_score = None

        self.units_master_factory = dict() # unit_id -> factory_id

        self.robots_ice_factory = dict() # factory_id -> unit_id
        self.robots_ore_factory = dict() # factory_id -> unit_id
        self.robots_rubble_factory = dict() # factory_id -> unit_id

        self.action_start_power = dict()

        self.unit_states = dict() # unit_id -> UnitState
        self.factory_states = dict() # factory_id -> FactoryState

    def early_setup(self, step: int, obs, remainingOverageTime: int = 60):
        if step == 0:
            # bid 0 to not waste resources bidding and declare as the default faction
            # you can bid -n to prefer going second or n to prefer going first in placement
            return dict(faction="AlphaStrike", bid=0)
        else:
            game_state = obs_to_game_state(step, self.env_cfg, obs)
            # factory placement period
            if self.factory_score is None:
                map_size = self.env_cfg.map_size
                self.factory_score = np.zeros((map_size, map_size))
                self.rubble_score = conv2d(game_state.board.rubble, average_kernel(5), n=3)
                self.factory_score += (self.rubble_score / np.max(self.rubble_score)) * 0.5
                ice_tile_locations = np.argwhere(game_state.board.ice == 1)
                ore_tile_locations = np.argwhere(game_state.board.ore == 1)
                all_locations = np.array(np.meshgrid(np.arange(0, map_size), np.arange(0, map_size), indexing='xy')).swapaxes(0, 2).reshape(-1, 2)
                ice_distances = taxi_dist(np.expand_dims(all_locations, 1), np.expand_dims(ice_tile_locations, 0))
                ice_distances = np.min(ice_distances, axis=-1)
                ice_distances = ice_distances.reshape(map_size, map_size)

                ore_distances = taxi_dist(np.expand_dims(all_locations, 1), np.expand_dims(ore_tile_locations, 0))
                ore_distances = np.min(ore_distances, axis=-1)
                ore_distances = ore_distances.reshape(map_size, map_size)
                # TODO check for ice, ore routes
                # distances[x][y] is the distance to the nearest ice tile 
                self.factory_score += np.clip(ice_distances-2, a_min=0, a_max=None) + np.clip(ore_distances-2, a_min=0, a_max=None) * 0.3
            
            # how much water and metal you have in your starting pool to give to new factories
            water_left = game_state.teams[self.player].water
            metal_left = game_state.teams[self.player].metal
            
            # how many factories you have left to place
            factories_to_place = game_state.teams[self.player].factories_to_place
            if water_left == 0:
                factories_to_place = 0
            # whether it is your turn to place a factory
            my_turn_to_place = my_turn_to_place_factory(game_state.teams[self.player].place_first, step)
            if factories_to_place > 0 and my_turn_to_place:
                # Build factory at the position with the lowest factory_score
                factory_score = self.factory_score + (obs["board"]["valid_spawns_mask"] == 0) * 1e9
                
                spawn_loc = np.unravel_index(np.argmin(factory_score), factory_score.shape)
                map_size = self.env_cfg.map_size
                factory_score_spawn = factory_score[spawn_loc[0], spawn_loc[1]]
                rubble_score_spawn = self.rubble_score[spawn_loc[0], spawn_loc[1]]
                print(f"{self.player} placed factory {spawn_loc}, factory score: {factory_score_spawn}, rubble score: {rubble_score_spawn}", file=sys.stderr)
                resource_for_each_factory = 200
                resource = dict(water=resource_for_each_factory, metal=resource_for_each_factory)
                if water_left >= 2 * resource_for_each_factory and metal_left >= 2 * resource_for_each_factory:
                    water_left -= resource_for_each_factory
                    metal_left -= resource_for_each_factory
                    factories_to_place -= 1
                else:
                    resource = dict(water=water_left, metal=metal_left)
                return dict(spawn=spawn_loc, **resource)
            return dict()

    def _get_factory_misc(self, factories):

        factory_centers, factory_units = [], []
        factory_ids = []
        for factory_id, factory in factories.items():
            factory_centers += [factory.pos]
            factory_units += [factory]
            factory_ids += [factory_id]           

        return factory_centers, factory_units, factory_ids

    def _initialize_robot_bindings(self, factories):
        for factory_id in factories.keys():
            self.robots_ice_factory[factory_id] = list()
            self.robots_ore_factory[factory_id] = list()
            self.robots_rubble_factory[factory_id] = list()

    @staticmethod
    def _num_robot_type_from_list(robot_list, robot_type, units):
        return len([1 for unit_id in robot_list if units[unit_id].unit_type == robot_type])
    
    
    def num_lights_ore_factory(self, factory_id, units):
        return self._num_robot_type_from_list(self.robots_ore_factory[factory_id], "LIGHT", units)

    
    def num_heavies_ore_factory(self, factory_id, units):
        return self._num_robot_type_from_list(self.robots_ore_factory[factory_id], "HEAVY", units)
    
    
    def num_lights_ice_factory(self, factory_id, units):
        return self._num_robot_type_from_list(self.robots_ice_factory[factory_id], "LIGHT", units)
    
    
    def num_heavies_ice_factory(self, factory_id, units):
        return self._num_robot_type_from_list(self.robots_ice_factory[factory_id], "HEAVY", units)

    def handle_robot_resource_gathering(self, game_state, factory, factory_state, unit, resource: Literal["ice"] | Literal["ore"], actions, factory_pickup_robots):
        unit_id = unit.unit_id
        unit_state = self.unit_states[unit_id]
        factory_inf_distance = norm(factory.pos - unit.pos, ord=np.inf)
        can_pickup_from_factory = factory_inf_distance <= 1
        can_transfer_to_factory = factory_inf_distance <= 2
        resource_plan = factory_state.resources[resource]
        
        res_threshold = resource_plan.resource_threshold_light
        if unit.unit_type == "HEAVY":
            res_threshold *= 10
        for _ in range(len(UnitStateEnum)):
            if unit_state.state == UnitStateEnum.MOVING_TO_START:
                target = resource_plan.resource_factory_pos
                if np.all(target == unit.pos):
                    unit_state.state = UnitStateEnum.MOVING_TO_RESOURCE
                else:
                    direction = direction_to(unit.pos, target)
                    move_cost = unit.move_cost(game_state, direction)
                    if move_cost is not None and unit.power >= move_cost + unit.action_queue_cost(game_state):
                        actions[unit_id] = [unit.move(direction, repeat=0, n=1)]
                    break
            if unit_state.state == UnitStateEnum.MOVING_TO_RESOURCE:
                target = resource_plan.resource_pos
                if np.all(target == unit.pos):
                    unit_state.state = UnitStateEnum.DIGGING
                else:
                    route = resource_plan.resource_route
                    try:
                        route_step = route.path.index(tuple(unit.pos))
                    except ValueError:
                        # not in route
                        unit_state.state = UnitStateEnum.MOVING_TO_START
                        continue
                    next_in_route = route.path[route_step + 1]
                    direction = direction_to(unit.pos, next_in_route)
                    move_cost = unit.move_cost(game_state, direction)
                    if move_cost is not None and unit.power >= move_cost + unit.action_queue_cost(game_state):
                        actions[unit_id] = [unit.move(direction, repeat=0, n=1)]
                    break
            if unit_state.state == UnitStateEnum.DIGGING:
                if getattr(unit.cargo, resource) < res_threshold:
                    if unit.power >= unit.dig_cost(game_state) + unit.action_queue_cost(game_state):
                        actions[unit_id] = [unit.dig(repeat=0, n=1)]
                    break
                else:
                    unit_state.state = UnitStateEnum.MOVING_TO_FACTORY
            if unit_state.state == UnitStateEnum.MOVING_TO_FACTORY:
                target = resource_plan.resource_factory_pos
                if np.all(target == unit.pos):
                    unit_state.state = UnitStateEnum.DROPPING_RESOURCE
                else:
                    route = resource_plan.resource_route
                    try:
                        route_step = route.path.index(tuple(unit.pos))
                    except ValueError:
                        # not in route
                        unit_state.state = UnitStateEnum.MOVING_TO_START
                        continue
                    next_in_route = route.path[route_step + 1]
                    direction = direction_to(unit.pos, next_in_route)
                    move_cost = unit.move_cost(game_state, direction)
                    if move_cost is not None and unit.power >= move_cost + unit.action_queue_cost(game_state):
                        actions[unit_id] = [unit.move(direction, repeat=0, n=1)]
                    break
            if unit_state.state == UnitStateEnum.DROPPING_RESOURCE:
                if getattr(unit.cargo, resource) > 0:
                    if unit.power >= unit.action_queue_cost(game_state):
                        direction = 0
                        resource_type = 0 if resource == "ice" else 1
                        actions[unit_id] = [unit.transfer(direction, resource_type, getattr(unit.cargo, resource), repeat=0, n=1)]
                    break
                else:
                    unit_state.state = UnitStateEnum.RECHARGING
                    if unit.unit_type == "LIGHT":
                        factory_pickup_robots += 1
                    else:
                        factory_pickup_robots += 10
            if unit_state.state == UnitStateEnum.RECHARGING:
                target_power = unit.unit_cfg.INIT_POWER
                robots_multiple = factory_pickup_robots if unit.unit_type == "LIGHT" else factory_pickup_robots / 10
                if unit.power < target_power:
                    if (target_power - unit.power) * robots_multiple <= factory.power: # NOTE naive setup. need to check all needs
                        power = target_power - unit.power
                        actions[unit_id] = [unit.pickup(4, power, repeat=0, n=1)]
                        break
                    else:
                        if len(unit.action_queue) == 0:
                            actions[unit_id] = []  # may need to move(0) if there is existing action queue
                            del actions[unit_id]  # Save action queue update cost
                        else:
                            actions[unit_id] = unit.move(0, repeat=0, n=1)
                        break
                else:
                    unit_state.state = UnitStateEnum.MOVING_TO_RESOURCE
        else:
            print("Unit state machine failed to break", file=sys.stderr)
            raise ValueError()
        return actions

    def add_unit_to_factory(self, factory_id, unit_id, factory_state, resource):
            if resource == "ice":
                self.robots_ice_factory[factory_id].append(unit_id)
            elif resource == "ore":
                self.robots_ore_factory[factory_id].append(unit_id)
            else:
                raise ValueError("Unknown resource")
            self.unit_states[unit_id] = UnitState(following_route=factory_state.resources[resource].resource_route)

    def act(self, step: int, obs, remainingOverageTime: int = 60):
        """
            1. If robot is idle, then allocate it to a factory and give occupation (ice/ore)
            
            TODO
            2. collision
            3. factory needs to reallocate robots
                If unit is enough, then water to gain more power
                power_requirement
                    unit action_queue_cost

        """
        actions = dict()
        game_state = obs_to_game_state(step, self.env_cfg, obs)
        
        cycle = step // self.env_cfg.CYCLE_LENGTH
        turn_in_cycle = step % self.env_cfg.CYCLE_LENGTH
        is_day = turn_in_cycle < self.env_cfg.DAY_LENGTH
        is_night = not is_day
        remaining_days = 0 if is_night else (self.env_cfg.DAY_LENGTH - turn_in_cycle)
        remaining_nights = 0 if is_day else (self.env_cfg.CYCLE_LENGTH - turn_in_cycle)

        factories = game_state.factories[self.player]
        factory_centers, factory_units, factory_ids = self._get_factory_misc(factories)
        units = game_state.units[self.player]
        
        ice_map = game_state.board.ice
        ice_tile_locations = np.argwhere(ice_map == 1)
        ore_map = game_state.board.ore
        ore_tile_locations = np.argwhere(ore_map == 1)

        if obs['real_env_steps'] == 0:
            self._initialize_robot_bindings(factories)
            for factory_id, factory in factories.items():
                rubble_map = game_state.board.rubble.copy() * self.env_cfg.ROBOTS["LIGHT"].RUBBLE_MOVEMENT_COST + self.env_cfg.ROBOTS["LIGHT"].MOVE_COST
                factory_tiles = get_factory_tiles(factory.pos)
                ice_tile_distances = taxi_dist(np.expand_dims(ice_tile_locations, 1), np.expand_dims(factory_tiles, 0))
                argmin_ice_tile, argmin_factory_tile = np.unravel_index(np.argmin(ice_tile_distances), ice_tile_distances.shape)
                closest_ice_tile = ice_tile_locations[argmin_ice_tile]
                closest_ice_factory_tile = factory_tiles[argmin_factory_tile]
                factory_tiles = set(list(map(tuple, factory_tiles)))
                factory_tiles.remove(tuple(closest_ice_factory_tile))
                factory_tiles = np.array(list(factory_tiles))
                ore_tile_distances = taxi_dist(np.expand_dims(ore_tile_locations, 1), np.expand_dims(factory_tiles, 0))
                argmin_ore_tile, argmin_factory_tile = np.unravel_index(np.argmin(ore_tile_distances), ore_tile_distances.shape)
                closest_ore_tile = ore_tile_locations[argmin_ore_tile]
                closest_ore_factory_tile = factory_tiles[argmin_factory_tile]

                ice_route = get_shortest_loop(rubble_map, closest_ice_factory_tile, closest_ice_tile, [closest_ore_tile, closest_ore_factory_tile])
                if ice_route is None:
                    raise ValueError("No ice route found")
                for tile in ice_route.path:
                    rubble_map[tile[0], tile[1]] = -1
                ore_route = get_shortest_loop(rubble_map, closest_ore_factory_tile, closest_ore_tile)
                if ore_route is None:
                    raise ValueError("No ore route found")
                print(f"Factory {factory_id} ice route: {ice_route.path}", file=sys.stderr)
                self.factory_states[factory_id] = FactoryState(
                    resources=dict(
                        ice=ResourcePlan(
                            resource_pos=closest_ice_tile,
                            resource_factory_pos=closest_ice_factory_tile,
                            resource_route=ice_route,
                            max_resource_robots=len(ice_route)-1,
                            resource_threshold_light=8,
                        ),
                        ore=ResourcePlan(
                            resource_pos=closest_ore_tile,
                            resource_factory_pos=closest_ore_factory_tile,
                            resource_route=ore_route,
                            max_resource_robots=len(ore_route)-1,
                            resource_threshold_light=8,
                        ),
                    ),
                )

        # Remove robots from factories if factory is destroyed
        factory_ids_to_destroy = []
        for factory_id in self.robots_ice_factory.keys():
            if not factory_id in factory_ids:
                for unit_id in self.robots_ice_factory[factory_id]:
                    del self.units_master_factory[unit_id]
                    del self.action_start_power[unit_id]
                    del self.unit_states[unit_id]
                for unit_id in self.robots_ore_factory[factory_id]:
                    del self.units_master_factory[unit_id]
                    del self.action_start_power[unit_id]
                    del self.unit_states[unit_id]
                for unit_id in self.robots_rubble_factory[factory_id]:
                    del self.units_master_factory[unit_id]
                    del self.action_start_power[unit_id]
                    del self.unit_states[unit_id]
                factory_ids_to_destroy.append(factory_id)
        for factory_id in factory_ids_to_destroy:
            del self.robots_ice_factory[factory_id]
            del self.robots_ore_factory[factory_id]
            del self.robots_rubble_factory[factory_id]
        del factory_ids_to_destroy

        # Remove robots from factories if they are dead
        for factory_id, factory in factories.items():
            for robot_list in [self.robots_ice_factory[factory_id], self.robots_ore_factory[factory_id], self.robots_rubble_factory[factory_id]]:
                for unit_id in list(robot_list):
                    if not unit_id in units:
                        robot_list.remove(unit_id)
                        del self.units_master_factory[unit_id]
                        del self.action_start_power[unit_id]
                        del self.unit_states[unit_id]

        # Register robots to factories if not registered
        for unit_id, unit in units.items():
            if unit_id not in self.units_master_factory:
                factory_distances = taxi_dist(factory_centers, unit.pos)
                factory_id = factory_ids[np.argmin(factory_distances)]
                factory_state = self.factory_states[factory_id]
                self.units_master_factory[unit_id] = factory_id
                self.action_start_power[unit_id] = None    
                
                need_more_ice_robot = len(self.robots_ice_factory[factory_id]) < factory_state.resources["ice"].max_resource_robots
                need_more_ore_robot = len(self.robots_ore_factory[factory_id]) < factory_state.resources["ore"].max_resource_robots
                if need_more_ice_robot and need_more_ore_robot:
                    # Allocate robot
                    if unit.unit_type == "LIGHT":
                        if self.num_lights_ice_factory(factory_id, units) <= self.num_lights_ore_factory(factory_id, units):
                            self.add_unit_to_factory(factory_id, unit_id, factory_state, "ice")
                        else:
                            self.add_unit_to_factory(factory_id, unit_id, factory_state, "ore")
                    elif unit.unit_type == "HEAVY":
                        if self.num_heavies_ice_factory(factory_id, units) <= self.num_heavies_ore_factory(factory_id, units):
                            self.add_unit_to_factory(factory_id, unit_id, factory_state, "ice")
                        else:
                            self.add_unit_to_factory(factory_id, unit_id, factory_state, "ore")
                    else:
                        raise ValueError("Unknown unit type")
                elif need_more_ice_robot:
                    self.add_unit_to_factory(factory_id, unit_id, factory_state, "ice")
                elif need_more_ore_robot:
                    self.add_unit_to_factory(factory_id, unit_id, factory_state, "ore")
                else:
                    self.robots_rubble_factory[factory_id].append(unit_id)
                    self.unit_states[unit_id] = UnitState()
        # handle action of robots bound to factories
        for factory_id, factory in factories.items():
            factory_state = self.factory_states[factory_id]
            # will implicitly handle robot reallocation
            ice_robots = self.robots_ice_factory[factory_id].copy()
            ore_robots = self.robots_ore_factory[factory_id].copy()
            rubble_robots = self.robots_rubble_factory[factory_id].copy()
            factory_pickup_robots = 0
            for robot_list in [ice_robots, ore_robots, rubble_robots]:
                for unit_id in robot_list:
                    unit = units[unit_id]
                    factory_inf_distance = norm(factory.pos - unit.pos, ord=np.inf)
                    can_pickup_from_factory = factory_inf_distance <= 1
                    if can_pickup_from_factory and self.unit_states[unit_id].state == UnitStateEnum.RECHARGING:
                        if unit.unit_type == "LIGHT":
                            factory_pickup_robots += 1
                        else:
                            factory_pickup_robots += 10

            # handle robots adding ice to factory
            for unit_id in ice_robots:
                unit = units[unit_id]
                actions = self.handle_robot_resource_gathering(game_state, factory, factory_state, unit, "ice", actions, factory_pickup_robots)
            # handle robots adding ore to factory
            for unit_id in ore_robots:
                unit = units[unit_id]
                actions = self.handle_robot_resource_gathering(game_state, factory, factory_state, unit, "ore", actions, factory_pickup_robots)

            # handle factory action
            n_lights = self.num_lights_ore_factory(factory_id, units) + self.num_lights_ice_factory(factory_id, units)
            n_heavies = self.num_heavies_ore_factory(factory_id, units) + self.num_heavies_ice_factory(factory_id, units)

            # if factory can manage current water usage, then water
            #if factory.cargo.water > 100:
            #    actions[factory_id] = factory.water()
            remaining_steps = self.env_cfg.max_episode_length - game_state.real_env_steps
            water_cost = factory.water_cost(game_state)
            spreads = (remaining_steps / self.env_cfg.MIN_LICHEN_TO_SPREAD)
            multiple = (spreads * (spreads + 1) * (2 * spreads + 1) / 6) * self.env_cfg.MIN_LICHEN_TO_SPREAD
            estimated_water_cost = water_cost * multiple
            if (estimated_water_cost + remaining_steps) <= factory.cargo.water:
                actions[factory_id] = factory.water()
            elif remaining_steps == 1 and water_cost < factory.cargo.water:
                actions[factory_id] = factory.water()
            else: # or build robots
                if factory.can_build_heavy(game_state):
                    actions[factory_id] = factory.build_heavy()
                #elif n_lights <= n_heavies * 4 and factory.can_build_light(game_state):
                #   actions[factory_id] = factory.build_light()

        actions = stop_movement_collisions(obs, game_state, self.env_cfg, self.player, actions, self.unit_states)
        return actions


def average_kernel(size):
    return np.ones((size, size)) / (size * size)

def conv2d(a, f, pad='zero', n=1):
    if pad == 'zero':
        pad = (f.shape[0] - 1) // 2

    strd = np.lib.stride_tricks.as_strided
    a = np.pad(a, pad)
    s = f.shape + tuple(np.subtract(a.shape, f.shape) + 1)
    for i in range(n):
        if i > 0:
            a = np.pad(a, pad)
        subM = strd(a, shape = s, strides = a.strides * 2)
        a = np.einsum('ij,ijkl->kl', f, subM)
    return a


def main(env, agents, steps, seed):
    # reset our env
    obs, _ = env.reset(seed=seed)
    np.random.seed(0)

    step = 0
    # Note that as the environment has two phases, we also keep track a value called 
    # `real_env_steps` in the environment state. The first phase ends once `real_env_steps` is 0 and used below

    # iterate until phase 1 ends
    while env.state.real_env_steps < 0:
        if step >= steps: break
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
        if step >= steps: break
        actions = {}
        for player in env.agents:
            o = obs[player]
            a = agents[player].act(step, o)
            actions[player] = a
        step += 1
        obs, rewards, terminations, truncations, infos = env.step(actions)
        dones = {k: terminations[k] or truncations[k] for k in terminations}
        done = dones["player_0"] and dones["player_1"]
    
if __name__=='__main__':
    from luxai_s2.env import LuxAI_S2
    env = LuxAI_S2() # create the environment object
    agents = {player: Agent(player, env.state.env_cfg) for player in ['player_0', 'player_1']}
    main(env, agents, 100, 101)