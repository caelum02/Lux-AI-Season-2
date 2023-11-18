import sys
from lux.kit import obs_to_game_state, GameState, EnvConfig
from lux.utils import direction_to, my_turn_to_place_factory
from lux.forward_sim import stop_movement_collisions
import numpy as np
from numpy.linalg import norm

from lux.states import UnitStates

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

        self.action_start_power = dict()

        self.unit_states = dict() # unit_id -> UnitStates

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
                ice_tile_locations = np.argwhere(game_state.board.ice.T == 1)
                ore_tile_locations = np.argwhere(game_state.board.ore.T == 1)
                all_locations = np.mgrid[:map_size, :map_size].swapaxes(0, 2).reshape(-1, 2)
                ice_distances = np.linalg.norm(np.expand_dims(all_locations, 1) - np.expand_dims(ice_tile_locations, 0), ord=1, axis=-1)
                ice_distances = np.min(ice_distances, axis=-1)
                ice_distances = ice_distances.reshape(map_size, map_size)

                ore_distances = np.linalg.norm(np.expand_dims(all_locations, 1) - np.expand_dims(ore_tile_locations, 0), ord=1, axis=-1)
                ore_distances = np.min(ore_distances, axis=-1)
                ore_distances = ore_distances.reshape(map_size, map_size)

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
                
                spawn_loc = np.argmin(factory_score)
                map_size = self.env_cfg.map_size
                
                spawn_loc = np.array([spawn_loc // map_size, spawn_loc % map_size])
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

        factory_tiles, factory_units = [], []
        factory_ids = []
        for factory_id, factory in factories.items():
            factory_tiles += [factory.pos]
            factory_units += [factory]
            factory_ids += [factory_id]           

        return factory_tiles, factory_units, factory_ids

    def _initialize_robot_bindings(self, factories):
        for factory_id in factories.keys():
            self.robots_ice_factory[factory_id] = list()
            self.robots_ore_factory[factory_id] = list()

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
        factory_tiles, factory_units, factory_ids = self._get_factory_misc(factories)
        units = game_state.units[self.player]
        
        if obs['real_env_steps'] == 0:
            self._initialize_robot_bindings(factories)
    
        ice_map = game_state.board.ice
        ice_tile_locations = np.argwhere(ice_map == 1)
        ore_map = game_state.board.ore
        ore_tile_locations = np.argwhere(ore_map == 1)

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
                factory_ids_to_destroy.append(factory_id)
        for factory_id in factory_ids_to_destroy:
            del self.robots_ice_factory[factory_id]
            del self.robots_ore_factory[factory_id]
        del factory_ids_to_destroy
        # Remove robots from factories if they are dead
        for factory_id, factory in factories.items():
            for robot_list in [self.robots_ice_factory[factory_id], self.robots_ore_factory[factory_id]]:
                for unit_id in list(robot_list):
                    if not unit_id in units:
                        robot_list.remove(unit_id)
                        del self.units_master_factory[unit_id]
                        del self.action_start_power[unit_id]
                        del self.unit_states[unit_id]

        # Register robots to factories if not registered
        for unit_id, unit in units.items():
            if unit_id not in self.units_master_factory:
                factory_distances = norm(factory_tiles - unit.pos, ord=1, axis=1)
                factory_id = factory_ids[np.argmin(factory_distances)]
                
                self.units_master_factory[unit_id] = factory_id
                self.action_start_power[unit_id] = None    
                
                # Allocate robot
                if unit.unit_type == "LIGHT":
                    if self.num_lights_ice_factory(factory_id, units) <= self.num_lights_ore_factory(factory_id, units):
                        self.robots_ice_factory[factory_id].append(unit_id)
                    else:
                        self.robots_ore_factory[factory_id].append(unit_id)
                elif unit.unit_type == "HEAVY":
                    if self.num_heavies_ice_factory(factory_id, units) <= self.num_heavies_ore_factory(factory_id, units):
                        self.robots_ice_factory[factory_id].append(unit_id)
                    else:
                        self.robots_ore_factory[factory_id].append(unit_id)
                else:
                    raise ValueError("Unknown unit type")        
            if unit_id not in self.unit_states:
                self.unit_states[unit_id] = UnitStates.MOVING_TO_RESOURCE
        loop_turns = 10 if is_day else 6
        assert loop_turns <= self.env_cfg.DAY_LENGTH
        min_dig_turns = 3
        # handle action of robots bound to factories
        for factory_id, factory in factories.items():

            # will implicitly handle robot reallocation
            ice_robots = self.robots_ice_factory[factory_id].copy()
            ore_robots = self.robots_ore_factory[factory_id].copy()

            # handle robots adding ice to factory
            for unit_id in ice_robots:
                unit = units[unit_id]
                factory_inf_distance = norm(factory.pos - unit.pos, ord=np.inf)
                can_pickup_from_factory = factory_inf_distance <= 1
                can_transfer_to_factory = factory_inf_distance <= 2

                ice_tile_distances = np.linalg.norm(ice_tile_locations - unit.pos, ord=1, axis=1)
                closest_ice_tile = ice_tile_locations[np.argmin(ice_tile_distances)]
                dig_turns = max(min_dig_turns, loop_turns - (2 * norm(closest_ice_tile - factory.pos, ord=1)))
                target_power = unit.unit_cfg.DIG_COST * dig_turns
                target_power += (unit.unit_cfg.MOVE_COST + unit.unit_cfg.RUBBLE_MOVEMENT_COST * self.rubble_score[unit.pos[0], unit.pos[1]]) * loop_turns
                if is_day:
                    target_power += unit.unit_cfg.ACTION_QUEUE_POWER_COST * max(0, loop_turns - remaining_days)
                else:
                    target_power += unit.unit_cfg.ACTION_QUEUE_POWER_COST * max(loop_turns, remaining_nights)
                target_power += unit.unit_cfg.ACTION_QUEUE_POWER_COST * loop_turns * 0.3  # spare power
                target_power = int(target_power)
                ice_threshold = min(unit.unit_cfg.CARGO_SPACE, unit.unit_cfg.DIG_RESOURCE_GAIN * dig_turns)
                # previous ice mining code
                
                if self.unit_states[unit_id] == UnitStates.MOVING_TO_RESOURCE:
                    if np.all(closest_ice_tile == unit.pos):
                        self.unit_states[unit_id] = UnitStates.DIGGING_RESOURCE
                        if unit.power >= unit.dig_cost(game_state) + unit.action_queue_cost(game_state):
                            actions[unit_id] = [unit.dig(repeat=0, n=1)]
                    else:
                        direction = direction_to(unit.pos, closest_ice_tile)
                        move_cost = unit.move_cost(game_state, direction)
                        if move_cost is not None and unit.power >= move_cost + unit.action_queue_cost(game_state):
                            actions[unit_id] = [unit.move(direction, repeat=0, n=1)]
                elif self.unit_states[unit_id] == UnitStates.DIGGING_RESOURCE:
                    if unit.cargo.ice < ice_threshold:
                        if unit.power >= unit.dig_cost(game_state) + unit.action_queue_cost(game_state):
                            actions[unit_id] = [unit.dig(repeat=0, n=1)]
                    else:
                        self.unit_states[unit_id] = UnitStates.MOVING_TO_FACTORY
                        direction = direction_to(unit.pos, factory.pos)
                        if can_transfer_to_factory:
                            self.unit_states[unit_id] = UnitStates.DROPPING_OFF_RESOURCE
                            if unit.power >= unit.action_queue_cost(game_state):
                                actions[unit_id] = [unit.transfer(direction, 0, unit.cargo.ice, repeat=0, n=1)]
                                self.unit_states[unit_id] = UnitStates.RECHARING
                        else:
                            move_cost = unit.move_cost(game_state, direction)
                            if move_cost is not None and unit.power >= move_cost + unit.action_queue_cost(game_state):
                                actions[unit_id] = [unit.move(direction, repeat=0, n=1)]
                elif self.unit_states[unit_id] == UnitStates.MOVING_TO_FACTORY:
                    direction = direction_to(unit.pos, factory.pos)
                    if can_transfer_to_factory:
                        self.unit_states[unit_id] = UnitStates.DROPPING_OFF_RESOURCE
                        if unit.power >= unit.action_queue_cost(game_state):
                            actions[unit_id] = [unit.transfer(direction, 0, unit.cargo.ice, repeat=0, n=1)]
                            self.unit_states[unit_id] = UnitStates.RECHARING
                    else:
                        move_cost = unit.move_cost(game_state, direction)
                        if move_cost is not None and unit.power >= move_cost + unit.action_queue_cost(game_state):
                            actions[unit_id] = [unit.move(direction, repeat=0, n=1)]
                elif self.unit_states[unit_id] == UnitStates.DROPPING_OFF_RESOURCE:
                    if unit.power >= unit.action_queue_cost(game_state):
                        actions[unit_id] = [unit.transfer(direction, 0, unit.cargo.ice, repeat=0, n=1)]
                        self.unit_states[unit_id] = UnitStates.RECHARING
                        if unit.power < target_power:
                            if can_pickup_from_factory:
                                if target_power - unit.power <= factory.power:
                                    power = target_power - unit.power
                                    actions[unit_id] = [unit.pickup(4, power, repeat=0, n=1)]
                                    self.unit_states[unit_id] = UnitStates.MOVING_TO_RESOURCE
                                else:
                                    actions[unit_id] = []
                                    del actions[unit_id]  # Save action queue update cost
                            else:
                                direction = direction_to(unit.pos, factory.pos)
                                move_cost = unit.move_cost(game_state, direction)
                                if move_cost is not None and unit.power >= move_cost + unit.action_queue_cost(game_state):
                                    actions[unit_id] = [unit.move(direction, repeat=0, n=1)]
                        else:
                            self.unit_states[unit_id] = UnitStates.MOVING_TO_RESOURCE
                elif self.unit_states[unit_id] == UnitStates.RECHARING:
                    if unit.power < target_power:
                        if can_pickup_from_factory:
                            if target_power - unit.power <= factory.power:
                                power = target_power - unit.power
                                actions[unit_id] = [unit.pickup(4, power, repeat=0, n=1)]
                                self.unit_states[unit_id] = UnitStates.MOVING_TO_RESOURCE
                            else:
                                actions[unit_id] = []
                                del actions[unit_id]  # Save action queue update cost
                        else:
                            direction = direction_to(unit.pos, factory.pos)
                            move_cost = unit.move_cost(game_state, direction)
                            if move_cost is not None and unit.power >= move_cost + unit.action_queue_cost(game_state):
                                actions[unit_id] = [unit.move(direction, repeat=0, n=1)]
                    else:
                        self.unit_states[unit_id] = UnitStates.MOVING_TO_RESOURCE
            # handle robots adding ore to factory
            for unit_id in ore_robots:
                unit = units[unit_id]
                factory_inf_distance = norm(factory.pos - unit.pos, ord=np.inf)
                can_pickup_from_factory = factory_inf_distance <= 1
                can_transfer_to_factory = factory_inf_distance <= 2
                ore_threshold = min(unit.unit_cfg.CARGO_SPACE * 0.8, unit.unit_cfg.DIG_RESOURCE_GAIN * 6)
                
                ore_tile_distances = norm(ore_tile_locations - unit.pos, ord=1, axis=1)
                closest_ore_tile = ore_tile_locations[np.argmin(ore_tile_distances)]
                dig_turns = max(min_dig_turns, loop_turns - (2 * norm(closest_ore_tile - factory.pos, ord=1)))
                if can_transfer_to_factory and unit.cargo.ore == 0 and unit.power < unit.unit_cfg.INIT_POWER:
                    if can_pickup_from_factory:
                        power = max(unit.unit_cfg.INIT_POWER * 2 - unit.power, 0)
                        actions[unit_id] = [unit.pickup(4, power, repeat=0, n=1)]
                    else:
                        direction = direction_to(unit.pos, factory.pos)
                        move_cost = unit.move_cost(game_state, direction)
                        if move_cost is not None and unit.power >= move_cost + unit.action_queue_cost(game_state):
                            actions[unit_id] = [unit.move(direction, repeat=0, n=1)]
                    # 4 means power
                elif unit.cargo.ore < ore_threshold:
                    
                    if np.all(closest_ore_tile == unit.pos):
                        if unit.power >= unit.dig_cost(game_state) + unit.action_queue_cost(game_state):
                            actions[unit_id] = [unit.dig(repeat=0, n=1)]
                    else:
                        direction = direction_to(unit.pos, closest_ore_tile)

                        move_cost = unit.move_cost(game_state, direction)
                        if move_cost is not None and unit.power >= move_cost + unit.action_queue_cost(game_state):
                            actions[unit_id] = [unit.move(direction, repeat=0, n=1)]
                # else if we have enough ore, we go back to the factory and dump it.
                elif unit.cargo.ore >= ore_threshold:
                    direction = direction_to(unit.pos, factory.pos)
                    if can_transfer_to_factory and can_pickup_from_factory:  # NOTE workaround of statekeeping
                        if unit.power >= unit.action_queue_cost(game_state):
                            actions[unit_id] = [unit.transfer(direction, 1, unit.cargo.ore, repeat=0, n=1)]
                            
                            # Reallocate robot
                            #if len(self.robots_ice_factory[factory_id]) < 3:
                            #    self.robots_ore_factory[factory_id].remove(unit_id)
                            #    self.robots_ice_factory[factory_id].append(unit_id)
                        
                    else:
                        move_cost = unit.move_cost(game_state, direction)
                        if move_cost is not None and unit.power >= move_cost + unit.action_queue_cost(game_state):
                            actions[unit_id] = [unit.move(direction, repeat=0, n=1)]

            # handle factory action
            n_lights = self.num_lights_ore_factory(factory_id, units) + self.num_lights_ice_factory(factory_id, units)
            n_heavies = self.num_heavies_ore_factory(factory_id, units) + self.num_heavies_ice_factory(factory_id, units)

            # if factory can manage current water usage, then water
            #if factory.cargo.water > 100:
            #    actions[factory_id] = factory.water()
            remaining_steps = self.env_cfg.max_episode_length - game_state.real_env_steps
            if remaining_steps < 100:
                if factory.water_cost(game_state) <= factory.cargo.water - remaining_steps:
                    actions[factory_id] = factory.water()
            else: # or build robots
                if factory.can_build_heavy(game_state):
                    actions[factory_id] = factory.build_heavy()
                #elif n_lights <= n_heavies * 4 and factory.can_build_light(game_state):
                #   actions[factory_id] = factory.build_light()

        actions = stop_movement_collisions(obs, game_state, self.env_cfg, self.player, actions)

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