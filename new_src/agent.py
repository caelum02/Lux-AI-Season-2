from typing import Dict
from matplotlib import pyplot as plt
from lux.kit import obs_to_game_state, GameState, EnvConfig
from lux.utils import direction_to, my_turn_to_place_factory
from lux.forward_sim import stop_movement_collisions
import numpy as np
from numpy.linalg import norm
import sys
class Agent():
    def __init__(self, player: str, env_cfg: EnvConfig) -> None:
        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        np.random.seed(0)
        self.env_cfg: EnvConfig = env_cfg
        self.factory_score = None

        self.units_master_factory = dict() # unit_id -> factory_id

        self.robots_ice_factory = dict() # factory_id -> {unit_id: unit}
        self.robots_ore_factory = dict() # factory_id -> {unit_id: unit}

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
                self.factory_score += conv2d(game_state.board.rubble, average_kernel(5), n=3) * 0.05
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
                self.factory_score += np.clip(ice_distances-3, a_min=0, a_max=None) + np.clip(ore_distances-3, a_min=0, a_max=None) * 0.3
            
            # how much water and metal you have in your starting pool to give to new factories
            water_left = game_state.teams[self.player].water
            metal_left = game_state.teams[self.player].metal
            
            # how many factories you have left to place
            factories_to_place = game_state.teams[self.player].factories_to_place
            # whether it is your turn to place a factory
            my_turn_to_place = my_turn_to_place_factory(game_state.teams[self.player].place_first, step)
            if factories_to_place > 0 and my_turn_to_place:
                # Build factory at the position with the lowest factory_score
                factory_score = self.factory_score + (obs["board"]["valid_spawns_mask"] == 0) * 1e9
                

                spawn_loc = np.argmin(factory_score)
                map_size = self.env_cfg.map_size
                
                spawn_loc = np.array([spawn_loc // map_size, spawn_loc % map_size])
                return dict(spawn=spawn_loc, metal=150, water=150)
            return dict()

    def _get_factory_misc(self, factories: Dict[str, "Factory"]):

        factory_tiles, factory_units = [], []
        factory_ids = []
        for factory_id, factory in factories.items():
            factory_tiles += [factory.pos]
            factory_units += [factory]
            factory_ids += [factory_id]           

        return factory_tiles, factory_units, factory_ids

    def _initialize_robot_bindings(self, factories):
        for factory_id in factories.keys():
            self.robots_ice_factory[factory_id] = dict()
            self.robots_ore_factory[factory_id] = dict()

    @staticmethod
    def _num_robot_type_from_list(robot_dict, robot_type):
        num_robots = 0
        for unit_id, unit in robot_dict.items():
            if unit.unit_type == robot_type:
                num_robots += 1
        return num_robots
    
    
    def num_lights_ore_factory(self, factory_id):
        return self._num_robot_type_from_list(self.robots_ore_factory[factory_id], "LIGHT")

    
    def num_heavies_ore_factory(self, factory_id):
        return self._num_robot_type_from_list(self.robots_ore_factory[factory_id], "HEAVY")
    
    
    def num_lights_ice_factory(self, factory_id):
        return self._num_robot_type_from_list(self.robots_ice_factory[factory_id], "LIGHT")
    
    
    def num_heavies_ice_factory(self, factory_id):
        return self._num_robot_type_from_list(self.robots_ice_factory[factory_id], "HEAVY")

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
        
        
        factories = game_state.factories[self.player]
        factory_tiles, factory_units, factory_ids = self._get_factory_misc(factories)
        units = game_state.units[self.player]
        
        if obs['real_env_steps'] == 0:
            self._initialize_robot_bindings(factories)
    
        ice_map = game_state.board.ice
        ice_tile_locations = np.argwhere(ice_map == 1)
        ore_map = game_state.board.ore
        ore_tile_locations = np.argwhere(ore_map == 1)

        # TODO Remove robots from factories factory is dead
        
        # Register robots to factories if not registered
        for unit_id, unit in units.items():
            if not unit_id in self.units_master_factory:
                factory_distances = norm(factory_tiles - unit.pos, ord=1, axis=1)
                factory_id = factory_ids[np.argmin(factory_distances)]
                
                self.units_master_factory[unit_id] = factory_id
                
                # Allocate robot
                if unit.unit_type == "LIGHT":
                    if self.num_lights_ice_factory(factory_id) < self.num_lights_ore_factory(factory_id):
                        self.robots_ice_factory[factory_id][unit_id] = unit
                    else:
                        self.robots_ore_factory[factory_id][unit_id] = unit
                else:
                    if self.num_heavies_ice_factory(factory_id) < self.num_heavies_ore_factory(factory_id):
                        self.robots_ice_factory[factory_id][unit_id] = unit
                    else:
                        self.robots_ore_factory[factory_id][unit_id] = unit
        
        # Remove robots from factories if they are dead
        for factory_id, factory in factories.items():
            for robot_dict in [self.robots_ice_factory[factory_id], self.robots_ore_factory[factory_id]]:
                for unit_id, unit in list(robot_dict.items()):
                    if not unit_id in units:
                        del robot_dict[unit_id]
                        del self.units_master_factory[unit_id]
  
        # handle action of robots bound to factories
        for factory_id, factory in factories.items():
            # handle robots adding ice to factory
            for unit_id, unit in self.robots_ice_factory[factory_id].items():
                
                adjacent_to_factory = norm(factory.pos - unit.pos, ord=np.inf) <= 1
                ice_threshold = 40
                # previous ice mining code
                if adjacent_to_factory and unit.power < unit.unit_cfg.INIT_POWER:
                    actions[unit_id] = [unit.pickup(4, unit.unit_cfg.BATTERY_CAPACITY, repeat=0, n=1)]
                    # 4 means power
                elif unit.cargo.ice < ice_threshold:
                    ice_tile_distances = norm(ice_tile_locations - unit.pos, ord=1, axis=1)
                    closest_ice_tile = ice_tile_locations[np.argmin(ice_tile_distances)]
                    if np.all(closest_ice_tile == unit.pos):
                        if unit.power >= unit.dig_cost(game_state) + unit.action_queue_cost(game_state):
                            actions[unit_id] = [unit.dig(repeat=0, n=1)]
                    else:
                        direction = direction_to(unit.pos, closest_ice_tile)
                        move_cost = unit.move_cost(game_state, direction)
                        if move_cost is not None and unit.power >= move_cost + unit.action_queue_cost(game_state):
                            actions[unit_id] = [unit.move(direction, repeat=0, n=1)]
                # else if we have enough ice, we go back to the factory and dump it.
                elif unit.cargo.ice >= ice_threshold:
                    direction = direction_to(unit.pos, factory.pos)
                    if adjacent_to_factory:
                        if unit.power >= unit.action_queue_cost(game_state):
                            actions[unit_id] = [unit.transfer(direction, 0, unit.cargo.ice, repeat=0, n=1)]
                        
                    else:
                        move_cost = unit.move_cost(game_state, direction)
                        if move_cost is not None and unit.power >= move_cost + unit.action_queue_cost(game_state):
                            actions[unit_id] = [unit.move(direction, repeat=0, n=1)]
            
            # handle robots adding ore to factory
            for unit_id, unit in self.robots_ore_factory[factory_id].items():
                adjacent_to_factory = norm(factory.pos - unit.pos, ord=np.inf) <= 1
                ore_threshold = 40
                # previous ore mining code
                if adjacent_to_factory and unit.power < unit.unit_cfg.INIT_POWER:
                    actions[unit_id] = [unit.pickup(4, unit.unit_cfg.BATTERY_CAPACITY, repeat=0, n=1)]
                    # 4 means power
                elif unit.cargo.ore < ore_threshold:
                    ore_tile_distances = norm(ore_tile_locations - unit.pos, ord=1, axis=1)
                    closest_ore_tile = ore_tile_locations[np.argmin(ore_tile_distances)]
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
                    if adjacent_to_factory:
                        if unit.power >= unit.action_queue_cost(game_state):
                            actions[unit_id] = [unit.transfer(direction, 0, unit.cargo.ore, repeat=0, n=1)]
                        
                    else:
                        move_cost = unit.move_cost(game_state, direction)
                        if move_cost is not None and unit.power >= move_cost + unit.action_queue_cost(game_state):
                            actions[unit_id] = [unit.move(direction, repeat=0, n=1)]

            # handle factory action
            n_lights = self.num_lights_ore_factory(factory_id) + self.num_lights_ice_factory(factory_id)
            n_heavies = self.num_heavies_ore_factory(factory_id) + self.num_heavies_ice_factory(factory_id)

            # if factory can manage current water usage, then water
            remaining_steps = self.env_cfg.max_episode_length - game_state.real_env_steps
            if remaining_steps < 70:
                if factory.water_cost(game_state) <= factory.cargo.water - remaining_steps:
                    actions[unit_id] = factory.water()

            else: # or build robots
                if n_lights > n_heavies * 10 and factory.can_build_heavy(game_state):
                    actions[factory_id] = factory.build_heavy()

                elif n_lights <= n_heavies * 8 and factory.can_build_light(game_state):
                    actions[factory_id] = factory.build_light()

        
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


def main(env, agents, steps):
    # reset our env
    obs, _ = env.reset()
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
    main(env, agents, 100)