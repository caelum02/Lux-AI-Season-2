from lux.kit import obs_to_game_state, GameState, EnvConfig
from lux.utils import direction_to, my_turn_to_place_factory
import numpy as np
import sys
class Agent():
    def __init__(self, player: str, env_cfg: EnvConfig) -> None:
        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        np.random.seed(0)
        self.env_cfg: EnvConfig = env_cfg
        self.factory_score = None

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
                self.factory_score += conv2d(game_state.board.rubble, average_kernel(5), n=3)
                ice_tile_locations = np.argwhere(game_state.board.ice == 1)
                all_locations = np.mgrid[:map_size, :map_size].swapaxes(0, 2).reshape(-1, 2)
                distances = np.linalg.norm(np.expand_dims(all_locations, 1) - np.expand_dims(ice_tile_locations, 0), ord=1, axis=-1)
                distances = np.min(distances, axis=-1)
                distances = distances.reshape(map_size, map_size)
                # distances[x][y] is the distance to the nearest ice tile 
                self.factory_score += distances
            
            # how much water and metal you have in your starting pool to give to new factories
            water_left = game_state.teams[self.player].water
            metal_left = game_state.teams[self.player].metal
            
            # how many factories you have left to place
            factories_to_place = game_state.teams[self.player].factories_to_place
            # whether it is your turn to place a factory
            my_turn_to_place = my_turn_to_place_factory(game_state.teams[self.player].place_first, step)
            if factories_to_place > 0 and my_turn_to_place:
                # we will spawn our factory in a random location with 150 metal and water if it is our turn to place
                factory_score = self.factory_score + (obs["board"]["valid_spawns_mask"] == 0) * 1e9
                spawn_loc = np.argmin(factory_score)
                map_size = self.env_cfg.map_size
                spawn_loc = np.array([spawn_loc // map_size, spawn_loc % map_size])
                return dict(spawn=spawn_loc, metal=150, water=150)
            return dict()

    def act(self, step: int, obs, remainingOverageTime: int = 60):
        actions = dict()
        game_state = obs_to_game_state(step, self.env_cfg, obs)
        factories = game_state.factories[self.player]
        factory_tiles, factory_units = [], []
        for unit_id, factory in factories.items():
            if factory.power >= self.env_cfg.ROBOTS["HEAVY"].POWER_COST and \
            factory.cargo.metal >= self.env_cfg.ROBOTS["HEAVY"].METAL_COST:
                actions[unit_id] = factory.build_heavy()
            factory_tiles += [factory.pos]
            factory_units += [factory]
        factory_tiles = np.array(factory_tiles)

        units = game_state.units[self.player]
        ice_map = game_state.board.ice
        ice_tile_locations = np.argwhere(ice_map == 1)
        for unit_id, unit in units.items():
            
            # track the closest factory
            closest_factory = None
            adjacent_to_factory = False
            if len(factory_tiles) > 0:
                factory_distances = np.linalg.norm(factory_tiles - unit.pos, ord=1, axis=1)
                closest_factory_tile = factory_tiles[np.argmin(factory_distances)]
                closest_factory = factory_units[np.argmin(factory_distances)]
                adjacent_to_factory = np.linalg.norm(closest_factory_tile - unit.pos, ord=1) == 0

                ice_threshold = 40
                # previous ice mining code
                if adjacent_to_factory and unit.power < unit.unit_cfg.INIT_POWER:
                    actions[unit_id] = [unit.pickup(4, unit.unit_cfg.BATTERY_CAPACITY, repeat=0, n=1)]
                    # 4 means power
                elif unit.cargo.ice < ice_threshold:
                    ice_tile_distances = np.linalg.norm(ice_tile_locations - unit.pos, ord=1, axis=1)
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
                    direction = direction_to(unit.pos, closest_factory_tile)
                    if adjacent_to_factory:
                        if unit.power >= unit.action_queue_cost(game_state):
                            actions[unit_id] = [unit.transfer(direction, 0, unit.cargo.ice, repeat=0, n=1)]
                        
                    else:
                        move_cost = unit.move_cost(game_state, direction)
                        if move_cost is not None and unit.power >= move_cost + unit.action_queue_cost(game_state):
                            actions[unit_id] = [unit.move(direction, repeat=0, n=1)]
        return actions
