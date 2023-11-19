from collections import defaultdict
from typing import Dict, List, Set, Tuple, Union

from lux.unit import Unit
import numpy as np
import sys

from action_enum import ACTION_T, DIRECTION_T

def forward_sim(full_obs, env_cfg, n=2):
    """
    Forward sims for `n` steps given the current full observation and env_cfg

    If forward sim leads to the end of a game, it won't return any additional observations, just the original one
    """
    from luxai_s2 import LuxAI_S2
    from luxai_s2.config import UnitConfig
    import copy
    agent = "player_0"
    env = LuxAI_S2(collect_stats=False, verbose=0)
    env.reset(seed=0)
    env.state = env.state.from_obs(full_obs, env_cfg)
    env.env_cfg = env.state.env_cfg
    env.env_cfg.verbose = 0
    env.env_steps = env.state.env_steps
    forward_obs = [full_obs]
    for _ in range(n):
        empty_actions = dict()
        for agent in env.agents:
            empty_actions[agent] = dict()
        if len(env.agents) == 0:
            # can't step any further
            return [full_obs]
        obs, _, _, _, _ = env.step(empty_actions)
        forward_obs.append(obs[agent])
    return forward_obs

def forward_sim_act(full_obs, env_cfg, player, action):
    from luxai_s2 import LuxAI_S2
    # from luxai_s2.config import UnitConfig
    # import copy
    env = LuxAI_S2(collect_stats=False, verbose=0)
    env.reset(seed=0)
    env.state = env.state.from_obs(full_obs, env_cfg)
    env.env_cfg = env.state.env_cfg
    env.env_cfg.verbose = 0
    env.env_steps = env.state.env_steps
    forward_obs = [full_obs]
    empty_actions = dict()
    for agent in env.agents:
        empty_actions[agent] = dict()
    empty_actions[player] = action
    obs, _, _, _, _ = env.step(empty_actions)
    return obs[player]

move_deltas = np.array([[0, 0], [0, -1], [1, 0], [0, 1], [-1, 0]])

def stop_movement_collisions(obs, game_state, env_cfg, agent, actions):
    units_map = defaultdict(list)
    move_actions = []

    for unit in game_state.units[agent].values():
        units_map[tuple(unit.pos)].append(unit)
        unit_action = None
        if unit.unit_id in actions:
            unit_action = actions[unit.unit_id][0]
        elif unit.action_queue:
            unit_action = unit.action_queue[0]
        
        if unit_action is not None and unit_action[0] == ACTION_T.MOVE:
            move_actions.append((unit, unit_action))

    new_units_map: Dict[str, List[Unit]] = defaultdict(list)
    heavy_entered_pos: Dict[str, List[Unit]] = defaultdict(list)
    light_entered_pos: Dict[str, List[Unit]] = defaultdict(list)
    
    for unit, move_action in move_actions:
        # skip move center
        direction = move_action[1]
        if direction != DIRECTION_T.CENTER:
            old_pos_hash = tuple(unit.pos)
            target_pos = (
                unit.pos + move_deltas[direction]
            )
            new_pos_hash = tuple(target_pos)

            # Remove moving units from units_map
            if len(units_map[old_pos_hash]) == 1:
                del units_map[old_pos_hash]
            else:
                units_map[old_pos_hash].remove(unit)

            new_units_map[new_pos_hash].append(unit)

            if unit.unit_type == "HEAVY":
                heavy_entered_pos[new_pos_hash].append(unit)
            else:
                light_entered_pos[new_pos_hash].append(unit)
    
    # Only stationary units are left in units_map
    # add in all the stationary units
    for pos_hash, units in units_map.items():
        new_units_map[pos_hash] += units

    all_stopped_units: Set[Unit] = set()
    # new_units_map_after_collision: Dict[str, List[Unit]] = defaultdict(list)
    for pos_hash, units in new_units_map.items():
        stopped_units: Set[Unit] = set()
        if len(units) <= 1:
            continue

        if len(units_map[pos_hash]) > 0:
            # There is a stationary unit, avoid.
            surviving_unit = units_map[pos_hash][0]
            for u in units:
                if u.unit_id != surviving_unit.unit_id:
                    stopped_units.add(u)
        elif len(heavy_entered_pos[pos_hash]) > 1:
            # more than two heavy collide while moving, less powerful unit yields.
            most_power_unit = units[0]
            for u in units:
                if u.unit_type == "HEAVY":
                    if u.power > most_power_unit.power:
                        most_power_unit = u
            surviving_unit = most_power_unit
            for u in units:
                if u.unit_id != surviving_unit.unit_id:
                    stopped_units.add(u)
        elif len(heavy_entered_pos[pos_hash]) > 0:
            # one heavy and other light collide while moving, light yields.
            surviving_unit = heavy_entered_pos[pos_hash][0]
            for u in units:
                if u.unit_id != surviving_unit.unit_id:
                    stopped_units.add(u)
            # new_units_map_after_collision[pos_hash].append(surviving_unit)

        all_stopped_units.update(stopped_units)

    for u in all_stopped_units:
        actions[u.unit_id] = [u.move(0)]
        
    return actions
