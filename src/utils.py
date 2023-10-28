from jax import jit
from jax.tree_util import tree_map
import jax.numpy as jnp

from luxai_s2.env import LuxAI_S2
from luxai_s2.actions import format_action_vec

import jux
from jux.env import JuxEnv
from jux.config import JuxBufferConfig
from jux.state import State
from jux.actions import JuxAction, ActionQueue
from jux.unit import Unit

def get_unit_idx(state: State, id: int):
    """
    Return unit_idx of unit with id `id`
    """
    unit_idx = state.unit_id2idx[id]
    return unit_idx[...,0], unit_idx[...,1]

def get_unit_pos(state: State, id: int):
    """
    Return unit position of unit with id `id`

    return: Array (2,)
    """
    return state.units.pos.pos[get_unit_idx(state, id)]

def get_action_queue_from_id(state: State, id: int)->ActionQueue:
    """
    Return action queue of unit with id `id`
    """
    idx = get_unit_idx(state, id)
    from_jux = tree_map(lambda x: x[idx], state.units.action_queue).to_lux()
    return from_jux

def get_unit_from_id(state: State, id: id)->Unit:
    """
    Return unit with id `id`

    """
    idx = get_unit_idx(state, id)
    return tree_map(lambda x: x[idx], state.units)

def replay_run_early_phase(jux_env: JuxEnv, state: State, lux_actions, lux_env=None):
    """
        Util function | skip game until late_game stage
        
        return:
            state: game state right after factory placement phase
            lux_actions: lux action iterator synchronized to `state`
    """    

    print(f"[Replay Util] Replaying early steps")

    with_lux_env = lux_env is not None

    # Bid Step
    lux_action = next(lux_actions)
    bid, faction = jux.actions.bid_action_from_lux(lux_action)
    state, (obs, rwd, dones, infos) = jux_env.step_bid(state, bid, faction)
    if with_lux_env:
        lux_env.step(lux_action)

    # Factory Placement Step
    while state.real_env_steps < 0:
        lux_action = next(lux_actions)
        spawn, water, metal = jux.actions.factory_placement_action_from_lux(lux_action)
        state, _ = jux_env.step_factory_placement(state, spawn, water, metal)
        if with_lux_env:
            lux_env.step(lux_action)
    
    print(f"[Replay Util] Replaying early steps - Done")

    if with_lux_env:
        return state, lux_actions, lux_env
    return state, lux_actions

def replay_run_n_late_game_step(n: int, jux_env: JuxEnv, state: State, lux_actions, lux_env=None):
    
    with_lux_env = lux_env is not None

    for i in range(n):
        print(f"[Replay Util] Replaying {i}/{n} steps")
        lux_act = next(lux_actions)
        jux_act = JuxAction.from_lux(state, lux_act)
        # assert jux_act.to_lux(state) == lux_act, f"JuxAction.to_lux() is not reversible: {jux_act.to_lux(state)} != {lux_act}"

        # step
        state, (obs, rwd, dones, infos) = jux_env.step_late_game(state, jux_act)

        # assert action_queue_validity(state).all()

        if with_lux_env:
            obs = lux_env.step(lux_act)[0]
            obs = obs['player_0']

            # if i >= 8: # unit 88 action_queue mismatch
            #     run_check_action_queue(obs, state)
            # if i % 20 == 10:
            #     run_check_pos(obs, state)
            #     run_check_action_queue(obs, state)
        
        if dones[0]:
            print(f"[Replay Util] Replaying {i+1}/{n} steps - Done")
            break
    
    if with_lux_env:
        return state, lux_actions, lux_env
    return state, lux_actions

def action_queue_valid(state: State):
    """
    Check if action queue is valid
    """
    return (~state.unit_mask | (state.units.action_queue.count == 0) | \
                 ((0 <= state.units.action_queue.front) & (state.units.action_queue.front < 20) & 
                   (0 <= state.units.action_queue.rear) & (state.units.action_queue.rear < 20))).all()

def unit_pos_cross_valid(obs, state):
    """
    Check if unit position is consistent between jux and lux
    obs: lux observation
    state: jux state
    """
    for player_id, player_units in obs['units'].items():
        for unit_id, unit in player_units.items():
            assert unit_id.startswith('unit_')
            int_id = int(unit_id.split('_')[1])
            pos = get_unit_pos(state, int_id)
            if not (pos == unit['pos']).all():
                return False
    return True



def action_queue_cross_valid(obs, state):
    """
    Check if action queue is consistent between jux and lux
    obs: lux observation
    state: jux state
    """
    for player_id, player_units in obs['units'].items():
        for unit_id, unit in player_units.items():
            int_id = int(unit_id.split('_')[1])
            from_jux = get_action_queue_from_id(state, int_id)
            if len(unit['action_queue']) == 0:
                continue
            from_lux = jnp.stack(unit['action_queue'])
            if not (from_lux == from_jux).all():
                return False
    return True

def print_action_queue(array):
    """
    Print action queue in a human readable format
    array : (n, 6)    
    """
    for i, a in enumerate(array):
        print(f"{i}: {format_action_vec(a)}")
