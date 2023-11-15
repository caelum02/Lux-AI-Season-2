import jux
from jux.env import JuxEnv
from jux.config import JuxBufferConfig
from jux.unit import UnitType 
from jux.state import State

import jax
import jax.numpy as jnp
from jax import tree_map, vmap, jit
from jax import Array

from functools import partial
from typing import NamedTuple
from space import ObsSpace

from utils import StateSkeleton, flip_states, replay_run_early_phase, replay_run_n_late_game_step, get_water_info, UnitCargo
from constants import *


# First vectorize on the feature axis, and then on the team axis
@partial(vmap, in_axes=0, out_axes=0) # team axis
@partial(vmap, in_axes=(None, -1), out_axes=-1) # flax follows channels-last convention
def to_board(pos, unit_info):
    '''
    n_info: number of features to embed in the board (vectorized axis)

    unit_info: ShapedArray(int8[2, MAX_N_UNITS, n_info])
    pos: ShapedArray(int8[2, MAX_N_UNITS, 2])
   
    out: ShapedArray(int8[2, MAP_SIZE, MAP_SIZE, n_info])
    '''

    zeros = jnp.zeros((MAP_SIZE, MAP_SIZE))

    # `mode=drop` prevents unexpected index-out-of-bound behavior
    out = zeros.at[pos.x, pos.y].set(unit_info, mode='drop')
    

    return out


@partial(vmap, in_axes=0, out_axes=0) # batch axis
@partial(vmap, in_axes=0, out_axes=-2) # team axis
def to_board_for(pos: Array, unit_info: Array):
    map = jnp.zeros((MAP_SIZE, MAP_SIZE, unit_info.shape[-1]))
    def _to_board_i(i, map):
        loc = pos[i]
        return map.at[loc[0], loc[1]].set(unit_info[i], mode='drop')
    map = jax.lax.fori_loop(0, pos.shape[0], _to_board_i, map)
    return map

def get_unit_feature(states: State)->Array:
    '''
        state: State
        output: ShapedArray(int8[MAP_SIZE, MAP_SIZE, 24])

        feature: [light_existence, heavy_existence, (current) ice, ore, water, metal, power, (cargo empty space) ice, ore, water, metal, power]
    ''' 

    unit_mask, unit_type, cargo, power, pos = states.unit_mask, states.units.unit_type, states.units.cargo.stock, states.units.power, states.units.pos

    light_mask = unit_mask & (unit_type==UnitType.LIGHT)  # NOTE : is unit_mask necessary?
    heavy_mask = unit_mask & (unit_type==UnitType.HEAVY)
    unit_mask_per_type = jnp.stack((light_mask, heavy_mask), axis=-1)

    cargo_space = light_mask * LIGHT_CARGO_SPACE + heavy_mask * HEAVY_CARGO_SPACE
    batttery_capacity = light_mask * LIGHT_BATTERY_CAPACITY + heavy_mask * HEAVY_BATTERY_CAPACITY
    
    cargo_left = cargo_space[...,None] - cargo
    battery_left = batttery_capacity - power

    feature = jnp.concatenate((unit_mask_per_type, cargo, power[...,None], cargo_left, battery_left[...,None]), axis=-1)  
    unit_feature_map = to_board_for(pos.pos, feature)

    return unit_feature_map

def get_factory_feature(states: State)->Array:
    """
        state: State
        output: ShapedArray(int8[MAP_SIZE, MAP_SIZE, 16])
    """
    factory_mask, cargo, power, pos = states.factory_mask, states.factories.cargo.stock, states.factories.power, states.factories.pos.pos
    _, grow_lichen_size, connected_lichen_size = vmap(get_water_info, in_axes=(StateSkeleton,))(states)
    water_cost = jnp.ceil(grow_lichen_size / LICHEN_WATERING_COST_FACTOR).astype(UnitCargo.dtype())
    delta_power = FACTORY_CHARGE + connected_lichen_size * POWER_PER_CONNECTED_LICHEN_TILE
    feature = jnp.concatenate((factory_mask[..., None], cargo, power[...,None], water_cost[..., None], delta_power[..., None]), axis=-1)

    factory_feature_map = to_board_for(pos, feature)

    return factory_feature_map

def get_board_feature(states: State) -> Array:
    """
        state: State
        output: ShapedArray(int8[MAP_SIZE, MAP_SIZE, 4])
    """
    board_feature_map = jnp.stack([states.board.lichen, states.board.map.rubble, states.board.map.ice, states.board.map.ore], axis=-1)
    return board_feature_map

def get_global_feature(states: State) -> Array:
    real_env_steps = states.real_env_steps
    cycle, turn_in_cycle = jnp.divmod(real_env_steps, CYCLE_LENGTH)
    is_day = (real_env_steps % CYCLE_LENGTH) < DAY_LENGTH
    def lichen_score(state: State):
        return state.team_lichen_score()
    return jnp.concatenate([
        jax.nn.one_hot(cycle, TOTAL_CYCLES),
        jax.nn.one_hot(turn_in_cycle, CYCLE_LENGTH),
        is_day[..., None],
        vmap(lichen_score, in_axes=(StateSkeleton,))(states),
    ], axis=-1)

def get_feature(states: State) -> ObsSpace:
    """
        state: State
        output: ShapedArray(int8[MAP_SIZE, MAP_SIZE, C])
    """
    unit_feature_map = get_unit_feature(states)
    unit_feature_map = unit_feature_map.reshape((unit_feature_map.shape[0], MAP_SIZE, MAP_SIZE, -1))
    factory_feature_map = get_factory_feature(states)
    factory_feature_map = factory_feature_map.reshape((factory_feature_map.shape[0], MAP_SIZE, MAP_SIZE, -1))
    board_feature_map = get_board_feature(states)
    global_feature = get_global_feature(states)

    local_feature = jnp.concatenate([
        unit_feature_map,
        factory_feature_map,
        board_feature_map,
        ], axis=-1, dtype=jnp.float32)
    return ObsSpace(local_feature, global_feature, states)

def get_feature_teams(states: State)-> tuple[ObsSpace, ObsSpace]:
    unit_feature_map = get_unit_feature(states)
    factory_feature_map = get_factory_feature(states)
    board_feature_map = get_board_feature(states)
    global_feature = get_global_feature(states)

    local_feature = jnp.concatenate([
        unit_feature_map.reshape((unit_feature_map.shape[0], MAP_SIZE, MAP_SIZE, -1)),
        factory_feature_map.reshape((factory_feature_map.shape[0], MAP_SIZE, MAP_SIZE, -1)),
        board_feature_map,
        ], axis=-1, dtype=jnp.float32)
    
    global_feature2 = global_feature.at[:, [-1, -2]].set(global_feature[:, [-2, -1]])
    local_feature2 = jnp.concatenate([
        jnp.flip(unit_feature_map, -2).reshape((unit_feature_map.shape[0], MAP_SIZE, MAP_SIZE, -1)),
        jnp.flip(factory_feature_map, -2).reshape((factory_feature_map.shape[0], MAP_SIZE, MAP_SIZE, -1)),
        board_feature_map,
        ], axis=-1, dtype=jnp.float32)
    return ObsSpace(local_feature, global_feature, states), ObsSpace(local_feature2, global_feature2, flip_states(states))


def main_replay():

    MAX_N_UNITS = 200

    import matplotlib.pyplot as plt

    lux_env, lux_actions = jux.utils.load_replay('https://www.kaggleusercontent.com/episodes/52900827.json')
    jux_env, state = JuxEnv.from_lux(lux_env, buf_cfg=JuxBufferConfig(MAX_N_UNITS=MAX_N_UNITS))
    print("Done loading replay https://www.kaggleusercontent.com/episodes/52900827.json")

    state, lux_actions = replay_run_early_phase(jux_env, state, lux_actions)

    state, lux_actions = replay_run_n_late_game_step(100, jux_env, state, lux_actions)    
    local_feature, global_feature = get_feature(state)

    print(local_feature.shape, global_feature.shape)

def main():
    from utils import get_seeds
    from jax import random
    from jux.env import JuxEnv, JuxEnvBatch
    from jux.config import JuxBufferConfig, EnvConfig

    from agent import naive_bid_agent, random_factory_agent_batched

    N_ENV = 4

    batch_env = JuxEnvBatch(EnvConfig(), JuxBufferConfig(MAX_N_UNITS=200))
    key = random.PRNGKey(0)
    key, subkey = random.split(key)
    seeds = get_seeds(subkey, (N_ENV,))
    state = batch_env.reset(seeds)

    # handle bidaction
    rng, _rng = jax.random.split(key)
    _rng = jax.random.split(_rng, num=N_ENV)
    bid, faction = jax.vmap(naive_bid_agent)(state, _rng)
    state, _ = batch_env.step_bid(state, bid, faction)

    print('bidaction done')

     # Factory placement step
    n_factories = state.board.factories_per_team[0].astype(jnp.int32)

    def _factory_placement_step(i, env_state_rng):
        env_state, rng = env_state_rng
        rng, _rng = jax.random.split(rng)
        factory_placement = random_factory_agent_batched(env_state, _rng)
        env_state, _ = batch_env.step_factory_placement(env_state, *factory_placement)
        return env_state, rng

    rng, _rng = jax.random.split(rng)
    state, _ = jax.lax.fori_loop(0, 2*n_factories, _factory_placement_step, (state, _rng))

    local_feature, global_feature = get_feature(state)
    print(local_feature.shape)
    print(global_feature.shape)


if __name__=="__main__":
    main()
