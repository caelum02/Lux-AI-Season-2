import jux
from jux.env import JuxEnv
from jux.config import JuxBufferConfig
from jux.unit import UnitType 

import jax
import jax.numpy as jnp
from jax import vmap, jit

from functools import partial

from utils import replay_run_early_phase, replay_run_n_late_game_step

MAP_SIZE = 64

# First vectorize on the feature axis, and then on the team axis
@partial(vmap, in_axes=0, out_axes=0) # team axis
@partial(vmap, in_axes=(None, None, -1), out_axes=-1) # flax follows channels-last convention
def to_board(x, y, unit_info):
    '''
    n_info: number of features to embed in the board (vectorized axis)

    unit_info: ShapedArray(int8[2, MAX_N_UNITS, n_info])
    x: ShapedArray(int8[2, MAX_N_UNITS])
    y: ShapedArray(int8[2, MAX_N_UNITS])
   
    out: ShapedArray(int8[2, MAP_SIZE, MAP_SIZE, n_info])
    '''

    zeros = jnp.zeros((MAP_SIZE, MAP_SIZE))

    # `mode=drop` prevents unexpected index-out-of-bound behavior
    out = zeros.at[x, y].add(unit_info, mode='drop')

    return out

@jit
def get_unit_existence(unit_mask, unit_type, x, y):
    '''
        unit_type : ShapedArray(int8[2, MAX_N_UNITS])
        unit_mask : ShapedArray(bool[2, MAX_N_UNITS])
        x : ShapedArray(int8[2, MAX_N_UNITS])
        y : ShapedArray(int8[2, MAX_N_UNITS])

        output: ShapedArray(int8[2, MAP_SIZE, MAP_SIZE, 2]) 

        feature: [light, heavy]   
    '''

    light_mask = unit_mask & (unit_type==UnitType.LIGHT)
    heavy_mask = unit_mask & (unit_type==UnitType.HEAVY)

    mask = jnp.stack((light_mask, heavy_mask), axis=-1)
    unit_map = to_board(x, y, mask)  
    
    return unit_map

@jit
def get_unit_resource(cargo, power, x, y,):
    '''
        unit_mask : ShapedArray(bool[2, MAX_N_UNITS])
        cargo: ShapedArray(int32[2, MAX_N_UNITS, 4])
        power: ShapedArray(int32[2, MAX_N_UNITS])
        x : ShapedArray(int8[2, MAX_N_UNITS])
        y : ShapedArray(int8[2, MAX_N_UNITS])

        output: ShapedArray(int8[2, MAP_SIZE, MAP_SIZE, 5])

        feature: [ice, ore, water, metal, power]
    ''' 

    resource = jnp.concatenate((cargo, power[...,None]), axis=-1)  

    unit_resource_map = to_board(x, y, resource)

    return unit_resource_map


if __name__=="__main__":

    MAX_N_UNITS = 200

    import matplotlib.pyplot as plt

    lux_env, lux_actions = jux.utils.load_replay('https://www.kaggleusercontent.com/episodes/52900827.json')
    jux_env, state = JuxEnv.from_lux(lux_env, buf_cfg=JuxBufferConfig(MAX_N_UNITS=MAX_N_UNITS))
    print("Done loading replay https://www.kaggleusercontent.com/episodes/52900827.json")

    state, lux_actions = replay_run_early_phase(jux_env, state, lux_actions)

    state, lux_actions = replay_run_n_late_game_step(100, jux_env, state, lux_actions)    
    unit_map = get_unit_existence(state.unit_mask, state.units.unit_type, state.units.pos.x, state.units.pos.y)

    fig, axes = plt.subplots(2, 2)
    axes[0, 0].imshow(unit_map[0].T)
    axes[0, 1].imshow(unit_map[1].T)
    axes[1, 0].imshow(unit_map[2].T)
    axes[1, 1].imshow(unit_map[3].T)

    plt.show()

    plt.imshow(jux_env.render(state, 'rgb_array'))
    plt.show()