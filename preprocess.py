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

@vmap
def to_board(x, y, unit_info):
    '''
    n_info: number of features to embed in the board (vectorized axis)

    unit_info: ShapedArray(int8[n_info, MAX_N_UNITS])
    x: ShapedArray(int8[n_info, MAX_N_UNITS])
    y: ShapedArray(int8[n_info, MAX_N_UNITS])
   
    out: ShapedArray(int8[n_info, MAP_SIZE, MAP_SIZE])
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

        output: ShapedArray(int8[4, MAX_N_UNITS])

        light player 0, light player 1, heavy player 0, heavy player 1


                unit type goes to axis 0 to preserve locality of player & unit_id axis
    '''

    light_mask = unit_mask & (unit_type==UnitType.LIGHT)
    heavy_mask = unit_mask & (unit_type==UnitType.HEAVY)

    light_unit_map = to_board(x, y, light_mask)
    heavy_unit_map = to_board(x, y, heavy_mask)

    unit_map = jnp.concatenate((light_unit_map, heavy_unit_map))
    
    return unit_map

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