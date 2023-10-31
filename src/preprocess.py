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
LIGHT_BATTERY_CAPACITY = 150
HEAVY_BATTERY_CAPACITY = 3000
LIGHT_CARGO_SPACE = 100
HEAVY_CARGO_SPACE = 1000

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

@partial(vmap, in_axes=0, out_axes=0) # team axis
@partial(vmap, in_axes=(None, -1), out_axes=-1) # flax follows channels-last convention
def to_board_for(pos, unit_info):
    map = jnp.zeros((MAP_SIZE, MAP_SIZE))
    def _to_board_i(i, map):
        loc = pos.pos[i]
        return map.at[loc[0], loc[1]].set(unit_info[i], mode='drop')
    jax.lax.fori_loop(0, pos.pos.shape[0], _to_board_i, map)
    return map

@jit
def get_unit_feature(unit_mask, unit_type, cargo, power, pos):
    '''
        unit_mask : ShapedArray(bool[2, MAX_N_UNITS])
        unit_type : ShapedArray(bool[2, MAX_N_UNITS])
        cargo: ShapedArray(int32[2, MAX_N_UNITS, 4])
        power: ShapedArray(int32[2, MAX_N_UNITS])
        pos : ShapedArray(int8[2, MAX_N_UNITS, 2])

        output: ShapedArray(int8[2, MAP_SIZE, MAP_SIZE, 12])

        feature: [light_existence, heavy_existence, (current) ice, ore, water, metal, power, (cargo empty space) ice, ore, water, metal, power]
    ''' 

    light_mask = unit_mask & (unit_type==UnitType.LIGHT)  # NOTE : is unit_mask necessary?
    heavy_mask = unit_mask & (unit_type==UnitType.HEAVY)
    unit_mask_per_type = jnp.stack((light_mask, heavy_mask), axis=-1)

    cargo_space = light_mask * LIGHT_CARGO_SPACE + heavy_mask * HEAVY_CARGO_SPACE
    batttery_capacity = light_mask * LIGHT_BATTERY_CAPACITY + heavy_mask * HEAVY_BATTERY_CAPACITY
    
    cargo_left = cargo_space[...,None] - cargo
    battery_left = batttery_capacity - power

    feature = jnp.concatenate((unit_mask_per_type, cargo, power[...,None], cargo_left, battery_left[...,None]), axis=-1)  

    unit_resource_map = to_board_for(pos, feature)

    return unit_resource_map

def main():

    MAX_N_UNITS = 200

    import matplotlib.pyplot as plt

    lux_env, lux_actions = jux.utils.load_replay('https://www.kaggleusercontent.com/episodes/52900827.json')
    jux_env, state = JuxEnv.from_lux(lux_env, buf_cfg=JuxBufferConfig(MAX_N_UNITS=MAX_N_UNITS))
    print("Done loading replay https://www.kaggleusercontent.com/episodes/52900827.json")

    state, lux_actions = replay_run_early_phase(jux_env, state, lux_actions)

    state, lux_actions = replay_run_n_late_game_step(100, jux_env, state, lux_actions)    
    unit_feature = get_unit_feature(state.unit_mask, state.units.unit_type, state.units.pos.x, state.units.pos.y)

    fig, axes = plt.subplots(2, 12, figsize=(48, 8))
    features = ['light_existence', 'heavy_existence', 'ice', 'ore', 'water', 'metal', 'power', 'ice_left', 'ore_left', 'water_left', 'metal_left', 'powerleft']
    for i in range(2):
        for j in range(12):
            axes[i, j].imshow(unit_feature[i, :, :, j], cmap='gray')
            axes[i, j].set_title(f"Player {i}, {features[j]}")
    fig.suptitle("Unit Features")
    plt.show()

    plt.imshow(jux_env.render(state, 'rgb_array'))
    plt.show()

if __name__=="__main__":
    main()
