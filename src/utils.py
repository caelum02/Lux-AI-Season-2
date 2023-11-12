import jax
from jax import jit, Array
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
from jux.utils import imax
from jux.unit_cargo import UnitCargo
from jux.map.position import Position
from jux.factory import Factory

from typing import Tuple

def get_seeds(rng: Array, shape=Tuple)->Array:
    """
    Get a random seed from rng
    """
    return jax.random.randint(rng, shape=shape, minval=-2**31, maxval=2**31-1)


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
        print(f"[Replay Util] Replaying {i+1}/{n} steps")
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

@jit
def get_water_info(state: State) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Run flood fill algorithm to color cells. All cells to be watered by the
        same factory will have the same color.

        Returns:
            color: int[H, W, 2].
                The first dimension represent the 'color'.The 'color' is
                represented by the coordinate of the factory a tile belongs to.
                If a tile is not connected to any factory, its color its own
                coordinate. In such a way, different lichen strains will have
                different colors.

            grow_lichen_size: int[2, F].
                The number of positions to be watered by each factory.
        """
        # The key idea here is to prepare a list of neighbors for each cell it
        # connects to when watered. neighbor_ij is a 4x2xHxW array, where the
        # first dimension is the neighbors (4 at most), the second dimension is
        # the coordinates (x,y) of neighbors.
        H, W = state.board.lichen_strains.shape

        ij = jnp.mgrid[:H, :W].astype(Position.dtype())
        delta_ij = jnp.array([
            [-1, 0],
            [0, 1],
            [1, 0],
            [0, -1],
        ], dtype=ij.dtype)  # int[2, H, W]
        neighbor_ij = delta_ij[..., None, None] + ij[None, ...]  # int[4, 2, H, W]

        # handle map boundary.
        neighbor_ij = neighbor_ij.at[0, 0, 0, :].set(0)
        neighbor_ij = neighbor_ij.at[1, 1, :, W - 1].set(W - 1)
        neighbor_ij = neighbor_ij.at[2, 0, H - 1, :].set(H - 1)
        neighbor_ij = neighbor_ij.at[3, 1, :, 0].set(0)

        # 1. calculate strain connections.
        strains_and_factory = jnp.minimum(state.board.lichen_strains, state.board.factory_occupancy_map)  # int[H, W]

        # handle a corner case where there may be rubbles on strains when movement collision happens.
        strains_and_factory = jnp.where(state.board.rubble == 0, strains_and_factory, imax(strains_and_factory.dtype))

        neighbor_color = strains_and_factory.at[(
            neighbor_ij[:, 0],
            neighbor_ij[:, 1],
        )].get(mode='fill', fill_value=imax(strains_and_factory.dtype))

        connect_cond = (
            (strains_and_factory == neighbor_color) & (strains_and_factory != imax(strains_and_factory.dtype))
        )  # bool[4, H, W]

        color = jux.map_generator.flood._flood_fill(  # int[H, W, 2]
            jnp.concatenate(  # int[H, W, 5, 2]
                [
                    jnp.where(connect_cond[:, None], neighbor_ij, ij).transpose(2, 3, 0, 1),  # int[H, W, 4, 2]
                    ij[None].transpose(2, 3, 0, 1),  # int[H, W, 1, 2]
                ],
                axis=-2,
            ))
        factory_color = color.at[state.factories.pos.x,state.factories.pos.y] \
                             .get(mode='fill', fill_value=imax(color.dtype))  # int[2, F, 2]
        connected_lichen = jnp.full((H, W), fill_value=imax(Factory.id_dtype()))  # int[H, W]
        connected_lichen = connected_lichen.at[factory_color[..., 0], factory_color[..., 1]] \
                                           .set(state.factories.unit_id, mode='drop')
        connected_lichen = connected_lichen.at[color[..., 0], color[..., 1]]\
                                           .get(mode='fill', fill_value=imax(connected_lichen.dtype))

        # compute connected lichen size
        connected_lichen_size = jux.map_generator.flood.component_sum(UnitCargo.dtype()(1), color)  # int[H, W]
        # -9 for the factory occupied cells
        connected_lichen_size = connected_lichen_size[state.factories.pos.x, state.factories.pos.y] - 9  # int[2, F]

        # 2. handle cells to expand to.
        # 2.1 cells that are allowed to expand to, only if
        #   1. it is not a lichen strain, and
        #   2. it has no rubble, and
        #   3. it is not resource.
        allow_grow = (state.board.rubble == 0) & \
                     ~(state.board.ice | state.board.ore) & \
                     (state.board.lichen_strains == imax(state.board.lichen_strains.dtype)) & \
                     (state.board.factory_occupancy_map == imax(state.board.factory_occupancy_map.dtype))

        # 2.2 when a non-lichen cell connects two different strains, then it is not allowed to expand to.
        neighbor_lichen_strain = strains_and_factory[neighbor_ij[:, 0], neighbor_ij[:, 1]]  # int[4, H, W]
        neighbor_is_lichen = neighbor_lichen_strain != imax(neighbor_lichen_strain.dtype)
        center_connects_two_different_strains = (strains_and_factory == imax(strains_and_factory.dtype)) & ( \
            ((neighbor_lichen_strain[0] != neighbor_lichen_strain[1]) & neighbor_is_lichen[0] & neighbor_is_lichen[1]) | \
            ((neighbor_lichen_strain[0] != neighbor_lichen_strain[2]) & neighbor_is_lichen[0] & neighbor_is_lichen[2]) | \
            ((neighbor_lichen_strain[0] != neighbor_lichen_strain[3]) & neighbor_is_lichen[0] & neighbor_is_lichen[3]) | \
            ((neighbor_lichen_strain[1] != neighbor_lichen_strain[2]) & neighbor_is_lichen[1] & neighbor_is_lichen[2]) | \
            ((neighbor_lichen_strain[1] != neighbor_lichen_strain[3]) & neighbor_is_lichen[1] & neighbor_is_lichen[3]) | \
            ((neighbor_lichen_strain[2] != neighbor_lichen_strain[3]) & neighbor_is_lichen[2] & neighbor_is_lichen[3]) \
        )
        allow_grow = allow_grow & ~center_connects_two_different_strains

        # 2.3 calculate the strains id, if it is expanded to.
        expand_center = (connected_lichen != imax(connected_lichen.dtype)) & \
                        (state.board.lichen >= state.env_cfg.MIN_LICHEN_TO_SPREAD)
        factory_occupancy = state.factories.occupancy
        expand_center = expand_center.at[factory_occupancy.x, factory_occupancy.y].set(True, mode='drop')
        expand_center = jnp.where(expand_center, connected_lichen, imax(connected_lichen.dtype))
        INT_MAX = imax(expand_center.dtype)
        strain_id_if_expand = jnp.minimum(  # int[H, W]
            jnp.minimum(
                jnp.roll(expand_center, 1, axis=0).at[0, :].set(INT_MAX),
                jnp.roll(expand_center, -1, axis=0).at[-1, :].set(INT_MAX),
            ),
            jnp.minimum(
                jnp.roll(expand_center, 1, axis=1).at[:, 0].set(INT_MAX),
                jnp.roll(expand_center, -1, axis=1).at[:, -1].set(INT_MAX),
            ),
        )
        strain_id_if_expand = jnp.where(allow_grow, strain_id_if_expand, INT_MAX)

        # 3. get the final color result.
        strain_id = jnp.minimum(connected_lichen, strain_id_if_expand)  # int[H, W]
        factory_idx = state.factory_id2idx[strain_id]  # int[2, H, W]
        color = state.factories.pos.pos[factory_idx[..., 0], factory_idx[..., 1]]  # int[H, W, 2]
        color = jnp.where((strain_id == imax(strain_id.dtype))[..., None], ij.transpose(1, 2, 0), color)

        # 4. grow_lichen_size
        cmp_cnt = jux.map_generator.flood.component_sum(UnitCargo.dtype()(1), color)  # int[H, W]
        # -9 for the factory occupied cells
        grow_lichen_size = cmp_cnt[state.factories.pos.x, state.factories.pos.y] - 9  # int[2, F]

        return color, grow_lichen_size, connected_lichen_size


StateSkeleton = State(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
