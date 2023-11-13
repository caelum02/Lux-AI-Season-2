import jax
import jax.numpy as jnp
from jax import Array

from jux.state import State

from constants import MAP_SIZE

def naive_bid_agent(state: State, _rng: Array):
    return jnp.zeros((2,)), jnp.zeros((2,))

def random_factory_agent_batched(env_state: State, rng: Array):
    valid_mask = env_state.board.valid_spawns_mask
    n_envs = valid_mask.shape[0]

    @jax.vmap
    def _access(mask, pos):
        return mask[pos[0], pos[1]]

    # get spawn
    def not_valid(pos_rng):
        pos, rng = pos_rng
        return ~_access(valid_mask, pos).all()
    
    def body(pos_rng):
        pos, rng = pos_rng
        rng, _rng = jax.random.split(rng)
        return (jax.random.randint(_rng, (n_envs, 2), minval = 0, maxval = MAP_SIZE), rng)
    
    rng, _rng = jax.random.split(rng)
    init_pos = jax.random.randint(_rng, (n_envs, 2), minval = 0, maxval = MAP_SIZE)
    pos, _ = jax.lax.while_loop(not_valid, body, (init_pos, _rng))

    spawn = jnp.empty((n_envs, 2, 2), dtype=jnp.int8)
    spawn = spawn.at[:, env_state.next_player[0], :].set(pos)

    default_resource = jnp.repeat(jnp.array([[150, 150]], dtype=jnp.int8), repeats=n_envs, axis=0)

    return spawn, default_resource, default_resource
