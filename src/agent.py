import jax
import jax.numpy as jnp
from jax import Array

from jux.state import State

from constants import MAP_SIZE

def naive_bid_agent(state: State, _rng: Array):
    return jnp.zeros((2,))

def random_factory_agent(state: State, rng: Array):
    valid_mask = state.board.valid_spawns_mask

    # get spawn
    def not_valid(pos, rng):
        return 1-valid_mask[pos[0], pos[1]]
    
    def body(pos, rng):
        rng, _rng = jax.random.split(rng)
        return jax.random.randint(_rng, (MAP_SIZE, MAP_SIZE), minval = 0, maxval = MAP_SIZE), rng
    
    rng, _rng = jax.random.split(rng)
    init_pos = jax.random.randint(_rng, (MAP_SIZE, MAP_SIZE), minval = 0, maxval = MAP_SIZE)
    pos = jax.lax.while_loop(not_valid, body, (init_pos, _rng))

    spawn = jnp.empty((2, 2))
    spawn = spawn.at[state.next_player, :].set(pos)

    return pos, jnp.array((150, 150)), jnp.zeros((150, 150))