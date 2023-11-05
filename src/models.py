import flax.linen as nn
import jax.numpy as jnp
import jax

class ActorCritic(nn.Module):
    max_n_units: int
    
    @nn.compact
    def __call__(self, x):
        pass

