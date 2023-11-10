from typing import NamedTuple

from jax import Array

from jux.actions import JuxAction


class ObsSpace(NamedTuple): # Input to the model
    local_feature: Array
    global_feature: Array

    def to_whole_feature(self):
        return jnp.concatenate([
            jnp.broadcast_to(self.global_feature[None, None,...], (MAP_SIZE, MAP_SIZE, self.global_feature.shape[-1])),
            self.local_feature
        ], axis=-1, dtype=jnp.float32)
        
# TODO
class ActionSpace(NamedTuple): # Output of the model
    pass

def act_space_to_jux_action(act_space: ActionSpace)->JuxAction:
    pass