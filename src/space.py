from typing import NamedTuple

from jax import Array

from jux.actions import JuxAction


class ObsSpace(NamedTuple): # Input to the model
    local_feature: Array
    global_feature: Array
        
# TODO
class ActionSpace(NamedTuple): # Output of the model
    pass

def act_space_to_jux_action(act_space: ActionSpace)->JuxAction:
    pass