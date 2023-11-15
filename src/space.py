from typing import NamedTuple

from jax import Array

from jux.actions import JuxAction

from jux.state import State


class ObsSpace(NamedTuple): # Input to the model
    local_feature: Array
    global_feature: Array
    state: State
        
# TODO
class ActionSpace(NamedTuple): # Output of the model
    pass

def act_space_to_jux_action(act_space: ActionSpace)->JuxAction:
    pass