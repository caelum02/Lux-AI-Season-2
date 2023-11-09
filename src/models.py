import flax.linen as nn
import jax.numpy as jnp
import jax

from jux.actions import JuxAction
from jux.config import EnvConfig, JuxBufferConfig

from space import ObsSpace, ActionSpace


class NaiveActorCritic(nn.Module):
    env_config: EnvConfig
    buf_config: JuxBufferConfig
    
    def setup(self):
        self.empty_action = JuxAction.empty(env_cfg=self.env_config, buf_cfg=self.buf_config)

    @nn.compact
    def __call__(self, x: ObsSpace):
        x = x.board_like_feature

        x = nn.Conv(features=128, kernel_size=(1,1))(x)
        x = nn.swish(x)

        x_ = nn.Conv(features=128, kernel_size=(3,3))(x)
        x_ = nn.swish(x_)
        x = nn.Conv(features=128, kernel_size=(3,3))(x_) + x

        x_ = nn.Conv(features=128, kernel_size=(3,3))(x)
        x_ = nn.swish(x_)
        x = nn.Conv(features=128, kernel_size=(3,3))(x_) + x
        
        # flatten
        x = x.reshape((x.shape[0], -1))
        value = nn.Dense(1)(x)

        return self.empty_action, value

        


