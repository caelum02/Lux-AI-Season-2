from typing import Tuple
from jax import Array

import flax.linen as nn
import jax.numpy as jnp
import jax
import distrax

from jux.actions import JuxAction
from jux.config import EnvConfig, JuxBufferConfig

from space import ObsSpace, ActionSpace

GRUCarry = Tuple[Array, Array]

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



class DecoderGruCell(nn.RNNCellBase):
    n_action_type: int = 6
    n_direction: int = 5
    n_resource_type: int = 4

    features: int = 256  # Size of hidden state

    def setup(self):
        # The number of logits for each action type, direction, and resource type
        # +1 for EOS
        self.n_logits = self.n_action_type + self.n_direction + self.n_resource_type + 1

    @nn.compact
    def __call__(
        self, carry: Tuple[GRUCarry, Array, Array, Array], _
    ):
        gru_state, last_prediction = carry
        gru_state, y = nn.GRUCell(features=self.features)(gru_state, last_prediction)

        y = nn.Dense(self.n_logits)(y)

        categorical_rng = self.make_rng('gru')
        act_type_rng, direction_rng, resource_rng, eos_rng = jax.random.split(categorical_rng, 4)

        action_type_logits = y[:, :self.n_action_type]
        action_type_arg = jax.random.categorical(act_type_rng, action_type_logits)
        action_type = jax.nn.one_hot(
            action_type_arg, num_classes=self.n_action_type, dtype=jnp.float32
        )
        action_log_probs = jax.nn.log_softmax(y)[jnp.arange(y.shape[0]) , action_type_arg]

        direction_logits = y[:, self.n_action_type:self.n_action_type + self.n_direction]
        direction_arg = jax.random.categorical(direction_rng, direction_logits)
        direction = jax.nn.one_hot(
            direction_arg, num_classes=self.n_direction, dtype=jnp.float32
        )
        direction_log_probs = jax.nn.log_softmax(y)[jnp.arange(y.shape[0]) , direction_arg]

        resource_logits = y[:, self.n_action_type + self.n_direction:]
        resource_arg = jax.random.categorical(resource_rng, resource_logits)
        resource_type = jax.nn.one_hot(
            resource_arg, num_classes=self.n_resource_type, dtype=jnp.float32
        ) 
        resource_log_probs = jax.nn.log_softmax(y)[jnp.arange(y.shape[0]) , resource_arg]

        eos_p = jax.nn.sigmoid(y[:, -1])
        eos = jax.random.bernoulli(eos_rng, eos_p)
        eos_log_probs = jnp.log(y[:, -1])
      
        prediction = jnp.concatenate([action_type, direction, resource_type, eos], axis=-1)
        log_probs = jnp.concatenate([action_log_probs, direction_log_probs, resource_log_probs, eos_log_probs], axis=-1)

        return (gru_state, prediction), (prediction, log_probs)

class ActionDecoder(nn.Module):
    hidden_size: int = 256

    @nn.compact
    def __call__(self, state_embedding: Array, unit_info: Array):
      # Concatenate the state embedding and unit information
        x = jnp.concatenate([state_embedding, unit_info], axis=-1)
        x = nn.Dense(features=self.hidden_size)(x)

        decoder = nn.RNN(
            DecoderGruCell(
                features=self.hidden_size,
            ),
            split_rngs={'gru': True},
        )

        predictions, log_probs = decoder(
            None,
            initial_carry=x
        )

        return predictions, log_probs

if __name__=="__main__":
    n_batch = 10
    state_embedding = jnp.zeros((n_batch, 256))
    unit_info = jnp.zeros((n_batch, 256))

    action_decoder = ActionDecoder()
    params = action_decoder.init(jax.random.PRNGKey(0), state_embedding, unit_info)
    logits = action_decoder.apply(params, state_embedding, unit_info)