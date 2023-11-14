import functools
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
    def __call__(self, x: ObsSpace) -> tuple[JuxAction, Array]:
        xl = x.local_feature
        xg = x.global_feature
        xg = nn.Dense(features=16)(xg)
        xg = nn.swish(xg)
        x = jnp.concatenate([
            jnp.broadcast_to(jnp.expand_dims(xg, (1, 2)), xl.shape[:-1] + xg.shape[-1:]),
            xl,
        ], axis=-1)
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

class DecoderGru(nn.Module):
    n_action_type: int = 6
    n_direction: int = 5
    n_resource_type: int = 4

    features: int = 256  # Size of hidden state

    length: int = 10  # Max length of the sequence

    def setup(self):
        # The number of logits for each action type, direction, and resource type
        # +1 for EOS
        self.n_logits = self.n_action_type + self.n_direction + self.n_resource_type + 1
        self.action_type_slice = slice(0, self.n_action_type)
        self.direction_slice = slice(self.n_action_type, self.n_action_type + self.n_direction)
        self.resource_type_slice = slice(self.n_action_type + self.n_direction, self.n_action_type + self.n_direction + self.n_resource_type)

    @functools.partial(
        nn.scan,
        variable_broadcast='params',
        in_axes=1,
        out_axes=1,
        split_rngs={'params': False,'gru': True},
        length=length,
    )
    @nn.compact
    def __call__(
        self, carry: Tuple[GRUCarry, Array], _: None
    ):
        gru_state, last_prediction = carry
        gru_state, y = nn.GRUCell(features=self.features)(gru_state, last_prediction)

        y = nn.Dense(features=self.n_logits)(y)

        categorical_rng = self.make_rng('gru')
        act_type_rng, direction_rng, resource_rng, eos_rng = jax.random.split(categorical_rng, 4)

        # Now split y into each seperate logits, sample and calculate log probabilities
        action_type_logits = y[:, self.action_type_slice]
        action_type_arg = jax.random.categorical(act_type_rng, action_type_logits)
        action_type = jax.nn.one_hot(
            action_type_arg, num_classes=self.n_action_type, dtype=jnp.float32
        )
        action_log_probs = jax.nn.log_softmax(action_type_logits)[jnp.arange(y.shape[0]) , action_type_arg]

        direction_logits = y[:, self.direction_slice]
        direction_arg = jax.random.categorical(direction_rng, direction_logits)
        direction = jax.nn.one_hot(
            direction_arg, num_classes=self.n_direction, dtype=jnp.float32
        )
        direction_log_probs = jax.nn.log_softmax(direction_logits)[jnp.arange(y.shape[0]) , direction_arg]

        resource_logits = y[:, self.resource_type_slice]
        resource_arg = jax.random.categorical(resource_rng, resource_logits)
        resource_type = jax.nn.one_hot(
            resource_arg, num_classes=self.n_resource_type, dtype=jnp.float32
        ) 
        resource_log_probs = jax.nn.log_softmax(resource_logits)[jnp.arange(y.shape[0]) , resource_arg]
        eos_p = jax.nn.sigmoid(y[:, -1])
        eos = jax.random.bernoulli(eos_rng, eos_p).astype(jnp.float32)
        eos_log_probs = jnp.log(eos_p * eos + (1 - eos_p) * (1 - eos))
      
        # this concatenation is essential because this one-hot encoded vectors
        # should be fed into the next GRU cell
        prediction = jnp.concatenate([action_type, direction, resource_type, eos[:, None]], axis=-1)

        log_probs = (action_log_probs, direction_log_probs, resource_log_probs, eos_log_probs)

        return (gru_state, prediction), (prediction, log_probs)

class ActionDecoder(nn.Module):
    hidden_size: int = 256
    length: int = 10

    n_action_type: int = 6
    n_direction: int = 5
    n_resource_type: int = 4

    def setup(self):
        # The number of logits for each action type, direction, and resource type
        # +1 for EOS
        self.n_logits = self.n_action_type + self.n_direction + self.n_resource_type + 1

    @nn.compact
    def __call__(self, state_embedding: Array, unit_info: Array):
      # Concatenate the state embedding and unit information
        x = jnp.concatenate([state_embedding, unit_info], axis=-1)
        x = nn.Dense(features=self.hidden_size)(x)

        decoder = DecoderGru(
                features=self.hidden_size,
                n_action_type=self.n_action_type,
                n_direction=self.n_direction,
                n_resource_type=self.n_resource_type,
                length=self.length
        )        
        
        # Feed First GRU input with zeros, hidden state from state embedding
        init_state = (x, jnp.zeros((x.shape[0], self.n_logits)))
        predictions, log_probs = decoder(init_state, None)

        return predictions, log_probs

def main():
    seed = 42
    n_batch = 3
    queue_length = 4

    state_embedding = jnp.zeros((n_batch, 256))
    unit_info = jnp.zeros((n_batch, 256))

    key = jax.random.PRNGKey(seed=seed)
    key, params_key, sample_key = jax.random.split(key=key, num=3)

    action_decoder = ActionDecoder(hidden_size=256, length=queue_length)
    params = action_decoder.init(
        {'params': params_key, 'gru': sample_key},
        state_embedding, unit_info
    )

    _, (predictions, log_probs) = action_decoder.apply(params, state_embedding, unit_info, rngs={'gru': sample_key})
    # action_type_log_probs, direction_log_probs, resource_type_log_probs, eos_log_probs = log_probs

    print(predictions.shape)
    print(predictions[0])
    
    for probs, name in zip(log_probs, ['action_type', 'direction', 'resource_type', 'eos']):
        print(name)
        print(probs.shape)
        print("log_probs")
        print(probs)
        print("probs")
        print(jnp.exp(probs))
    
    key, params_key = jax.random.split(key)
    actor_critic = NaiveActorCritic(env_config=EnvConfig(), buf_config=JuxBufferConfig(MAX_N_UNITS=500))
    features = ObsSpace(
        local_feature=jnp.zeros((n_batch, 64, 64, 44)),
        global_feature=jnp.zeros((n_batch, 73))
    )
    params = actor_critic.init({'params': params_key}, features)
    action, value = actor_critic.apply(params, features)
    print("value")
    print(value)

if __name__=="__main__":
    main()
