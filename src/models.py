import functools
from typing import Any, Tuple
from jax import Array

import flax.linen as nn
import jax.numpy as jnp
import jax
import distrax

from jux.actions import JuxAction
from jux.config import EnvConfig, JuxBufferConfig

from space import ObsSpace, ActionSpace

GRUCarry = Tuple[Array, Array]


class CBAM(nn.Module):
    """
    CBAM: Convolutional Block Attention Module
    https://arxiv.org/abs/1807.06521
    """
    r: int = 16
        
    @nn.compact
    def __call__(self, x: Array) -> Array:
        channels = x.shape[-1]
        reduced_channels = channels // self.r
        c = jnp.concatenate([
            jnp.average(x, axis=(-3, -2)),
            jnp.max(x, axis=(-3, -2)),
            ], axis=-1)
        c = nn.Dense(features=reduced_channels)(c)
        c = nn.swish(c)
        c = nn.Dense(features=channels)(c)
        c = nn.sigmoid(c)
        x = x * jnp.expand_dims(c, (-3, -2))
        s = jnp.concatenate([
            jnp.average(x, axis=-1, keepdims=True),
            jnp.max(x, axis=-1, keepdims=True),
            ], axis=-1)
        s = nn.Conv(features=1, kernel_size=(7,7))(s)
        s = nn.sigmoid(s)
        x = x * s
        return x


class CBAMBackBone(nn.Module):
    
    @nn.compact
    def __call__(self, x: ObsSpace) -> Array:
        xl = x.local_feature
        xg = x.global_feature
        xg = nn.Dense(features=16)(xg)
        xg = nn.swish(xg)
        x = jnp.concatenate([
            jnp.broadcast_to(jnp.expand_dims(xg, (1, 2)), xl.shape[:-1] + xg.shape[-1:]),
            xl,
        ], axis=-1)
        x = nn.Conv(features=64, kernel_size=(1,1))(x)
        x = nn.swish(x)
        x = CBAM()(x)

        x_ = nn.Conv(features=64, kernel_size=(3,3))(x)
        x_ = nn.swish(x_)
        x_ = CBAM()(x_)
        x = x_ + x

        x_ = nn.Conv(features=64, kernel_size=(3,3))(x)
        x_ = nn.swish(x_)
        x_ = CBAM()(x_)
        x = x_ + x

        x_ = nn.Conv(features=64, kernel_size=(3,3))(x)
        x_ = nn.swish(x_)
        x_ = CBAM()(x_)
        x = x_ + x

        return x


class PoolCriticHead(nn.Module):
    """
    Critic Head
    1. Take the output of the backbone (CNN), and apply global pooling (avg, max)
    2. Concatenate the pooled features
    3. Linearly transform the concatenated features to a scalar (value)
    """
    @nn.compact
    def __call__(self, x: Array) -> Array:
        c = jnp.concatenate([
            jnp.average(x, axis=(-3, -2)),
            jnp.max(x, axis=(-3, -2)),
            ], axis=-1).reshape((x.shape[0], -1))
        value = nn.Dense(1)(c)
        
        return value

class NaiveActorCritic(nn.Module):
    env_config: EnvConfig
    buf_config: JuxBufferConfig
    
    def setup(self):
        self.empty_action = JuxAction.empty(env_cfg=self.env_config, buf_cfg=self.buf_config)

    @nn.compact
    def __call__(self, x: ObsSpace) -> tuple[JuxAction, Array]:
        
        Critic = nn.Sequential((
            CBAMBackBone(name='critic_backbone'),
            PoolCriticHead(name='critic_head'),
        ))

        value = Critic(x)
        
        return self.empty_action, value

class ActorCritic(nn.Module):
    env_config: EnvConfig
    buf_config: JuxBufferConfig

    @nn.compact
    def __call__(self, x: ObsSpace) -> tuple[JuxAction, Array]:
        raise NotImplementedError
    

class DecoderGruCell(nn.Module):
    n_action_type: int = 6
    n_direction: int = 5
    n_resource_type: int = 4

    features: int = 256  # Size of hidden state

    def setup(self):
        # The number of logits for each action type, direction, and resource type
        # +1 for EOS
        self.n_logits = self.n_action_type + self.n_direction + self.n_resource_type + 1
        self.action_type_slice = slice(0, self.n_action_type)
        self.direction_slice = slice(self.n_action_type, self.n_action_type + self.n_direction)
        self.resource_type_slice = slice(self.n_action_type + self.n_direction, self.n_action_type + self.n_direction + self.n_resource_type)

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
        action_type_logits = y[..., self.action_type_slice]
        action_type_dist = distrax.Categorical(logits=action_type_logits)
        action_type_args = action_type_dist.sample(seed=act_type_rng)
        action_type = jax.nn.one_hot(
            action_type_args, num_classes=self.n_action_type, dtype=jnp.float32
        )
        action_log_probs = action_type_dist.log_prob(action_type_args)

        direction_logits = y[..., self.direction_slice]
        direction_dist = distrax.Categorical(direction_logits)
        direction_arg = direction_dist.sample(seed=direction_rng)
        direction = jax.nn.one_hot(
            direction_arg, num_classes=self.n_direction, dtype=jnp.float32
        )
        direction_log_probs = direction_dist.log_prob(direction_arg)

        resource_type_logits = y[..., self.resource_type_slice]
        resource_type_dist = distrax.Categorical(resource_type_logits)
        resource_type_arg = resource_type_dist.sample(seed=resource_rng)
        resource_type = jax.nn.one_hot(
            resource_type_arg, num_classes=self.n_resource_type, dtype=jnp.float32
        )
        resource_type_log_probs = resource_type_dist.log_prob(resource_type_arg)

        eos_p = jax.nn.sigmoid(y[..., -1])
        eos = jax.random.bernoulli(eos_rng, eos_p).astype(jnp.float32)
        eos_log_probs = jnp.log(eos_p * eos + (1 - eos_p) * (1 - eos))
      
        # this concatenation is essential because this one-hot encoded vectors
        # should be fed into the next GRU cell
        prediction = jnp.concatenate([action_type, direction, resource_type, eos[..., None]], axis=-1)

        log_probs = (action_log_probs, direction_log_probs, resource_type_log_probs, eos_log_probs)

        return (gru_state, prediction), (prediction, log_probs)

class GruActionHead(nn.Module):
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
        """
            axis 0: batch
            axis 1: unit
            axis 2: time
            axis 3: feature
        """

        # Concatenate the state embedding and unit information
        n_unit = unit_info.shape[1]

        state_embedding_broadcasted = jnp.broadcast_to(
            jnp.expand_dims(state_embedding, axis=1), 
            shape=(*unit_info.shape[:2], state_embedding.shape[-1])
        )

        x = jnp.concatenate((
            state_embedding_broadcasted,
            unit_info), 
            axis=-1
        )
        init_carry = nn.Dense(features=self.hidden_size)(x)

        DecoderGru = nn.scan(
            DecoderGruCell, variable_broadcast='params', 
            in_axes=2, out_axes=2, # support unit batch dimensions
            split_rngs={'params': False, 'gru': True},
            length=self.length,
        )

        decoder = DecoderGru(
            features=self.hidden_size,
            n_action_type=self.n_action_type,
            n_direction=self.n_direction,
            n_resource_type=self.n_resource_type,
        )
        
        # Feed First GRU input with zeros, hidden state from state embedding
        input_shape = (*x.shape[:-1], self.length, self.n_logits)
        dummy_input = jnp.empty(shape=input_shape)
        init_carry = (x, dummy_input[..., 0, :])
        predictions, log_probs = decoder(init_carry, dummy_input)

        return predictions, log_probs

def main():
    seed = 42
    n_batch = 10
    n_unit = 48
    queue_length = 5

    key = jax.random.PRNGKey(seed=seed)
    key, subkey1, subkey2 = jax.random.split(key, num=3)
    state_embedding = jax.random.uniform(subkey1, (n_batch, 256))
    unit_info = jax.random.uniform(subkey2, (n_batch, n_unit, 256))
    
    key, params_key, sample_key = jax.random.split(key=key, num=3)

    action_decoder = GruActionHead(hidden_size=256, length=queue_length)
    params = action_decoder.init(
        {'params': params_key, 'gru': sample_key},
        state_embedding, unit_info
    )

    _, (predictions, log_probs) = action_decoder.apply(params, state_embedding, unit_info, rngs={'gru': sample_key})

    print(predictions.shape)
    print(log_probs[0].shape)

    key, subkey1, subkey2 = jax.random.split(key, num=3)
    state_embedding = jax.random.normal(subkey1, (n_batch, 1, 256))
    unit_info = jax.random.normal(subkey2, (n_batch, 256))
    
    key, params_key = jax.random.split(key)
    key, local_key, global_key = jax.random.split(key, 3)
    actor_critic = NaiveActorCritic(env_config=EnvConfig(), buf_config=JuxBufferConfig(MAX_N_UNITS=500))
    features = ObsSpace(
        local_feature=jax.random.normal(local_key, (n_batch, 64, 64, 44)),
        global_feature=jax.random.normal(global_key, (n_batch, 73)),
    )
    params = actor_critic.init({'params': params_key}, features)
    action, value = actor_critic.apply(params, features)
    print("value")
    print(value)

if __name__=="__main__":
    main()
