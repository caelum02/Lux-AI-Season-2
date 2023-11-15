import jax
import jax.numpy as jnp
from jax import Array, tree_map
import optax
from typing import NamedTuple
from flax.training.train_state import TrainState
import distrax

from jux.env import JuxEnv, JuxEnvBatch
from jux.config import JuxBufferConfig, EnvConfig
from jux.actions import JuxAction

from preprocess import get_feature
from constants import *
from space import ObsSpace, ActionSpace
from utils import get_seeds


class PPOConfig(NamedTuple):
    LR: float = 1e-4
    MAX_GRAD_NORM: float = 0.5

    N_ENVS: int = 4
    N_UPDATES: int = 1000
    N_EPISODES_PER_ENV: int = 16
    UPDATE_EPOCHS: int = 4
    NUM_MINIBATCHES: int = 32 

    GAMMA: float = 0.99
    GAE_LAMBDA: float = 0.96
    CLIP_EPS: float = 0.2
    ENT_COEF: float = 0.01  # Entropy loss coefficient
    VF_COEF: float = 0.5  # Critic loss coefficient


class Trajectory(NamedTuple):
    done: Array
    action: Array
    value: Array
    reward: Array
    log_prob: Array
    obs: ObsSpace

class RunnerState(NamedTuple):
    train_state: TrainState
    env_state: JuxEnv
    obs: ObsSpace # extracted feature
    rng: jax.Array
    
class UpdateState(NamedTuple):
    train_state: TrainState
    rng: jax.Array

def make_train(env_config: EnvConfig, buf_config: JuxBufferConfig, ppo_config: PPOConfig, 
               actor_critic, bid_agent, factory_placement_agent, rng):
    """
        batch_env: JuxEnvBatch initialized with env_config and buf_config
        actor_critic: ActorCritic model
        bid_handler: return bid action from state 
            Note: To be vmapped
        factory_placement_handler: return factory placement action from state
            Note: To be vmaped
        obs_transform: apply preprocessing to observation
    
        # some numbers
        feature ~ 2MB
        total feature size per epsiode ~ 2GB 
        state ~ 0.5MB (max 1000 units)
                0.1MB (max 100 units)
        8GB 
        
        # psudocode
            Run total `N_STEPS_PER_UPDATE` steps
                with `N_ENVS` environments in parallel
            Caculate GAE    
            Update `UPDATE_EPOCHS` epochs
            
    """

    def train(rng):
        batch_env = JuxEnvBatch(env_config, buf_config)
        num_envs = ppo_config.N_ENVS

        # Initialize network
        rng, _rng = jax.random.split(rng)
        
        dummy_seeds = get_seeds(_rng, (1,))
        dummy_state = batch_env.reset(dummy_seeds)
        feature = get_feature(dummy_state)
        network = actor_critic(env_config=env_config, buf_config=buf_config)
        param_rng, gru_rng = jax.random.split(rng)
        network_params = network.init({'params': param_rng, 'gru': gru_rng}, feature)
   
        # Optimizer - clip_by_global_norm / adam
        tx = optax.chain(
            optax.clip_by_global_norm(ppo_config.MAX_GRAD_NORM),
            optax.adam(ppo_config.LR, eps=1e-5),
        )

        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )

        # TRAIN LOOP
        def _update_step(update_state: UpdateState, _):
            train_state, rng = update_state
            
            # Initialize env_state
            rng, _rng = jax.random.split(rng)
            seeds = get_seeds(_rng, (num_envs,))
            env_state = batch_env.reset(seeds)

            # Bidding step
            rng, _rng = jax.random.split(rng)
            _rng = jax.random.split(_rng, num=num_envs)
            bid, faction = jax.vmap(bid_agent)(env_state, _rng)
            env_state, _ = batch_env.step_bid(env_state, bid, faction)

            # Factory placement step
            n_factories = env_state.board.factories_per_team[0].astype(jnp.int32)

            def _factory_placement_step(i, env_state_rng):
                env_state, rng = env_state_rng
                rng, _rng = jax.random.split(rng)
                factory_placement = factory_placement_agent(env_state, _rng)
                env_state, _ = batch_env.step_factory_placement(env_state, *factory_placement)
                return env_state, rng

            rng, _rng = jax.random.split(rng)
            env_state, _ = jax.lax.fori_loop(0, 2*n_factories, _factory_placement_step, (env_state, _rng))

            # Late game step
            def _env_step(runner_state, _):
                train_state, env_state, feature, rng = runner_state

                rng, gru_rng = jax.random.split(rng)
                # SELECT ACTION
                action, log_prob, value = network.apply(train_state.params, feature, rngs={'gru': gru_rng})
                
                empty_action = JuxAction.empty(env_cfg=env_config, buf_cfg=buf_config)
                empty_action_batch = tree_map(lambda x: jnp.broadcast_to(x, shape=(num_envs, *x.shape)), empty_action)
                # STEP ENV
                env_state, (_, reward, done, _) = batch_env.step_late_game(env_state, empty_action_batch)
                
                # Player 0 is the agent, Player 1 plays with null action (TODO)
                reward = reward[0]
                done = done[0]

                feature = get_feature(env_state)
                transition = Trajectory(
                    done, action, value, reward, log_prob, feature
                )
                runner_state = RunnerState(train_state, env_state, feature, rng)
                return runner_state, transition

            rng, _rng = jax.random.split(rng)
            runner_state = RunnerState(train_state, env_state, get_feature(env_state), _rng)

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, MAX_EPISODE_LENGTH
            )

            rng, _rng = jax.random.split(rng)
            _, _, last_val = network.apply(train_state.params, runner_state.obs, rngs={'gru': _rng})

            def _calculate_gae(traj_batch: Trajectory, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )
                    delta = reward + ppo_config.GAMMA * next_value * (1 - done) - value
                    gae = (
                        delta
                        + ppo_config.GAMMA * ppo_config.GAE_LAMBDA * (1 - done) * gae
                    )
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val)

            return advantages, targets

        rng, _rng = jax.random.split(rng)
        update_state = UpdateState(train_state, _rng)
        advantages, targets = jax.lax.scan(
            _update_step, update_state, None, ppo_config.N_UPDATES
        )
        return advantages, targets
        # return {"runner_state": runner_state, "metrics": metric}

    return train

def main(env_config, buf_config, ppo_config, seed):
    from agent import naive_bid_agent, random_factory_agent_batched
    from models import NaiveActorCritic

    rng = jax.random.PRNGKey(seed)
    rng, _rng = jax.random.split(rng)
    train = make_train(env_config, buf_config, ppo_config, NaiveActorCritic, naive_bid_agent, random_factory_agent_batched, _rng)
    advantages, targets = train(rng)
    print(advantages.shape)
    print(advantages)

if __name__ == "__main__":
    env_config = EnvConfig()
    buf_config = JuxBufferConfig(MAX_N_UNITS=1000)
    ppo_config = PPOConfig(
        N_UPDATES=10,
        N_ENVS=4,
        N_EPISODES_PER_ENV=4,
    )
    
    prng_seed = 42

    main(env_config, buf_config, ppo_config, prng_seed)
