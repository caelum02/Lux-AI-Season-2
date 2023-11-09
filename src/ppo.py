import jax
import jax.numpy as jnp
import optax
from typing import NamedTuple
from flax.training.train_state import TrainState
import distrax

from jux.env import JuxEnv, JuxEnvBatch
from jux.config import JuxBufferConfig, EnvConfig

from preprocess import get_feature, batch_get_feature
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
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
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
        feature = batch_get_feature(dummy_state)
        network = actor_critic
        network_params = network.init(_rng, feature)
   
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
                factory_placement = agent.random_factory_agent_batched(env_state, _rng)
                env_state, _ = batch_env.step_factory_placement(env_state, *factory_placement)
                return env_state, rng

            rng, _rng = jax.random.split(rng)
            env_state, _ = jax.lax.fori_loop(0, 2*n_factories, _factory_placement_step, (env_state, _rng))

            # Late game step
            def _env_step(runner_state):
                train_state, env_state, feature, rng = runner_state

                # SELECT ACTION
                # TODO: Check if get_feature is vectorized
                action, value = network.apply(train_state.params, feature)

                rng, _rng = jax.random.split(rng)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)

                # STEP ENV
                env_state, (_, reward, done, _) = env_state.late_game_step(env_state, action)
                transition = Trajectory(
                    done, action, value, reward, log_prob, get_feature(env_state)
                )
                runner_state = RunnerState(train_state, env_state, feature, rng)
                return runner_state, transition

            rng, _rng = jax.random.split(rng)
            runner_state = RunnerState(train_state, env_state, feature, _rng)

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, ppo_config.NUM_STEPS
            )

            # TODO : From here 11/8
            return UpdateState(train_state, rng), traj_batch

        rng, _rng = jax.random.split(rng)
        update_state = UpdateState(train_state, _rng)
        update_state, traj_batch = jax.lax.scan(
            _update_step, update_state, None, n_updates
        )
        return update_state, traj_batch
        # return {"runner_state": runner_state, "metrics": metric}

    return train

def main(env_config, buf_config, ppo_config, seed):
    from agent import naive_bid_agent, random_factory_agent
    from models import NaiveActorCritic

    rng = jax.random.PRNGKey(seed)
    rng, _rng = jax.random.split(rng)
    train = make_train(env_config, buf_config, ppo_config, NaiveActorCritic, naive_bid_agent, random_factory_agent, _rng)
    train(rng)


if __name__ == "__main__":
    env_config = EnvConfig()
    buf_config = JuxBufferConfig(MAX_N_UNITS=1000)
    ppo_config = PPOConfig
    
    prng_seed = 42

    main(env_config, buf_config, ppo_config, prng_seed)
