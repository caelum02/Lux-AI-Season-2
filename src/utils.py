import jux

from jux.env import JuxEnv
from jux.config import JuxBufferConfig
from jux.state import State
from jux.actions import JuxAction

def replay_run_early_phase(jux_env: JuxEnv, state: State, lux_actions):
    """
        Util function | skip game until late_game stage
        
        return:
            state: game state right after factory placement phase
            lux_actions: lux action iterator synchronized to `state`
    """    

    print(f"[Replay Util] Replaying early steps")

    # Bid Step
    bid, faction = jux.actions.bid_action_from_lux(next(lux_actions))
    state, (obs, rwd, dones, infos) = jux_env.step_bid(state, bid, faction)

    # Factory Placement Step
    while state.real_env_steps < 0:
        lux_act = next(lux_actions)
        spawn, water, metal = jux.actions.factory_placement_action_from_lux(lux_act)
        state, _ = jux_env.step_factory_placement(state, spawn, water, metal)
    
    print(f"[Replay Util] Replaying early steps - Done")

    return state, lux_actions

def replay_run_n_late_game_step(n: int, jux_env: JuxEnv, state: State, lux_actions):
    
    for i in range(n):
        print(f"[Replay Util] Replaying {i+1}/{n} steps")
        lux_act = next(lux_actions)
        jux_act = JuxAction.from_lux(state, lux_act)

        # step
        state, _ = jux_env.step_late_game(state, jux_act)
    
    return state, lux_actions

if __name__ == "__main__":
    lux_env, lux_actions = jux.utils.load_replay('replays/52958192.json')
    jux_env, state = JuxEnv.from_lux(lux_env, buf_cfg=JuxBufferConfig(MAX_N_UNITS=200))
    state, lux_actions = replay_run_early_phase(jux_env, state, lux_actions)
    state, lux_actions = replay_run_n_late_game_step(100, jux_env, state, lux_actions)