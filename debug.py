import jux
from jux.env import JuxEnv
from jux.config import JuxBufferConfig

from utils import replay_run_early_phase, replay_run_n_late_game_step

MAX_N_UNITS = 200
n = 281

# real_env_step 278에서 갑자기 245번 유닛의 정보가 사라지고 그 자리가 246번으로 대체됨
# id2idx가 바뀌어서 발생하는 것으로 예상이 됨

# replay_id = 52900827
replay_id = 52958192
lux_env, lux_actions = jux.utils.load_replay(f'data/{replay_id}.json')
jux_env, state = JuxEnv.from_lux(lux_env, buf_cfg=JuxBufferConfig(MAX_N_UNITS=MAX_N_UNITS))

# state, lux_actions = replay_run_early_phase(jux_env, state, lux_actions)
# state, lux_actions = replay_run_n_late_game_step(n, jux_env, state, lux_actions)  

state, lux_actions, lux_env = replay_run_early_phase(jux_env, state, lux_actions, lux_env=lux_env)
state, lux_actions, lux_env = replay_run_n_late_game_step(n, jux_env, state, lux_actions, lux_env=lux_env)  