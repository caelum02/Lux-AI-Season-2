import jux
from jux.env import JuxEnv
from jux.config import JuxBufferConfig

from utils import replay_run_early_phase, replay_run_n_late_game_step

MAX_N_UNITS = 200
n = 376

lux_env, lux_actions = jux.utils.load_replay('https://www.kaggleusercontent.com/episodes/52900827.json')
jux_env, state = JuxEnv.from_lux(lux_env, buf_cfg=JuxBufferConfig(MAX_N_UNITS=MAX_N_UNITS))

state, lux_actions = replay_run_early_phase(jux_env, state, lux_actions)
state, lux_actions = replay_run_n_late_game_step(n, jux_env, state, lux_actions)    