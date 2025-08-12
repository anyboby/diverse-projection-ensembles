from common.env_wrappers import BASE_FPS_ATARI, BASE_FPS_PROCGEN, create_env
from common.pedqn import PEDQN
from common.ids import IDS
from common.bdqn import BDQN
import torch
import time
from pathlib import Path
from common.utils import LinearSchedule
from tqdm import trange
import matplotlib.pyplot as plt
import numpy as np
from os import walk

PATH = Path("/home/run.pt")

algo="pedqn"
chckpt = torch.load(PATH)

args = chckpt["args"]
args.stack_skip = 1
args.save_dir = "checkpoints/debug"
args.parallel_envs=1
args.subproc_vecenv=False
args.record=False
args.verbosity=1
state_dict = chckpt["state_dict"]
game_frame = chckpt["game_frame"]
args.record_every=1e6
env = create_env(args, decorr_steps=None)
states = env.reset()

if algo=="pedqn":
    args.mean_adjustment_ri = 0.0
    args.gamma_i = 0.997
    model = PEDQN(env, args)
    model.q_policy.load_state_dict(state_dict)
    model.q_target.load_state_dict(state_dict)
    model.intr_rms.load_state_dict(chckpt["intr_rms_state_dict"])
    beta_schedule = LinearSchedule(0, initial_value=args.init_beta, final_value=args.final_beta, decay_time=args.beta_decay_frames)
    beta = beta_schedule(game_frame)
    model.beta = beta
elif algo=="ids":
    model = IDS(env, args)
    model.q_policy.load_state_dict(state_dict)
    model.q_target.load_state_dict(state_dict)
elif algo=="bdqn":
    model = BDQN(env, args)
    model.q_policy.load_state_dict(state_dict)
    model.q_target.load_state_dict(state_dict)
    model.reset_active_model(0)

batch_size = 512

eps = 0.05
sleep = 0.01
last_actions = 0
frame_skip = 1

total_timsteps = 10000
t = trange(0, total_timsteps + 1, args.parallel_envs)
t_n = range(0, total_timsteps + 1, args.parallel_envs)

for frame in t_n:
    iter_start = time.time()
    # compute actions to take in all parallel envs, asynchronously start environment step
    actions = model.act(states, eps)
    if algo=="pedqn":
        bonus, u, q, q_a = model.get_bonuses(states)

    if not frame%frame_skip==0:
        actions = last_actions
    env.step_async(actions)
    
    last_actions = actions
    next_states, rewards, dones, infos = env.step_wait()   
    
    states = next_states
    env.render()
    time.sleep(sleep)
env.close()
