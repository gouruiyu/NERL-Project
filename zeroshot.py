#%%
import numpy as np
from envs.PreyArena_v2 import PreyArena_v2
from algos.policy import MlpPolicy, RandomBaselinePolicy
import os
from PIL import Image
import pygame
import torch
from tqdm.notebook import tqdm

#%% Iniliaze logging
log_dir = f"figs/zero-shot"
os.makedirs(log_dir, exist_ok=True)

#%% Initialize containers
reciprocal_policy = "resultsnece/never-ending_1124-220836_flowing-totem-31/390000_record_holder.pt"
selfish_policy = "resultsnece/never-ending_1124-220836_sage-haze-39/390000_record_holder.pt"
reciprocal_color = pygame.Color(71, 154, 95)
selfish_color = pygame.Color(193, 67, 60)
reciprocal_regular = []
reciprocal_faster = []
reciprocal_cramp = []
selfish_regular = []
selfish_faster = []
selfish_cramp = []

#%% Experiment config
faster_predator_accel = 10
cramper_width = 300
max_cycles = 500
eat_predation_ratio = 20

#%%
def match(env, policy1, policy2, gif_name=None, render=False):
    obs, rews, done, _ = env.reset()
    done = False
    accumulated_rewards = np.zeros(2)
    timestep = 0
    if render: 
        frame_buffer = []
    while not done:
        if render:
            frame_buffer.append(env.render()) 
        action1 = policy1.act(obs[0])
        action2 = policy2.act(obs[1])
        timestep += 1
        obs, rews, done, _ = env.step([action1, action2])
        accumulated_rewards += rews
    if render:
        gif_path = f"{log_dir}/{gif_name}.gif"
        frame_buffer = [Image.fromarray(frame) for frame in frame_buffer]
        frame_buffer[0].save(gif_path, save_all=True, append_images=frame_buffer[1:], duration=30, loop=0)
    return accumulated_rewards

#%% Regular
n_rounds = 20
env = PreyArena_v2()
policy1 = MlpPolicy(env.observation_dim(), env.action_dim(), n_hidden_layer=1, hidden_size=16, init='xavier_normal', activation='relu', backprop=False)
policy2 = MlpPolicy(env.observation_dim(), env.action_dim(), n_hidden_layer=1, hidden_size=16, init='xavier_normal', activation='relu', backprop=False)
policy1.load_state_dict(torch.load(reciprocal_policy))
policy2.load_state_dict(torch.load(reciprocal_policy))

for i in tqdm(range(n_rounds)):
    if i % int(n_rounds / 5) == 0:
        acc_rew = match(env, policy1, policy2, gif_name=f"reciprocal-{i}.gif", render=True)
    else:
        acc_rew = match(env, policy1, policy2)
    reciprocal_regular.append(acc_rew)
# %%
reciprocal_regular
# %%
