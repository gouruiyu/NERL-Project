#%%
import numpy as np
from envs.PreyArena_v1 import PreyArena_v1
from envs.PreyArena_v2 import PreyArena_v2
from algos.policy import MlpPolicy, RandomBaselinePolicy
import os
from PIL import Image
import pygame
import torch
from tqdm.notebook import tqdm
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

#%% Iniliaze logging
log_dir = f"figs/zero-shot_prosocial"
os.makedirs(log_dir, exist_ok=True)

#%% Initialize containers
reciprocal_policy = "resultsnece/never-ending_1124-220836_flowing-totem-31/100000_record_holder.pt"
selfish_policy = "resultsnece/never-ending_1124-220836_sage-haze-39/100000_record_holder.pt"
reciprocal_color = pygame.Color(71, 154, 95)
selfish_color = pygame.Color(193, 67, 60)
reciprocal_regular = []
reciprocal_faster = []
reciprocal_cramp = []
selfish_regular = []
selfish_faster = []
selfish_cramp = []

df = []

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
env = PreyArena_v1(
    render_mode = None,
    predator_accel= 6,
    max_cycles = 500,
    eat_predation_ratio= 20,
)
policy_reciprocal = MlpPolicy(env.observation_dim(), env.action_dim(), n_hidden_layer=1, hidden_size=16, init='xavier_normal', activation='tanh', backprop=False)
policy_selfish = MlpPolicy(env.observation_dim(), env.action_dim(), n_hidden_layer=1, hidden_size=16, init='xavier_normal', activation='tanh', backprop=False)
policy_reciprocal.load_state_dict(torch.load(reciprocal_policy))
policy_selfish.load_state_dict(torch.load(selfish_policy))


def rollout(env, df, policy1, policy2, name, n_rounds=100, separate=False):
    rews0 = []
    rews1 = []
    for i in range(n_rounds):
        if i % int(n_rounds / 5) == 0:
            acc_rew = match(env, policy1, policy2, gif_name=f"{name}-{i}", render=True)
        else:
            acc_rew = match(env, policy1, policy2)
        if separate:
            rews0.append(acc_rew[0])
            rews1.append(acc_rew[1])
        else:
            rews0.append(acc_rew[0])
            rews0.append(acc_rew[1])
    if separate:
        print(f"{name} mean: ", np.mean(rews0), " std ", np.std(rews0)) 
        print(f"{name} mean: ", np.mean(rews1), " std ", np.std(rews1)) 
    else:
        print(f"{name} mean: ", np.mean(rews0), "std: " , np.std(rews0), "25% percentile: ", np.percentile(rews0, 25), "75% percentile: ", np.percentile(rews0, 75))
        agenttype, envtype = name.split("_")
        if agenttype == "reciprocal":
            agenttype = "prosocial"
        if envtype == "cramp": 
            rews0 = [rew / 3 for rew in rews0]
        df.append([agenttype, envtype, np.mean(rews0), np.std(rews0), np.percentile(rews0, 25), np.percentile(rews0, 75)])

env_faster = PreyArena_v1(
    render_mode = None,
    predator_accel= 2,
    max_cycles = 500,
    eat_predation_ratio= 20,
)

rollout(env_faster, df, policy_reciprocal, policy_reciprocal, "reciprocal_slower2")
rollout(env_faster, df, policy_selfish, policy_selfish, "selfish_slower2")


rollout(env, df, policy_reciprocal, policy_reciprocal, "reciprocal_regular")
rollout(env, df, policy_selfish, policy_selfish, "selfish_regular")

rollout(env, df, policy_reciprocal, policy_selfish, "mixed", separate=True)
rollout(env, df, policy_selfish, policy_reciprocal, "mixed", separate=True)



env_faster = PreyArena_v1(
    render_mode = None,
    predator_accel= 8,
    max_cycles = 500,
    eat_predation_ratio= 20,
)

rollout(env_faster, df, policy_reciprocal, policy_reciprocal, "reciprocal_faster8")
rollout(env_faster, df, policy_selfish, policy_selfish, "selfish_faster8")


env_faster = PreyArena_v1(
    render_mode = None,
    predator_accel= 10,
    max_cycles = 500,
    eat_predation_ratio= 20,
)

rollout(env_faster, df,  policy_reciprocal, policy_reciprocal, "reciprocal_faster10")
rollout(env_faster, df, policy_selfish, policy_selfish, "selfish_faster10")


# env_faster = PreyArena_v1(
#     render_mode = None,
#     predator_accel= 6,
#     max_cycles = 500,
#     eat_predation_ratio= 20,
# )

# rollout(env_faster, df, policy_reciprocal, policy_reciprocal, "reciprocal_faster6")
# rollout(env_faster, df, policy_selfish, policy_selfish, "selfish_faster6")




env_cramp = PreyArena_v2(
    render_mode = None,
    predator_accel= 6,
    max_cycles = 500,
    eat_predation_ratio=20
)

rollout(env_cramp, df, policy_reciprocal, policy_reciprocal, "reciprocal_cramp")
rollout(env_cramp, df, policy_selfish, policy_selfish, "selfish_cramp")

print(df)
df = pd.DataFrame(df, columns=["agent", "env variant", "mean reward", "std", "25%", "75%"])
palette = ['tab:green', 'tab:orange']
sns.barplot(data=df, x="env variant", y="mean reward", hue="agent", palette=palette)
plt.savefig('zeroshot1_ps.png', dpi=300)

# %%
# reciprocal_regular
# %%
