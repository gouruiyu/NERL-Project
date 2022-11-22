import numpy as np
from envs.PreyArena import *
from utils import get_args
import wandb
import time
from tqdm import tqdm
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--expr_name', type=str, default='test')
    parser.add_argument('--agent_acceleration', type=float, default=4.)
    parser.add_argument('--predator_acceleration', type=float, default=2.)
    parser.add_argument('--eval_num_episodes', type=int, default=100)
    parser.add_argument('--noise_regulator', type=float, default=0.0)
    args = parser.parse_args()
    
    # Initialize wandb
    wandb.init(
        project="prey-arena-env",
        entity="ruiyg",
        config=vars(args)
    )
    
    env = PreyArena(
        render_mode=None, 
        agent_accel=args.agent_acceleration, 
        predator_accel=args.predator_acceleration,
        noise_regulator=args.noise_regulator
    )

    for episode_length in range(0, 501, 25):
    # for episode_length in [100]:
        win_rates = [0,0]
        food_eaten_freq = [0,0]
        chewed_rates = []
        for i in range(args.eval_num_episodes):
            env.reset()
            chewed = 0
            j = 0
            for _ in range(episode_length):
                # env.render()
                # print(env.global_state())
                actions = np.random.randint(0, 5, 2)
                next_obs, rews, done, _= env.step(actions)
                # print(f"Reward at step {j}: {rews}")
                if rews[0] > 10:
                    food_eaten_freq[0] += 1
                elif rews[0] < 0:
                    chewed += 1
                if rews[1] > 10:
                    food_eaten_freq[1] += 1
                elif rews[1] < 0:
                    chewed += 1
                j += 1
                if done:
                    break
            # print(f"Episode {i} finished. accumulated rewards: {env.agents[0]._accumulate_rewards}, {env.agents[1]._accumulate_rewards}")
            chewed_rate = chewed / (j+1)
            chewed_rates.append(chewed_rate)
            win_agent = env.agents[1]._accumulate_rewards > env.agents[0]._accumulate_rewards
            win_rates[win_agent] += 1

        wandb.log({
            "win_rates": {
                "R2D2": win_rates[0],
                "C-3PO": win_rates[1]
            },
            "food_eaten_freq": {
                "R2D2": food_eaten_freq[0],
                "C-3PO": food_eaten_freq[1]
            },
            "win_rate": win_rates[0] / args.eval_num_episodes,
            "food_eaten_rate": food_eaten_freq[0] / args.eval_num_episodes, # Percentage of times an agent can get food through random action
            "chewed_on_time": np.mean(chewed_rates), # Percentage of times an agent can get chewed on through random action
        }, step=episode_length)
    
    # print(f"Game over!\n win_rate: {win_rates} \n food_eaten_freq: {food_eaten_freq}")
    env.close()