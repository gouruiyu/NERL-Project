import argparse
import wandb
import time
import os
from envs.PreyArena import PreyArena
from envs.PreyArena_v1 import PreyArena_v1
from algos.policy import MlpPolicy, RandomBaselinePolicy
from PIL import Image
import numpy as np
import copy

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--expr_name', type=str, default='never-ending')
    parser.add_argument('--env_name', type=str, default='PreyArena-v1')
    parser.add_argument('--altruism', type=float, default=0) # 0: self-interest, 0.5: egalitarian, 1.0: altruistic
    # Environment parameters
    parser.add_argument('--predator_acceleration', type=float, default=4.)
    parser.add_argument('--max_cycles', type=int, default=500)
    parser.add_argument('--eat_predation_ratio', type=int, default=20)
    # Agent parameters
    parser.add_argument('--hidden_size', type=int, default=16)
    parser.add_argument('--n_hidden_layer', type=int, default=1)
    parser.add_argument('--init', type=str, default='xavier_normal')
    parser.add_argument('--activation', type=str, default='tanh')
    parser.add_argument('--deterministic', type=bool, default=True)
    # Training parameters
    parser.add_argument('--n_tournament', type=int, default=int(5e4))
    parser.add_argument('--n_rounds', type=int, default=8)
    parser.add_argument('--n_population', type=int, default=100)
    parser.add_argument('--mutation_std', type=float, default=0.1)
    args = parser.parse_args()
    
    # Initialize wandb
    wandb.init(
        project="prey-arena-tournament-gs-push",
        entity="ruiyg",
        config=vars(args)
    )

    args.expr_name = f"{args.expr_name}_{time.strftime('%m%d-%H%M%S')}_{wandb.run.name}"
    # Set up local gif directory
    log_dir = f"resultsnecepush/{args.expr_name}"
    os.makedirs(log_dir, exist_ok=True)
    
    SAVE_FREQ = 100
    # NUM_GIF = 10
    # GIF_FREQ = int(args.n_tournament / NUM_GIF)
    GIF_FREQ = 10000
    assert GIF_FREQ % SAVE_FREQ == 0, "GIF_FREQ must be a multiple of SAVE_FREQ"
    
    ############################### Training ###############################

    if args.env_name == 'PreyArena':
        env = PreyArena(
            render_mode = None,
            predator_accel = args.predator_acceleration,
            max_cycles = args.max_cycles,
        )
    elif args.env_name == 'PreyArena-v1':
        env = PreyArena_v1(
            render_mode = None,
            predator_accel = args.predator_acceleration,
            max_cycles = args.max_cycles,
            eat_predation_ratio = args.eat_predation_ratio
        )
    
    action_dim = env.action_dim()
    obs_dim = env.observation_dim()
    
    def match(env, policy1, policy2, tournament_idx, idx, gif_name=None, render=False):
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
            wandb.log(
                {f"{gif_name}_gif": wandb.Video(np.transpose(np.array(frame_buffer), (0, 3, 1, 2)), fps=30, format="gif")},
                step=tournament_idx, commit=False)
            gif_path = f"{log_dir}/{tournament:05d}_record_holder_{gif_name}.gif"
            frame_buffer = [Image.fromarray(frame) for frame in frame_buffer]
            frame_buffer[0].save(gif_path, save_all=True, append_images=frame_buffer[1:], duration=20, loop=0)

        accumulated_rewards_orig = accumulated_rewards.copy()
        accumulated_rewards[0] = (1-args.altruism) * accumulated_rewards[0] + args.altruism * accumulated_rewards[1]
        accumulated_rewards[1] = (1-args.altruism) * accumulated_rewards[1] + args.altruism * accumulated_rewards[0]
        return accumulated_rewards, accumulated_rewards_orig, timestep
    
    def rollout(env, policy1, policy2, tournament_idx, n_rounds=args.n_rounds, gif_name=None, render=False):
        """ return >0 if policy1 wins, <0 if policy2 wins, 0 if tie """
        wins = 0
        accs = []
        for i in range(n_rounds):
            if i == 0:
                acc_adj, acc_orig, episode_len = match(env, policy1, policy2, tournament_idx, i, gif_name=gif_name, render=render)
            else:
                acc_adj, acc_orig, episode_len = match(env, policy1, policy2, tournament_idx, i, gif_name=None, render=False)
            wins += acc_adj[0] > acc_adj[1]
            accs.append([acc_adj, acc_orig, episode_len])
        return wins - n_rounds/2, accs
        
    population = [MlpPolicy(state_dim=obs_dim, action_dim=action_dim, hidden_size=args.hidden_size, n_hidden_layer=args.n_hidden_layer, init=args.init, activation=args.activation) for _ in range(args.n_population)]
    
    winning_streak = [0] * args.n_population
    # history = []
    baseline = RandomBaselinePolicy()
    
    for tournament in range(args.n_tournament+1):
        a, b = np.random.choice(args.n_population, 2, replace=False)
        score, accs = rollout(env, population[a], population[b], tournament_idx=tournament)
        # history.append(accs)
        if score == 0: # tie
            population[a].add_noise(args.mutation_std)
        elif score > 0: # policy a won
            population[b] = copy.deepcopy(population[a])
            population[b].add_noise(args.mutation_std)
            winning_streak[b] = winning_streak[a]
            winning_streak[a] += 1
        elif score < 0: # policy b won
            population[a] = copy.deepcopy(population[b])
            population[a].add_noise(args.mutation_std)
            winning_streak[a] = winning_streak[b]
            winning_streak[b] += 1
        
        if tournament % SAVE_FREQ == 0:
            record_holder = np.argmax(winning_streak)
            render = tournament % GIF_FREQ == 0
            if render:
                population[record_holder].save(f"{log_dir}/{tournament:05d}_record_holder.pt")
            # self-play evaluation
            _, accs_self = rollout(env, population[record_holder], population[record_holder], tournament_idx=tournament, n_rounds=args.n_rounds, gif_name="self_play", render=render)
            # baseline evaluation
            _, accs_bsln = rollout(env, population[record_holder], baseline, tournament_idx=tournament, n_rounds=args.n_rounds, gif_name="baseline", render=render)
            wandb.log({
                "reward_self_play": np.mean([acc[0][0] for acc in accs_self]),
                "reward_self_play_with_altruism_adaption": np.mean([acc[1][0] for acc in accs_self]),
                "reward_baseline": np.mean([acc[0][0] for acc in accs_bsln]),
                "reward_baseline_with_altruism_adaption": np.mean([acc[1][0] for acc in accs_bsln])
            }, step=tournament, commit=True)
            
            
    # Save run information
    # np.savetxt(f"{log_dir}/history.txt", history)
    # np.savetxt(f"{log_dir}/winning_streak.txt", winning_streak)

    env.close()
