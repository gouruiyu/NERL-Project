import argparse
import wandb
import time
import os
from envs.PreyArena import PreyArena
from algos.policy import MlpPolicy
from PIL import Image
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--expr_name', type=str, default='test')
    parser.add_argument('--render_mode', type=str, default='rgb_array')
    # Environment parameters
    parser.add_argument('--predator_acceleration', type=float, default=0.5)
    # Agent parameters
    parser.add_argument('--hidden_size', type=int, default=16)
    parser.add_argument('--n_hidden_layer', type=int, default=1)
    parser.add_argument('--init', type=str, default='xavier_uniform')
    parser.add_argument('--activation', type=str, default='relu')
    # Training parameters

    args = parser.parse_args()

    args.expr_name = f"{args.expr_name}_{time.strftime('%Y%m%d-%H%M%S')}"
    # Set up local gif directory
    gif_dir = f"results/{args.expr_name}/gifs"
    os.makedirs(gif_dir, exist_ok=True)

    env = PreyArena(
        render_mode=args.render_mode,
        predator_accel = args.predator_acceleration
    )

    for i in range(1):
        random_policy = MlpPolicy(
            state_dim=10, 
            action_dim=5, 
            n_hidden_layer=args.n_hidden_layer, 
            hidden_size=args.hidden_size, 
            init=args.init,
            activation=args.activation)
        obs, rews, done, global_info = env.reset()
        frame_buffer = []
        for _ in range(100):
            frame_buffer.append(env.render())
            actions = random_policy.act(obs, deterministic=False)
            # actions = np.random.randint(0, 5, 2)
            obs, rews, done, _= env.step(actions)
            if done:
                break

        # Save gif to local directory
        gif_path = f"{gif_dir}/run{i}.gif"
        frame_buffer = [Image.fromarray(frame) for frame in frame_buffer]
        frame_buffer[0].save(gif_path, save_all=True, append_images=frame_buffer[1:], duration=20, loop=0)
        # wandb.log({"gif": wandb.Video(np.array(frame_buffer), fps=10, format="gif")}, commit=False)
        # wandb.save(gif_path)
        

    env.close()