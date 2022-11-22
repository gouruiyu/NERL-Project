import numpy as np
from envs.PreyArena import *
from utils import get_args
import wandb
import time

if __name__ == "__main__":
    env = PreyArena(render_mode="human")
    env.reset()
    # args = get_args()
    # wandb.init(project="nerl", entity="ruiyug", name=args.expr_name, config=delattr(args, "expr_name"))

    for i in range(10000):
        env.render()
        # print(env.global_state())
        actions = np.random.randint(0, 5, 2)
        env.step(actions)
    
    env.close()