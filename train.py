import numpy as np
from envs.predation import env, parallel_env
from utils import get_args
import wandb

if __name__ == "__main__":
    env = env(render_mode="human")
    env.reset()
    agents = env.possible_agents
    args = get_args()
    # wandb.init(project="nerl", entity="ruiyug", name=args.expr_name, config=delattr(args, "expr_name"))

    for i in range(100):
        import ipdb; ipdb.set_trace()
        env.step(env.action_space.sample())
        env.render()