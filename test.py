from tianshou.data import Collector
from tianshou.env import DummyVectorEnv, PettingZooEnv
from tianshou.policy import MultiAgentPolicyManager, RandomPolicy

from envs.predation import env, parallel_env # TODO switch this to custom env
from supersuit import pad_observations_v0

if __name__ == "__main__":
    # Step 1: Load the PettingZoo environment
    env = env(render_mode="human")
    agents = env.possible_agents
    env = pad_observations_v0(env)
    for agent in agents:
        print(f"Agent {agent} has action space {env.action_spaces[agent]} and observation space {env.observation_spaces[agent]}")

    # Step 2: Wrap the environment for Tianshou interfacing
    env = PettingZooEnv(env)

    # Step 3: Define policies for each agent
    policies = MultiAgentPolicyManager([RandomPolicy(), RandomPolicy(), RandomPolicy()], env)

    # Step 4: Convert the env to vector format
    env = DummyVectorEnv([lambda: env])

    # Step 5: Construct the Collector, which interfaces the policies with the vectorized environment
    collector = Collector(policies, env)

    # Step 6: Execute the environment with the agents playing for 1 episode, and render a frame every 0.1 seconds
    result = collector.collect(n_episode=1, render=0.05)