"""

Environment implemented based on PettingZoo MPE

"""
import numpy as np
from pettingzoo.utils.conversions import parallel_wrapper_fn
# from .utils.core import Agent, Landmark, World
from .utils.simple_env import SimpleEnv, make_env

class EntityState:  # physical/external base state of all entities
    def __init__(self):
        self.p_pos = None

class AgentState(EntityState):
    def __init__(self):
        super().__init__()

class Entity:  # properties and state of physical world entity
    def __init__(self):
        # name
        self.name = ""
        # properties:
        self.size = 0.050
        # entity can move / be pushed
        self.movable = False
        # entity collides with others
        self.collide = True
        # color
        self.color = None
        # state
        self.state = EntityState()

class Prey(Entity):
    def __init__(self):
        super().__init__()
        self.state = AgentState()
        self.action = Action()
        self.action_callback = None

class Predator(Entity):
    def __init__(self):
        super().__init__()

class Food(Entity):
    def __init__(self):
        super().__init__()

class World:
    def __init__(self):
        self.preys = []
        self.predators = []
        self.foods = []
        self.dt = 0.1
    
    @property
    def entities(self):
        return self.preys + self.predators + self.foods
    
    @property
    def step(self):


class raw_env(SimpleEnv):
    def __init__(
        self,
        num_prey=2,
        num_predator=1,
        num_food=1,
        max_cycles=25,
        continuous_actions=False,
        render_mode=None,
    ):
        scenario = Scenario()
        world = scenario.make_world(num_prey, num_predator, num_food)
        super().__init__(
            scenario=scenario,
            world=world,
            render_mode=render_mode,
            max_cycles=max_cycles,
            continuous_actions=continuous_actions,
        )


env = make_env(raw_env)
parallel_env = parallel_wrapper_fn(env)

class Scenario():
    def make_world(self, num_prey=2, num_predator=1, num_food=1):
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_agents = num_predator + num_prey
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.predate = True if i < num_predator else False
            base_name = "predator" if agent.predate else "prey"
            base_index = i if i < num_predator else i - num_predator
            agent.name = f"{base_name}_{base_index}"
            agent.size = 0.075 if agent.adversary else 0.05
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_food)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = "landmark %d" % i
            landmark.collide = True
            landmark.movable = False
            landmark.size = 0.2
            landmark.boundary = False
        return world

    def reset_world(self, world, np_random):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = (
                np.array([0.35, 0.85, 0.35])
                if not agent.adversary
                else np.array([0.85, 0.35, 0.35])
            )
            # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np_random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            if not landmark.boundary:
                landmark.state.p_pos = np_random.uniform(-0.9, +0.9, world.dim_p)
                landmark.state.p_vel = np.zeros(world.dim_p)

    def benchmark_data(self, agent, world):
        # returns data for benchmarking purposes
        if agent.adversary:
            collisions = 0
            for a in self.good_agents(world):
                if self.is_collision(a, agent):
                    collisions += 1
            return collisions
        else:
            return 0

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    # return all agents that are not adversaries
    def good_agents(self, world):
        return [agent for agent in world.agents if not agent.adversary]

    # return all adversarial agents
    def adversaries(self, world):
        return [agent for agent in world.agents if agent.adversary]

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark
        main_reward = (
            self.adversary_reward(agent, world)
            if agent.adversary
            else self.agent_reward(agent, world)
        )
        return main_reward

    def agent_reward(self, agent, world):
        # Agents are negatively rewarded if caught by adversaries
        rew = 0
        shape = False
        adversaries = self.adversaries(world)
        if (
            shape
        ):  # reward can optionally be shaped (increased reward for increased distance from adversary)
            for adv in adversaries:
                rew += 0.1 * np.sqrt(
                    np.sum(np.square(agent.state.p_pos - adv.state.p_pos))
                )
        if agent.collide:
            for a in adversaries:
                if self.is_collision(a, agent):
                    rew -= 10

        # agents are penalized for exiting the screen, so that they can be caught by the adversaries
        def bound(x):
            if x < 0.9:
                return 0
            if x < 1.0:
                return (x - 0.9) * 10
            return min(np.exp(2 * x - 2), 10)

        for p in range(world.dim_p):
            x = abs(agent.state.p_pos[p])
            rew -= bound(x)

        return rew

    def adversary_reward(self, agent, world):
        # Adversaries are rewarded for collisions with agents
        rew = 0
        shape = False
        agents = self.good_agents(world)
        adversaries = self.adversaries(world)
        if (
            shape
        ):  # reward can optionally be shaped (decreased reward for increased distance from agents)
            for adv in adversaries:
                rew -= 0.1 * min(
                    np.sqrt(np.sum(np.square(a.state.p_pos - adv.state.p_pos)))
                    for a in agents
                )
        if agent.collide:
            for ag in agents:
                for adv in adversaries:
                    if self.is_collision(ag, adv):
                        rew += 10
        return rew

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:
            if not entity.boundary:
                entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # communication of all other agents
        comm = []
        other_pos = []
        other_vel = []
        for other in world.agents:
            if other is agent:
                continue
            comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)
            if not other.adversary:
                other_vel.append(other.state.p_vel)
        return np.concatenate(
            [agent.state.p_vel]
            + [agent.state.p_pos]
            + entity_pos
            + other_pos
            + other_vel
        )