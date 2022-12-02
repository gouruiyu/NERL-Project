from pettingzoo.utils.env import ParallelEnv
from gymnasium import spaces
import numpy as np
import pygame
import torch

"""
Version changelog:
- Food will be generated randomly and will respawn after being eaten
- The reward for eating food changes from 100 to 20
"""


WORLD_WIDTH = 600
FRICTION_DECAY = 0.9
BACKGROUND_COLOR = pygame.Color(0, 0, 0)
FOOD_COLOR = pygame.Color(159, 226, 191) #9FE2BF
PREDATOR_COLOR = pygame.Color(255, 127, 80) #FF7F50
AGENT_COLOR = pygame.Color(100, 149, 237) # 6495ED
BUFFER_SPACE = 5.


PREDATOR_POS_ORIGINAL = np.array([300, 300.])
PREDATOR_POS_ORIGINAL.flags["WRITEABLE"] = False
FOOD_POS_ORIGINAL = np.array([300., 100.])
FOOD_POS_ORIGINAL.flags["WRITEABLE"] = False
AGENTS_POS_ORIGINAL = np.array([[100., 500.], [500., 500.]])
AGENTS_POS_ORIGINAL.flags["WRITEABLE"] = False

class Entity():
    def __init__(self, name=None, pos=None, noise=0., color=None, radius=0.):
        self.name = name
        self.pos_original = pos # immutable
        # pos_noise = np.random.uniform(-BUFFER_SPACE, BUFFER_SPACE, size=2)
        # self.pos = pos + pos_noise
        self.color = color
        self.radius = radius
        self.noise = noise
        self.reset()
    
    def reset(self):
        self.pos = np.copy(self.pos_original) # mutable
        self.pos += self.noise * np.random.uniform(-1, 1, size=2)
        
    def _bounce(self):
        if self.pos[0] < self.radius:
            self.pos[0] = self.radius
            self.vel[0] *= -0.5
        if self.pos[0] > WORLD_WIDTH - self.radius:
            self.pos[0] = WORLD_WIDTH - self.radius
            self.vel[0] *= -0.5
        if self.pos[1] < self.radius:
            self.pos[1] = self.radius
            self.vel[1] *= -0.5
        if self.pos[1] > WORLD_WIDTH - self.radius:
            self.pos[1] = WORLD_WIDTH - self.radius
            self.vel[1] *= -0.5
            
class Food(Entity):
    def __init__(self, name="food", pos=FOOD_POS_ORIGINAL, noise=0., color=FOOD_COLOR, radius=10.):
        super().__init__(name, pos, noise, color, radius)
    
    def reset(self):
        self.pos = np.random.uniform(0, WORLD_WIDTH, size=2)
        self.vel = np.zeros(2)

class Agent(Entity):
    def __init__(self, name, pos=AGENTS_POS_ORIGINAL[0], noise=0., color=AGENT_COLOR, radius=15., accel=5.):
        super().__init__(name, pos, noise, color, radius)
        self.accel = accel
        self.reset()
    
    def reset(self):
        # self.pos = np.copy(self.pos_original) # mutable
        # self.pos += self.noise * np.random.uniform(-1, 1, size=2)
        super().reset()
        self.vel = np.zeros(2)
        self._accumulate_rewards = 0
        
    def update(self, action):
        self.vel *= FRICTION_DECAY
        if (action == 0): # noop
            pass
        elif (action == 1): # up
            self.vel[1] -= self.accel
        elif (action == 2): # down
            self.vel[1] += self.accel
        elif (action == 3): # left
            self.vel[0] -= self.accel
        elif (action == 4): # right
            self.vel[0] += self.accel
        else:
            print("Invalid action!")
        self.pos += self.vel
        self._bounce()
        return 
        
class Predator(Entity):
    def __init__(self, name="predator", pos=PREDATOR_POS_ORIGINAL, noise=0., color=PREDATOR_COLOR, radius=20., accel=4.,):
        super().__init__(name, pos, noise, color, radius)
        self.accel = accel
        self.vel = np.zeros(2)
    
    def reset(self):
        # self.pos = np.copy(self.pos_original) # mutable
        # self.pos += self.noise * np.random.uniform(-1, 1, size=2)
        super().reset()
        self.vel = np.zeros(2)
        self._accumulate_rewards = 0
    
    def update(self, preys):
        """
        Given array of preys, predator follows simple heuristic to chase the nearest one
        # TODO: optimize to use distance matrix
        """
        rel_positions = np.array([prey.pos for prey in preys]) - self.pos
        closest = np.argmin(np.linalg.norm(rel_positions, axis=1))
        # print(f"Predator: I am chasing {preys[closest].name}!")
        # apply accelration in the direction of the closest prey inverse to the distance
        self.vel *= FRICTION_DECAY
        self.vel += self.accel * rel_positions[closest] / np.linalg.norm(rel_positions[closest])
        # self.vel += self.accel * rel_positions[closest] / np.sum(np.abs(rel_positions[closest]))
        self.pos += self.vel
        self._bounce()
        return


class PreyArena_v1(ParallelEnv):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }
    
    def __init__(
        self,
        max_cycles=1000,
        render_mode=None,
        agent_names=["R2D2", "C-3PO"],
        agent_accel=5.,
        predator_accel=3.,
        noise_regulator=0.1,
        eat_predation_ratio=20
        # denser_reward=False
        ) -> None:
        super().__init__()
        
        # rendering options
        self.render_mode = render_mode
        self.renderOn = False
        
        pygame.init()
        self.viewer = None
        self.width = WORLD_WIDTH
        self.height = WORLD_WIDTH
        self.screen = pygame.Surface((self.width, self.height))
        
        #  Stochasticity sources: food respawn location
        self.noise_regulator = noise_regulator
        self.eat_predation_ratio = eat_predation_ratio
        self.max_cycles = max_cycles
        
        # Create entities
        self.agent_names = agent_names
        self.predator = Predator(noise=self.noise_regulator, accel=predator_accel)
        self.food = Food(noise=self.noise_regulator)
        self.agents = [Agent(name, pos=AGENTS_POS_ORIGINAL[i], noise=self.noise_regulator, accel=agent_accel) for i, name in enumerate(agent_names)]
        self.entities = self.agents + [self.food, self.predator] # FIXME: decouple the order restriction
        self.reset()
        self.dt = 1 # simulation timestep
        
    def _update_dm(self):
        """Update the distance matrix.
        Fixed order for now: agent1, agent2, food, predator
        """
        x = np.array([entity.pos for entity in self.entities])
        self.dm = np.sqrt(np.sum((x[:, None] - x[None, :]) ** 2, axis=-1))
        
    def _collide(self, idx1, idx2):
        """Check if two entities have collided."""
        return self.dm[idx1, idx2] < (self.entities[idx1].radius + self.entities[idx2].radius)
        
    def reset(self, seed=None, return_info=False, options=None):
        # reset entities' location
        # np.random.seed(seed)
        self.predator.reset()
        self.food.reset()
        for agent in self.agents:
            agent.reset()
        self.timestep = 0
        self.terminate = False
        self._update_dm()
        # return self.observation_array(), self._cal_reward(), self.terminate, self.global_state()
        return self.observation_array(), self._cal_reward(), self.terminate, None
    
    
    def global_state(self):
        """
        Return a dict of the current state including all entities' pos and vel
        """
        s = {}
        s["food_pos"] = self.food.pos
        s["predator_pos"] = self.predator.pos
        s["predator_vel"] = self.predator.vel
        for agent in self.agents:
            s[agent.name + "_pos"] = agent.pos
            s[agent.name + "_vel"] = agent.vel
        return s
    
    def step(self, actions):
        """_summary_

        Args:
            actions (num_agents * Discrete(5)): noop, up, down, left, right acceleration
        """
        self.predator.update(self.agents)
        for i, agent in enumerate(self.agents):
            agent.update(actions[i])
        self._update_dm()
        rews = self._cal_reward()
        self.timestep += 1
        if self.timestep >= self.max_cycles:
            self.terminate = True
        # if self.render_mode == "human":
        #     self.render()
        return self.observation_array(), rews, self.terminate, None
            
    def _cal_reward(self):
        rews = np.zeros(len(self.agents))
        for i, agent in enumerate(self.agents):
            if self._collide(i, -1):
                rews[i] += -5
                # print(f"Predator is chewing {agent.name}!")
            if self._collide(i, -2):
                rews[i] += self.eat_predation_ratio * 5 # default 20
                self.food.reset()
            agent._accumulate_rewards += rews[i]
                # print(f"{agent.name} eats food!")
        return rews
        
    # def observation_dict(self):
    #     """
    #     returns a dict of observations for all agents
    #     each observation is a flattened array of relative positions of [food_pos, predator_pos, predator_vel, the_other_agent, the_other_agent_vel]
    #     """
    #     obs = {}
    #     for i, agent in enumerate(self.agents):
    #         obs[agent.name] = np.concatenate([self.food.pos - agent.pos, self.predator.pos - agent.pos, self.predator.vel, self.agents[1-i].pos - agent.pos, self.agents[1-i].vel])
    #         assert obs[agent.name].shape == (10,)
    #     return obs

    def observation_array(self):
        """
        returns an array of observations for all agents
        each observation is a flattened array of relative positions of [food_pos, predator_pos, predator_vel, the_other_agent, the_other_agent_vel]
        """
        return np.array([np.concatenate([self.food.pos - agent.pos, 
                                         self.predator.pos - agent.pos, 
                                         self.predator.vel, 
                                         self.agents[1-i].pos - agent.pos, 
                                         self.agents[1-i].vel,
                                         agent.pos,
                                         np.array([WORLD_WIDTH, WORLD_WIDTH]) - agent.pos]) 
                         for i, agent in enumerate(self.agents)])
        
    def observation_dim(self):
        return 14
    
    def action_dim(self):
        return 5
    
    def _enable_render(self):
        if not self.renderOn and self.render_mode == "human":
            self.screen = pygame.display.set_mode(self.screen.get_size())
            self.renderOn = True
    
    def render(self):
        # if self.render_mode is None:
        #     return 
        self._enable_render()
        self.draw()
        observation = np.array(pygame.surfarray.pixels3d(self.screen))
        if self.render_mode == "human":
            pygame.display.flip() # TODO: ? pyglet?
        return (
            np.transpose(observation, (1, 0, 2))
        )
        
    def draw(self):
        # def draw_agent(agent):
        #     pygame.draw.circle(self.screen, (0, 0, 0), agent.pos, 5)
        
        # Draw background
        self.screen.fill(BACKGROUND_COLOR)        
        # Draw entities
        pygame.draw.circle(
            self.screen, self.food.color, self.food.pos, self.food.radius
        )
        pygame.draw.circle(
            self.screen, self.predator.color, self.predator.pos, self.predator.radius
        )
        for agent in self.agents:
            pygame.draw.circle(self.screen, agent.color, agent.pos, agent.radius)
        # TODO: Add googley eyes to agents
        # TODO: https://www.reddit.com/r/proceduralgeneration/comments/edpbv1/made_a_noise_circle_using_python_and_opensimplex/
        
    def close(self):
        if self.renderOn:
            pygame.event.pump()
            pygame.display.quit()
            self.renderOn = False