from pettingzoo.utils.env import ParallelEnv
from gymnasium import spaces
import numpy as np
import pygame
import torch


WOLRD_WIDTH = 600
FRICTION_DECAY = 0.9
BACKGROUND_COLOR = pygame.Color(242, 215, 213)
FOOD_COLOR = pygame.Color(159, 226, 191) #9FE2BF
PREDATOR_COLOR = pygame.Color(255, 127, 80) #FF7F50
AGENT_COLOR = pygame.Color(100, 149, 237) # 6495ED
BUFFER_SPACE = 5.

PREDATOR_POS_ORIGINAL = np.array([300., 300.])
FOOD_POS_ORIGINAL = np.array([300., 100.])
AGENTS_POS_ORIGINAL = np.array([[100., 500.], [500., 500.]])


class Entity():
    def __init__(self, name=None, pos=None, color=None, radius=0.):
        self.name = name
        self.pos_original = pos
        # pos_noise = np.random.uniform(-BUFFER_SPACE, BUFFER_SPACE, size=2)
        # self.pos = pos + pos_noise
        self.pos = pos
        self.color = color
        self.radius = radius
        
    def _bounce(self):
        if self.pos[0] < self.radius:
            self.pos[0] = self.radius
            self.vel[0] *= -0.5
        if self.pos[0] > WOLRD_WIDTH - self.radius:
            self.pos[0] = WOLRD_WIDTH - self.radius
            self.vel[0] *= -0.5
        if self.pos[1] < self.radius:
            self.pos[1] = self.radius
            self.vel[1] *= -0.5
        if self.pos[1] > WOLRD_WIDTH - self.radius:
            self.pos[1] = WOLRD_WIDTH - self.radius
            self.vel[1] *= -0.5

class Food(Entity):
    def __init__(self, name="food", pos=FOOD_POS_ORIGINAL, color=FOOD_COLOR, radius=10.):
        super().__init__(name, pos, color, radius)

class Agent(Entity):
    def __init__(self, name, pos=AGENTS_POS_ORIGINAL[0], color=AGENT_COLOR, radius=15.):
        super().__init__(name, pos, color, radius)
        self.accel = 4.
        self.vel = np.zeros(2)
        
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
    def __init__(self, name="predator", pos=PREDATOR_POS_ORIGINAL, color=PREDATOR_COLOR, radius=20.):
        super().__init__(name, pos, color, radius)
        self.accel = 5.
        self.vel = np.zeros(2)
    
    def update(self, preys):
        """
        Given array of preys, predator follows simple heuristic to chase the nearest one
        # TODO: optimize to use distance matrix
        """
        rel_positions = np.array([prey.pos for prey in preys]) - self.pos
        closest = np.argmin(np.linalg.norm(rel_positions, axis=1))
        print(f"Predator: I am chasing {preys[closest].name}!")
        # apply accelration in the direction of the closest prey
        self.vel *= FRICTION_DECAY
        self.vel += self.accel * rel_positions[closest] / np.linalg.norm(rel_positions[closest])
        self.pos += self.vel
        self._bounce()
        return


class PreyArena(ParallelEnv):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }
    
    def __init__(
        self,
        max_cycles=100000,
        render_mode=None,
        agent_names=["R2D2", "C-3PO"]
        ) -> None:
        super().__init__()
        
        # rendering options
        self.render_mode = render_mode
        pygame.init()
        self.viewer = None
        self.width = WOLRD_WIDTH
        self.height = WOLRD_WIDTH
        self.screen = pygame.Surface((self.width, self.height))
        self.renderOn = False
        
        # Assumed deterministic envivornment for now: 
        #  Stochasticity sources: food respawn location
        
        self.max_cycles = max_cycles
        self.agent_names = agent_names
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
        return self.dm[idx1, idx2] < (self.radius[idx1] + self.radius[idx2])
        
    def reset(self, seed=None, return_info=False, options=None):
        # reset entities' location
        np.random.seed(seed)
        self.predator = Predator()
        self.food = Food()
        self.agents = [Agent(name, AGENTS_POS_ORIGINAL[i]) for i, name in enumerate(self.agent_names)]
        self.entities = self.agents + [self.food, self.predator]
        return
    
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
        if self.render_mode == "human":
            self.render()
        
        
    def observation(self):
        """
        returns a dict of observations for all agents
        each observation is a flattened array of relative positions of [food_pos, predator_pos, predator_vel, the_other_agent, the_other_agent_vel]
        """
        obs = {}
        for i, agent in enumerate(self.agents):
            obs[agent.name] = np.concatenate([self.food.pos - agent.pos, self.predator.pos - agent.pos, self.predator.vel, self.agents[1-i].pos - agent.pos, self.agents[1-i].vel])
            assert obs[agent.name].shape == (10,)
        return obs
    
    def _enable_render(self, mode="human"):
        if not self.renderOn and mode == "human":
            self.screen = pygame.display.set_mode(self.screen.get_size())
            self.renderOn = True
    
    def render(self):
        self._enable_render(self.render_mode)
        observation = np.array(pygame.surfarray.pixels3d(self.screen))
        if self.render_mode == "human":
            self.draw()
            pygame.display.flip() # TODO: ? pyglet?
        return (
            np.transpose(observation, (1, 0, 2))
            if self.render_mode == "rgb_array"
            else None
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