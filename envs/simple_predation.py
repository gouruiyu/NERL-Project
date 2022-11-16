from pettingzoo import AECEnv
from pettingzoo.utils import wrappers

def make_env(raw_env):
    def env(**kwargs):
        env = raw_env(**kwargs)
        env = wrappers.AssertOutOfBoundsWrapper(env)
        env = wrappers.OrderEnforcingWrapper(env)
        return env

    return env

class 