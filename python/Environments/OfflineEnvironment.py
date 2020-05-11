from Environments.BaseEnvironment import BaseEnvironment
import numpy as np
import math

class OfflineEnv(BaseEnvironment):

    def __init__(self):
        self.s_dim = 4
        return

    def set_param(self, param):
        self.offline_env = param.offline_env_model["offline_env"] # a list of trees
        self.offline_data = param.offline_env_model["offline_data"] # a list of data
        return

    def start(self):
        self.total_steps = 0
        self.state = np.random.uniform(low=-0.05, high=0.05, size=(4,))
        self.steps_beyond_done = None
        return self.state

    def step(self, action):
        dist, idx = self.offline_env[action].query([self.state], k=1)
        seq = self.offline_data[action][idx[0][0]]
        self.state, reward, done = seq[self.s_dim+1: self.s_dim*2+1], seq[self.s_dim*2+1], seq[self.s_dim*2+2]
        return self.state, reward, done, {}

    def num_action(self):
        return 2

    def state_dim(self):
        return 4

    def state_range(self):
        return [4.8, 8.0, 2 * 12 * 2 * math.pi / 360.0, 7.0]


def init_env():
    return OfflineEnv()
