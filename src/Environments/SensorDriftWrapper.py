from Environments.BaseEnvironment import BaseEnvironment
import numpy as np
from scipy.stats import bernoulli, logistic

class SensorDriftWrapper(BaseEnvironment):
    def __init__(self, env):
        self.env = env
        self.num_action = self.env.num_action
        self.state_dim = self.env.state_dim
        self.state_range = self.env.state_range
        self.state_max = np.array(self.state_range())
        self.noise = np.zeros(shape=self.state_dim())
        self.noise_std = self.state_max / 100
        self.noise_max = self.state_max
        self.drift_prob = 0

    def __getattr__(self, name):
        return getattr(self.env, name)

    def set_param(self, param):
        self.noise = np.zeros(shape=self.state_dim())
        self.sensor_steps = 0
        self.sensor_life = param.sensor_life
        self.noise_std = self.state_max / param.drift_scale
        self.drift_prob = param.drift_prob
        return self.env.set_param(param)

    def reset(self):
        self.noise = np.zeros(shape=self.state_dim())
        self.sensor_steps = 0

    def start(self):
        return self.env.start()

    def step(self, action):
        state, reward, done = self.env.step(action)
        return self.state_process(state), reward, done

    def state_process(self, state):
        self.sensor_steps += 1

        # The probability of drift at each timestep follows a scaled logistic function.
        prob_drift = logistic.cdf(self.sensor_steps, loc=self.sensor_life/2,
                                  scale=self.sensor_life/10)*self.drift_prob
        is_drift = bernoulli.rvs(p=prob_drift)
        
        # TODO Enable increasing mean and/or variance.
        noise_new = (np.random.normal(loc=np.zeros(shape=self.state_dim()),
                        scale=self.noise_std) if is_drift
                        else np.zeros(shape=self.state_dim()))
        self.noise = np.clip(self.noise+noise_new, -self.noise_max, self.noise_max)
        state = np.clip(state+self.noise, -self.state_max, self.state_max)
        return state