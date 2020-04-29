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
        self.noise_fn = None

    def __getattr__(self, name):
        return getattr(self.env, name)

    def set_param(self, param):
        self.noise = np.zeros(shape=self.state_dim())
        self.sensor_steps = 0
        self.sensor_life = np.array(param.sensor_life)
        self.noise_std = self.state_max / param.drift_scale
        if param.drift_prob < 0:
            self.noise_fn = self.gauss_noise
        else:
            self.noise_fn = self.prob_gauss_noise
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
    
    def gauss_noise(self):
        # Zero-mean gaussian noise applied at every time-step.
        return np.random.normal(loc=np.zeros(shape=self.state_dim()), scale=self.noise_std)

    def prob_gauss_noise(self):
        self.sensor_steps += 1

        # The probability of drift at each timestep follows a scaled logistic function.
        # Each element corresponds to the sensor of a component of the state.
        prob_drift = logistic.cdf(self.sensor_steps, loc=self.sensor_life/2,
                                  scale=self.sensor_life/10)*self.drift_prob
        is_drift = bernoulli.rvs(p=prob_drift)
        
        # TODO Enable increasing mean and/or variance.
        noise_new = np.multiply(
            np.random.normal(loc=np.zeros(shape=self.state_dim()), scale=self.noise_std),
            is_drift)
        return noise_new

    def state_process(self, state):
        noise_new = self.noise_fn()
        self.noise = np.clip(self.noise+noise_new, -self.noise_max, self.noise_max)
        state = np.clip(state+self.noise, -self.state_max, self.state_max)
        return state