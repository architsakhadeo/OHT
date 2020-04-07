from Environments.ContinuingCartpoleEnvironment import CartpoleEnvironmentContinuing
from Agents.ExpectedSarsaTileCodingContinuing import ExpectedSarsaTileCodingContinuing
from Experiments.BaseExperiment import BaseExperiment

import numpy as np
import random
import matplotlib
import matplotlib.pyplot as plt
import argparse
import os
import sys
import pickle
import tiles3 as tc
import math

class test(BaseExperiment):
	def __init__(self, agent, env):
		self.environment = env
		self.agent = agent
		self.last_action = None
		#self.total_reward = 0.0
		self.num_steps = 0
		self.num_resets = 0 #equivalent to num_resets_to_start
		self.reset_step_x = []
		self.reset_step_y = []
		self.rewards = []
		#self.total_return = []
		#self.total_steps_accumulated = []
		#self.total_steps_cumulative = []

	def start(self):
		self.num_steps = 0
		self.total_reward = 0
		s = self.environment.start()
		obs = self.observationChannel(s)
		self.last_action = self.agent.start(obs)
		return (obs, self.last_action)

	def step(self):
		(s, reward, term) = self.environment.step(self.last_action)
		obs = self.observationChannel(s)
		#self.total_reward += reward    #episodic
		self.rewards.append(reward)
		self.num_steps += 1
		self.last_action = self.agent.step(reward, obs, term)
		if term:
			self.num_resets += 1
			print('Failure ' + str(self.num_resets) + ', Step ' + str(self.num_steps))
			#self.total_return.append(self.total_reward)    #episodic
		#return term
		self.reset_step_x.append(self.num_steps)
		self.reset_step_y.append(self.num_resets)

	def observationChannel(self, s):
		return s


save_dir = 'Data/'
if not os.path.exists(save_dir):
	os.makedirs(save_dir)


#epsilon_init_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
#lmbda_values        = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
epsilon_init_values = [0.1]
lmbda_values        = [0.8]
#runs = 30
runs = 1
gm = 0.9
total_number_steps = 10000
for run in range(runs):
	for ev in epsilon_init_values:
		for lv in lmbda_values:
			agent = ExpectedSarsaTileCodingContinuing(gmma = gm, lmbda = lv, epsilon_init = ev, seed = run)
			environment = CartpoleEnvironmentContinuing(seed = run)
			experiment = test(agent, environment)
			experiment.start()

			while experiment.num_steps+1 <= total_number_steps:
				experiment.step()
				#print(experiment.num_steps)

'''				
returns = experiment.rewards[:]
for i in range(len(returns)-2,-1,-1):
	if returns[i] == -1:
		continue
	returns[i] = gm * returns[i+1]
#print(returns)
print(sum(returns))
'''
plt.plot(experiment.reset_step_x, experiment.reset_step_y) # X axis = total time, Y axis = total failures
#plt.plot(experiment.reset_step_x, returns)
plt.show()
