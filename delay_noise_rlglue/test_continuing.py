from Environments.ContinuingCartpoleEnvironment import CartpoleEnvironmentContinuing
from Agents.SarsaTileCodingContinuing import SarsaTileCodingContinuing
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
import time

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
		obs = self.environment.start()
		self.last_action = self.agent.start(obs)
		return (obs, self.last_action)

	def step(self):
		(obs, reward, term) = self.environment.step(self.last_action)
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


save_dir = 'Data/'
if not os.path.exists(save_dir):
	os.makedirs(save_dir)


lv = 0.8
run = 1
ev = 0.05
gm = 0.99
total_number_steps = 1000
'''
for run in range(runs):
	for ev in epsilon_init_values:
		for lv in lmbda_values:
			agent = SarsaTileCodingContinuing(gmma = gm, lmbda = lv, epsilon_init = ev, seed = run)
			environment = CartpoleEnvironmentContinuing(seed = run)
			experiment = test(agent, environment)
			experiment.start()

			while experiment.num_steps+1 <= total_number_steps:
				experiment.step()
				#print(experiment.num_steps)
'''

start = time.time()
agent = SarsaTileCodingContinuing(gmma = gm, lmbda = lv, epsilon_init = ev, seed = run)
environment = CartpoleEnvironmentContinuing(seed = run)
experiment = test(agent, environment)
experiment.start()

while experiment.num_steps+1 <= total_number_steps:
	experiment.step()
end = time.time()

print(end-start)


'''				
returns = experiment.rewards[:]
for i in range(len(returns)-2,-1,-1):
	if returns[i] == -1:
		continue
	returns[i] = gm * returns[i+1]
#print(returns)
print(sum(returns))
'''
#pickle.dump([experiment.reset_step_x, experiment.reset_step_y], open(save_dir+'L_'+str(int(lv*10))+'_run'+ str(run)+'.pkl', 'wb'))
plt.plot(experiment.reset_step_x, experiment.reset_step_y) # X axis = total time, Y axis = total failures
#plt.plot(experiment.reset_step_x, returns)
plt.show()
print('Done')