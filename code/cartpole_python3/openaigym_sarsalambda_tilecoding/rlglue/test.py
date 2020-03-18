from Environments.CartpoleEnvironment import CartpoleEnvironment
from Agents.SarsaTileCoding import SarsaTileCoding
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
		self.total_reward = 0.0
		self.num_steps = 0
		self.num_episodes = 0
		self.total_return = []
			
	def start(self):
		self.num_steps = 0
		self.total_reward = 0
		s = self.environment.start()
		obs = self.observationChannel(s)
		self.last_action = self.agent.start(obs)
		return (obs, self.last_action)
			
	def step(self):
		(s, reward, term, _) = self.environment.step(self.last_action)
		obs = self.observationChannel(s)
		self.total_reward += reward
		self.num_steps += 1
		self.last_action = self.agent.step(reward, obs, term)
		if term:
			self.num_episodes += 1
			self.total_return.append(self.total_reward)
		return term

	def observationChannel(self, s):
		return s	


total_episodes = 1500
#epsilon_init_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
#lmbda_values        = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
epsilon_init_values = [0.0]
lmbda_values        = [0.0]
#runs = 30
runs = 1
for run in range(runs):
	for ev in epsilon_init_values:
		for lv in lmbda_values:
			agent = SarsaTileCoding(total_episodes, lmbda = lv, epsilon_init = ev, seed = run)
			environment = CartpoleEnvironment(seed = run)
			experiment = test(agent, environment)

			for episode in range(total_episodes):
				experiment.start()
				complete = False
				while complete == False:
					complete = experiment.step()
				print(episode, experiment.total_reward)
			experiment.total_return.append([ev])
			experiment.total_return.append([lv])
			
			pickle.dump(experiment.total_return, open('Data/' + str(100*ev) + '_ev_' + str(100*lv) + '_lv' '.pkl','wb'))
	
