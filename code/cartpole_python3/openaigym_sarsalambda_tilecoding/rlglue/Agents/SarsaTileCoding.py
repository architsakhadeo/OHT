from Agents.BaseAgent import BaseAgent
import matplotlib
import matplotlib.pyplot as plt
import argparse
import numpy as np
import os
import sys
import pickle
import tiles3 as tc
import math


class SarsaTileCoding(BaseAgent):
	def __init__(self, total_episodes, lmbda, epsilon_init, seed):
		self.number_tilings = 32
		self.dim = 3
		self.ihtsize = self.number_tilings*self.dim**4
		self.gamma = 1.0
		self.lmbda = lmbda
		self.epsilon_init = epsilon_init
		self.stepsize = 0.1/self.number_tilings
		self.w = np.zeros(2*self.ihtsize)
		self.z = np.random.random(2*self.ihtsize)
		self.iht = tc.IHT(self.ihtsize)
		self.delta = 0
		self.state_0 = []
		self.state_1 = []
		self.action_0 = 0
		self.action_1 = 0
		self.obs_0 = []
		self.obs_1 = []
		self.num_episodes = 1
		self.total_episodes = total_episodes
		self.seed = seed
		np.random.seed(self.seed)
	
	def start(self, observation):
		self.obs_0 = observation
		self.z = np.zeros(2*self.ihtsize)
		self.state_0 = self.tilecoding(self.obs_0)
		self.action_0 = self.policy(self.state_0)
		return self.action_0

		
	def step(self, reward, observation, complete):
		#1self.state_0 = self.tilecoding(self.obs_0)
		self.obs_1 = observation
		self.state_1 = self.tilecoding(self.obs_1)
		self.delta = reward
		
		for i in self.F(self.state_0, self.action_0):
			self.delta = self.delta - self.w[i]
			self.z[i] = 1

		if complete == True:	
			self.w = self.w + self.stepsize * self.delta * self.z
			self.num_episodes += 1
			return
		
		self.action_1 = self.policy(self.state_1)
		for i in self.F(self.state_1, self.action_1):
			self.delta = self.delta + self.gamma * self.w[i]
		
		self.w = self.w + self.stepsize * self.delta * self.z
		self.z = self.gamma * self.lmbda * self.z
		#1self.obs_0 = self.obs_1
		self.state_0 = self.state_1
		self.action_0 = self.action_1
		return self.action_0
			
	
	
	def tilecoding(self, observation):
		totaltiles = tc.tiles(self.iht, self.number_tilings, [self.dim*(observation[0])/4.8, self.dim*(observation[1])/8, self.dim*(observation[2])/(2*12 * 2 * math.pi / 360), self.dim*(observation[3]) / 7.0])
		state = [0 for i in range(self.ihtsize)]
		for i in range(len(totaltiles)):
			state[totaltiles[i]]=1
		return state

	
	def policy(self, state):
		a = self.actionvalue(state, 0)
		b = self.actionvalue(state, 1)
		values = [0, 1]
		eps = self.epsilon_init + (1.0 - self.epsilon_init) * self.num_episodes * 1.0 / self.total_episodes
		#eps = 0.1
		if a < b:
			epsilon = [eps/2, 1 - eps + eps/2]
		else:
			epsilon = [1 - eps + eps/2, eps/2]
	
		return int(np.random.choice(values, 1, p=epsilon)[0])

	
	def actionvalue(self, state, action):
		if action == 0:
			features = np.concatenate((state, np.zeros(len(state))))
		if action == 1:
			features = np.concatenate((np.zeros(len(state)),state))
		return np.dot(features, self.w)
	
	
	def F(self, state, action):
		if action == 0:
			features = np.concatenate((state, np.zeros(len(state))))
		if action == 1:
			features = np.concatenate((np.zeros(len(state)),state))
		return [i for i in range(len(features)) if features[i] == 1]
	
	#def end(self, reward):
		#pass
