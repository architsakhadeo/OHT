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

# Not using wrap around tilings for angle and angular velocity
# as they are restricted by the environment to some thresholds

# We want wide tiles (small int(dim) value) and many tilings
# Many tilings results in slower but better learning

class ExpectedSarsaTileCodingContinuing(BaseAgent):
	def __init__(self, gmma, lmbda, epsilon_init, seed):
		self.number_tilings = 32
		self.dim = 3
		self.gamma = gmma
		self.lmbda = lmbda
		self.epsilon_init = epsilon_init
		self.stepsize = 0.1/self.number_tilings
		#Tiling all dimensions together
		'''
		self.ihtsize = self.number_tilings*(self.dim+1)**4
		self.iht = tc.IHT(self.ihtsize)
		self.w = np.zeros(2*self.ihtsize)
		self.z = np.random.random(2*self.ihtsize)		
		'''
		#Tiling all dimensions separately
		
		
		#self.ihtsize = int(self.number_tilings*(self.dim**4)/4)
		self.ihtsize_ind = int(self.number_tilings*(self.dim+1))
		self.ihtsize_pair = int(self.number_tilings*(self.dim+1)**2)
		self.iht0 = tc.IHT(self.ihtsize_ind)
		self.iht1 = tc.IHT(self.ihtsize_ind)
		self.iht2 = tc.IHT(self.ihtsize_ind)
		self.iht3 = tc.IHT(self.ihtsize_ind)

		# Tiling dimensions in pairs
		self.iht01 = tc.IHT(self.ihtsize_pair)
		self.iht02 = tc.IHT(self.ihtsize_pair)
		self.iht03 = tc.IHT(self.ihtsize_pair)
		self.iht12 = tc.IHT(self.ihtsize_pair)
		self.iht13 = tc.IHT(self.ihtsize_pair)
		self.iht23 = tc.IHT(self.ihtsize_pair)

		#self.w = np.zeros(2*self.ihtsize_ind*4) #separate 
		#self.z = np.random.random(2*self.ihtsize_ind*4) #separate


		self.w = np.zeros(2*(self.ihtsize_ind*4 + self.ihtsize_pair*6)) #separate and pairs 
		self.z = np.random.random(2*(self.ihtsize_ind*4 + self.ihtsize_pair*6)) #separate and pairs
		
		
		self.delta = 0
		self.state_0 = []
		self.state_1 = []
		self.action_0 = 0
		self.action_1 = 0
		self.obs_0 = []
		self.obs_1 = []
		#self.num_episodes = 1
		#self.total_episodes = total_episodes
		self.seed = seed
		np.random.seed(self.seed)
	
	def start(self, observation):
		self.obs_0 = observation
		#self.z = np.zeros(2*self.ihtsize) #Tiling dimensions together
		#self.z = np.zeros(2*self.ihtsize*4) #Tiling dimensions separately
		self.z = np.zeros(2*(self.ihtsize_ind*4 + self.ihtsize_pair*6)) #Tiling dimensions separately and in pairs
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
			self.z[i] = 1 #replacing
			#self.z[i] += 1 #accumulating

		
        #SARSA
		'''
		self.action_1 = self.policy(self.state_1)
		for i in self.F(self.state_1, self.action_1):
			self.delta = self.delta + self.gamma * self.w[i]
		'''
		#Expected SARSA
		
		self.action_1, a0, eps_a0, a1, eps_a1 = self.policy_expected_sarsa(self.state_1)
		actions = [a0, a1]
		probs = [eps_a0, eps_a1]
		for j in range(len(actions)):
			for i in self.F(self.state_1, actions[j]):
				self.delta = self.delta + self.gamma * probs[j] * self.w[i]
		
		self.w = self.w + self.stepsize * self.delta * self.z
		self.z = self.gamma * self.lmbda * self.z
		#1self.obs_0 = self.obs_1
		self.state_0 = self.state_1
		self.action_0 = self.action_1
		return self.action_0
			
	
	
	def tilecoding(self, observation):
		#Tiling all dimensions together
		'''
		totaltiles = tc.tiles(self.iht, self.number_tilings, [self.dim*(observation[0])/4.8, self.dim*(observation[1])/8, self.dim*(observation[2])/(2*12 * 2 * math.pi / 360), self.dim*(observation[3]) / 7.0])
		state = [0 for i in range(self.ihtsize)]
		for i in range(len(totaltiles)):
			state[totaltiles[i]]=1
		'''

		#Tiling all dimensions separately and in pairs
		
		state = []
		tiles0 = tc.tiles(self.iht0, self.number_tilings, [self.dim*(observation[0])/4.8])
		tiles1 = tc.tiles(self.iht1, self.number_tilings, [self.dim*(observation[1])/8.0])
		tiles2 = tc.tiles(self.iht2, self.number_tilings, [self.dim*(observation[2])/(2*12 * 2 * math.pi / 360)])
		tiles3 = tc.tiles(self.iht3, self.number_tilings, [self.dim*(observation[3])/7.0])
		tiles01 = tc.tiles(self.iht01, self.number_tilings, [self.dim*(observation[0])/4.8, self.dim*(observation[1])/8.0])
		tiles02 = tc.tiles(self.iht02, self.number_tilings, [self.dim*(observation[0])/4.8, self.dim*(observation[2])/(2*12 * 2 * math.pi / 360)])
		tiles03 = tc.tiles(self.iht03, self.number_tilings, [self.dim*(observation[0])/4.8, self.dim*(observation[3])/7.0])
		tiles12 = tc.tiles(self.iht12, self.number_tilings, [self.dim*(observation[1])/8.0, self.dim*(observation[2])/(2*12 * 2 * math.pi / 360)]) 
		tiles13 = tc.tiles(self.iht13, self.number_tilings, [self.dim*(observation[1])/8.0, self.dim*(observation[3])/7.0])
		tiles23 = tc.tiles(self.iht23, self.number_tilings, [self.dim*(observation[2])/(2*12 * 2 * math.pi / 360), self.dim*(observation[3])/7.0])
		totaltiles_ind = [tiles0, tiles1, tiles2, tiles3]
		totaltiles_pair = [tiles01, tiles02, tiles03, tiles12, tiles13, tiles23]
		for j in range(len(totaltiles_ind)):
			temp = [0 for i in range(self.ihtsize_ind)]
			for i in range(len(totaltiles_ind[j])):
				temp[totaltiles_ind[j][i]]=1
			state += temp

		for j in range(len(totaltiles_pair)):
			temp = [0 for i in range(self.ihtsize_pair)]
			for i in range(len(totaltiles_pair[j])):
				temp[totaltiles_pair[j][i]]=1
			state += temp

		return state
		

	def policy(self, state):
		a = self.actionvalue(state, 0)
		b = self.actionvalue(state, 1)
		actions = [0, 1]
		eps = self.epsilon_init
		if a < b:
			epsilon = [eps/2, 1 - eps + eps/2]
		else:
			epsilon = [1 - eps + eps/2, eps/2]
	
		return int(np.random.choice(actions, 1, p=epsilon)[0])

	def policy_expected_sarsa(self, state):
		a = self.actionvalue(state, 0)
		b = self.actionvalue(state, 1)
		actions = [0, 1]
		eps = self.epsilon_init
		if a < b:
			epsilon = [eps/2, 1 - eps + eps/2]
		else:
			epsilon = [1 - eps + eps/2, eps/2]
	
		return [int(np.random.choice(actions, 1, p=epsilon)[0]), 0, epsilon[0], 1, epsilon[1]]  

	
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
	
