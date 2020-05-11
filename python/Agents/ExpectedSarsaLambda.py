from Agents.BaseAgent import BaseAgent
import matplotlib
import matplotlib.pyplot as plt
import argparse
import numpy as np
import os
import sys
import pickle
import utils.tiles3 as tc
import math

# Not using wrap around tilings for angle and angular velocity
# as they are restricted by the environment to some thresholds

# We want wide tiles (small int(dim) value) and many tilings
# Many tilings results in slower but better learning

class ExpectedSarsaTileCodingContinuing(BaseAgent):
	def __init__(self):
		super().__init__()
		return

	def setDefaults(self, param):
		if 'num_tilings' not in param:
			param['num_tilings'] = 32

		if 'num_tiles' not in param:
			param['num_tiles'] = 4

		if 'gamma' not in param:
			param['gamma'] = 0.95

		if 'lmbda' not in param:
			param['lmbda'] = 0.8

		if 'epsilon' not in param:
			param['epsilon'] = 0.0001

		if 'alpha' not in param:
			param['alpha'] = 0.1

		return param

	def set_param(self, param):
		param = self.setDefaults(param)

		self.number_tilings = param['num_tilings']
		self.dim = param['num_tiles']
		self.gamma = param['gamma']
		self.lmbda = param['lmbda']
		self.epsilon_init = param['epsilon']

		self.stepsize = param['alpha'] / self.number_tilings

		self.lows = [-2.4, -4.0, -(12 * 2 * math.pi / 360), -3.5]
		self.highs = [2.4, 4.0, (12 * 2 * math.pi / 360), 3.5]
		self.range = [2.0* 2.4, 2.0* 4.0, 2.0* 12 * 2 * math.pi / 360, 2.0* 3.5]		
		self.lows = np.array([-2.4, -4.0, -(12 * 2 * math.pi / 360), -3.5])
		self.highs = np.array([2.4, 4.0, (12 * 2 * math.pi / 360), 3.5])
		self.range = np.array([2.0* 2.4, 2.0* 4.0, 2.0* 12 * 2 * math.pi / 360, 2.0* 3.5])
		self.tc_normalizer = np.array([self.dim / self.range[i] for i in range(len(self.range))])
		
		# Tiling all dimensions separately

		self.ihtsize_ind = int(self.number_tilings * (self.dim + 1))
		self.ihtsize_pair = int(self.number_tilings * (self.dim + 1) ** 2)
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

		self.w = np.zeros(2 * (self.ihtsize_ind * 4 + self.ihtsize_pair * 6))  # separate and pairs
		self.z = np.zeros(2 * (self.ihtsize_ind * 4 + self.ihtsize_pair * 6))  # separate and pairs

		# Adaptive stepsizes parameters
		self.adaptive_stepsize = 0.1/32
		self.beta1 = 0.9
		self.beta2 = 0.999
		self.e = 10**-8
		self.m = np.zeros(2*(self.ihtsize_ind*4 + self.ihtsize_pair*6))
		self.v = np.zeros(2*(self.ihtsize_ind*4 + self.ihtsize_pair*6))

		self.delta = 0
		self.state_0 = []
		self.state_1 = []
		self.action = 0
		self.action_1 = 0
		self.obs_0 = []
		self.obs_1 = []

	def start(self, observation):
		self.obs_0 = observation
		self.timesteps = 0
		self.z = np.zeros(2*(self.ihtsize_ind*4 + self.ihtsize_pair*6)) #Tiling dimensions separately and in pairs
		self.state_0 = self.tilecoding(self.obs_0)

		activef_s0_0 = self.activefeatures(self.state_0, 0)
		activef_s0_1 = self.activefeatures(self.state_0, 1)

		self.action, a0, eps_a0, a1, eps_a1 = self.policy_expected_sarsa(activef_s0_0, activef_s0_1)
		return self.action

		
	def step(self, reward, observation, complete=False):
		self.timesteps += 1

		self.obs_1 = [0 for i in range(len(observation))] 
		for i in range(len(observation)):
			if observation[i] >= self.lows[i] and observation[i] <= self.highs[i]:
				self.obs_1[i] = observation[i]
			elif observation[i] > self.highs[i]:
				self.obs_1[i] = self.highs[i]
			else:
				self.obs_1[i] = self.lows[i]

		self.state_1 = self.tilecoding(self.obs_1)
		self.delta = reward

		activef_s0_action = self.activefeatures(self.state_0, self.action)

		for i in activef_s0_action:
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
		
		activef_s1_0 = self.activefeatures(self.state_1, 0)
		activef_s1_1 = self.activefeatures(self.state_1, 1)
		activef_s1 = [activef_s1_0, activef_s1_1]


		self.action_1, a0, eps_a0, a1, eps_a1 = self.policy_expected_sarsa(activef_s1_0, activef_s1_1)
		actions = [a0, a1]
		probs = [eps_a0, eps_a1]

		for j in range(len(actions)):
			for i in activef_s1[j]:
				self.delta = self.delta + self.gamma * probs[j] * self.w[i]
		
		'''
		#Adam
		g = -self.delta*self.z
		self.m = self.beta1*self.m + (1-self.beta1)*g
		self.v = self.beta2*self.v + (1-self.beta2)*g*g
		mhat = self.m / (1 - self.beta1 ** self.timesteps)
		vhat = self.v / (1 - self.beta2 ** self.timesteps)
		self.w = self.w - self.adaptive_stepsize * self.m / ((self.v**0.5) + self.e)
		'''


		'''
		#RMSProp
		g = -self.delta*self.z
		self.v = self.beta1*self.v + (1-self.beta1)*g*g
		self.w = self.w - self.adaptive_stepsize*g/((self.v+self.e)**0.5)
		'''

		'''
		#AdaGrad
		g = -self.delta*self.z
		self.v = self.v + g*g
		self.w = self.w - self.adaptive_stepsize * g /((self.v + self.e)**0.5)
		'''

		'''
		#Momentum
		g = -self.delta*self.z
		self.m = self.beta1*self.m + (1-self.beta1)*g
		self.w = self.w - self.adaptive_stepsize*self.m
		'''

		#SemiGD
		zindices = np.where(self.z != 0)[0]
		activez = self.z[zindices]
		g = -self.delta*activez
		self.w[zindices] = self.w[zindices] - self.stepsize*g
		
		self.z[zindices] = self.gamma * self.lmbda * activez
		self.state_0 = self.state_1
		self.action = self.action_1
		return self.action, {}


	def end(self, reward):
		return


	def tilecoding(self, observation):
		#Tiling all dimensions separately and in pairs
		
		state = np.array([])
		tiles0 = np.array( tc.tiles(self.iht0, self.number_tilings, [ observation[0]*self.tc_normalizer[0] ]) )
		tiles1 = np.array( tc.tiles(self.iht1, self.number_tilings, [ observation[1]*self.tc_normalizer[1] ]) )
		tiles2 = np.array( tc.tiles(self.iht2, self.number_tilings, [ observation[2]*self.tc_normalizer[2] ]) )
		tiles3 = np.array( tc.tiles(self.iht3, self.number_tilings, [ observation[3]*self.tc_normalizer[3] ]) )
		tiles01 = np.array( tc.tiles(self.iht01, self.number_tilings, [ observation[0]*self.tc_normalizer[0], observation[1]*self.tc_normalizer[1] ]) )
		tiles02 = np.array( tc.tiles(self.iht02, self.number_tilings, [ observation[0]*self.tc_normalizer[0], observation[2]*self.tc_normalizer[2] ]) )
		tiles03 = np.array( tc.tiles(self.iht03, self.number_tilings, [ observation[0]*self.tc_normalizer[0], observation[3]*self.tc_normalizer[3] ]) )
		tiles12 = np.array( tc.tiles(self.iht12, self.number_tilings, [ observation[1]*self.tc_normalizer[1], observation[2]*self.tc_normalizer[2] ]) )
		tiles13 = np.array( tc.tiles(self.iht13, self.number_tilings, [ observation[1]*self.tc_normalizer[1], observation[3]*self.tc_normalizer[3] ]) )
		tiles23 = np.array( tc.tiles(self.iht23, self.number_tilings, [ observation[2]*self.tc_normalizer[2], observation[3]*self.tc_normalizer[3] ]) )
		totaltiles_ind = np.array([tiles0, tiles1, tiles2, tiles3])
		totaltiles_pair = np.array([tiles01, tiles02, tiles03, tiles12, tiles13, tiles23])
		for j in range(len(totaltiles_ind)):
			temp = np.zeros(self.ihtsize_ind)
			for i in range(len(totaltiles_ind[j])):
				temp[totaltiles_ind[j][i]]=1
			state = np.concatenate((state, temp))

		for j in range(len(totaltiles_pair)):
			temp = np.zeros(self.ihtsize_pair)
			for i in range(len(totaltiles_pair[j])):
				temp[totaltiles_pair[j][i]]=1
			state = np.concatenate((state, temp))
			
		return state


	def policy_expected_sarsa(self, activef_s_a0, activef_s_a1):
		a = self.actionvalue(activef_s_a0)
		b = self.actionvalue(activef_s_a1)
		actions = [0, 1]
		eps = self.epsilon_init
		#Epsilon greedy
		if a < b:
			epsilon = [eps/2, 1 - eps + eps/2]
		else:
			epsilon = [1 - eps + eps/2, eps/2]
		return [int(np.random.choice(actions, 1, p=epsilon)[0]), 0, epsilon[0], 1, epsilon[1]] 

		'''
		#Softmax
		#print(a,b, np.exp(a), np.exp(b))
		#a_exp = np.exp((a))
		#b_exp = np.exp((b))
		
		a_exp = np.exp((a - max(a,b)))
		b_exp = np.exp((b - max(a,b)))
		#print(a,b, np.exp(a), np.exp(b))

		total_exp = (a_exp + b_exp)
		softmax_prob = [a_exp / total_exp, b_exp / total_exp]
		#print(softmax_prob)
		return [int(np.random.choice(actions, 1, p=softmax_prob)[0]), 0, softmax_prob[0], 1, softmax_prob[1]]
		'''
	
	def actionvalue(self, activefeatures):
		return np.sum(self.w[activefeatures])
	
	def activefeatures(self, state, action):
		if action == 0:
			features = np.concatenate((state, np.zeros(len(state))))
		if action == 1:
			features = np.concatenate((np.zeros(len(state)),state))
		return np.where(features == 1)[0]
	
	def save(self, path):
		np.save(path+'.npy',self.w)
	
	def load(self, path):
		self.w = np.load(path + '.npy')


def init_agent():
	return ExpectedSarsaTileCodingContinuing()