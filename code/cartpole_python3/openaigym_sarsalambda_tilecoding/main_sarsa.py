#!/usr/bin/env python3

import gym
import matplotlib
import matplotlib.pyplot as plt
import argparse
import numpy as np
import os
import sys
import pickle
import tiles3 as tc
import math

# prevents type-3 fonts, which some conferences disallow.
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

number_tilings = 32
dim = 3
ihtsize = number_tilings*dim**4

def make_env():
	env = gym.make('CartPole-v0')
	return env


def tilecoding(iht, observation):
	#totaltiles = tc.tiles(iht, number_tilings, [dim*(observation[0]+2.4)/4.8, dim*(observation[1]+4)/8, dim*(observation[2]+(12 * 2 * math.pi / 360))/(2*12 * 2 * math.pi / 360), dim*(observation[3] + 3.5) / 7.0])
	totaltiles = tc.tiles(iht, number_tilings, [dim*(observation[0])/4.8, dim*(observation[1])/8, dim*(observation[2])/(2*12 * 2 * math.pi / 360), dim*(observation[3]) / 7.0])
	state = [0 for i in range(ihtsize)]
	for i in range(len(totaltiles)):
		state[totaltiles[i]]=1
	return state

	
def policy(state, w):
	a = actionvalue(state, 0, w)
	b = actionvalue(state, 1, w)
	values = [0, 1]
	if a < b:
		epsilon = [0.1, 0.9]
	else:
		epsilon = [0.9, 0.1]
	
	return int(np.random.choice(values, 1, p=epsilon)[0])


def actionvalue(state, action, w):
	if action == 0:
		features = np.concatenate((state, np.zeros(len(state))))
	if action == 1:
		features = np.concatenate((np.zeros(len(state)),state))
	return np.dot(features, w)


def F(state, action):
	if action == 0:
		features = np.concatenate((state, np.zeros(len(state))))
	if action == 1:
		features = np.concatenate((np.zeros(len(state)),state))
	return [i for i in range(len(features)) if features[i] == 1]

def sarsa(env, num_episodes, batch_size=1, gamma=1.0, lmbda=0.9):
				  
	stepsize = 0.1/number_tilings
	w = np.zeros(2*ihtsize)
	z = np.random.random(2*ihtsize)
	
	iht = tc.IHT(ihtsize)
	total_returns = []
	for ep in range(num_episodes):
		returns = 0
		obs_0 = env.reset()
		z = np.zeros(2*ihtsize)
		complete = False
		
		while complete == False:
			state_0 = tilecoding(iht, obs_0)

			action_0 = policy(state_0, w)
			obs_1, r, complete, _ = env.step(action_0)
			returns += r
			
			state_1 = tilecoding(iht, obs_1)
			delta = r
			
			for i in F(state_0,action_0):
				delta = delta - w[i]
				z[i] = 1
			
			if complete == True:	
				w = w + stepsize * delta * z
				print(ep, returns)
				total_returns.append(returns)
				continue
			
			action_1 = policy(state_1, w)
			for i in F(state_1, action_1):
				delta = delta + gamma * w[i]
			
			w = w + stepsize * delta * z
			z = gamma * lmbda * z
			obs_0 = obs_1
			action_0 = action_1
		
		
if __name__ == '__main__':

	"""
	python main.py --episodes ihtsize
	"""

	parser = argparse.ArgumentParser()
	parser.add_argument("--episodes", "-e", default=50000, type=int, help="Number of episodes to train for")
	args = parser.parse_args()

	episodes = args.episodes
	numrun = 1
	
	for run in range(numrun):
		env = make_env()
		in_size = env.observation_space.shape[0]
		num_actions = env.action_space.n
		sarsa(env, episodes)

