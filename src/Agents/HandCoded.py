#!/usr/bin/python3
from Agents.BaseAgent import BaseAgent
from math import pi

# The hand-coded agent chooses an action for 0.2s before reassessing.
class HandCoded(BaseAgent):

	 # Regarding plan duration: 0.2s is an appropriate human reaction time, which I'm also using as a the time it takes for a human to change plans, even
	 # though that's not necessarily the same number.)
	def __init__(self, plan_duration = 0.2):
		tau = 0.02 # The OpenAI episodic cartpole-v1 has tau=0.02s between steps. Properly this should come from the environment.

		self.actions_per_step = max(1, round(plan_duration / tau)) # Number of actions that should be taken before looking at state again, minimum 1.
		self.actions = []

		# The episode ends when the pole is more than 15 degrees from vertical, or the cart moves more than 2.4 units from the center.
		self.fail_angle = 15/180*pi
		self.fail_position = 2.4

		return

	def set_param(self, param):
		return

	"""
	Input: [x, y]
	Return: action
	"""
	def start(self, state):
		self.actions = []
		return self.choose_action(state)

	"""
	Input: int, [x, y]
	Return: action
	"""
	def step(self, reward, state, end_of_ep=False):
		return self.choose_action(state), None

	"""
	Input: int, [x, y]
	Return: None
	"""
	def end(self, reward):
		return


	def choose_action(self, state):
		# Act on a much slower timestep. (During each time period, can act with a specified ratio of left/right actions.)
		# Tile into large tiles. Based on current tile, choose a pre-set action or action series and follow it for a while.
		# Then see which tile I'm in and make a new choice.


		# TODO every run will be the same if we don't have some source of randomness. Maybe add ±0.1 to the target level.

		if len(self.actions) == 0:
			self.select_actions(state)

		action = self.actions.pop(0)
		# print('\t James:', state, action)
		return action


	# create_action_series creates a predetermined series of actions for the next actions_per_step steps.
	# `level` should be a number between 0 and 1. It's the average action value for this time period.
	def create_action_series(self, level):
		# We expect after `actions_per_step` steps, the sum of actions should be `level*actions_per_step`.
		# So at each step, we decide which action will keep the average level closest to `level`.
		sm = 0
		for x in range(1, self.actions_per_step+1):
			target_sum = level*x # By this time, the sum should be as close as possible to this value.

			# If the current sum is within 0.5 of the target, action is 0. Otherwise, the sum is too low and we need to increase it.
			if sm + 0.5 < target_sum:
				next_action = 1
			else:
				next_action = 0

			# print('\t\t', x, target_sum, sm, next_action)
			sm += next_action
			self.actions.append(next_action)


	# Same as create_action_series, but the input ranges from -1 to 1.
	def scaled_create_action_series(self, scaled_level):
		self.create_action_series((scaled_level+1)/2)


	# select_actions chooses the next action series based on the current state.
	def select_actions(self, state):
		position, accel, angle, ang_accel = state

		# This code will try to keep the angle balanced, but ignores the position condition.
		# I think it still usually fails to keep the pole up for more than 2–3s.

		# Respond in proportion to how far we've tilted
		if abs(angle) > 0.9*self.fail_angle:
			# Just do a maximum movement in the same direction
			self.scaled_create_action_series(angle/abs(angle))
		else:
			# Just do a proportional movement in the same direction
			self.scaled_create_action_series(angle/self.fail_angle)

	def save(self, filename):
		raise UserWarning("Hand Coded agent does not save Q")


def init_agent():
	agent = HandCoded()
	return agent
