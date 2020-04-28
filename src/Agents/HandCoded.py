#!/usr/bin/python3
from Agents.BaseAgent import BaseAgent
from math import pi

# The hand-coded agent chooses an action for 0.2s before reassessing.
class HandCoded(BaseAgent):

	"""
	Inputs: [x, y]
		plan_duration: Number of actions that should be taken before looking at state again, minimum 1.
			2 gives optimal behavior, insensitive to 'threshold'.
			0.2s is an appropriate human reaction time, which could also be used as a the time it takes for a human to
			change plans, even though that's not necessarily the same number.)
			This would correspond to a plan duration of reaction_time/tau = 0.2/0.02 = 10.
		threshold: A parameter between 0 and 1 to control behavior.
		tau: The OpenAI episodic cartpole-v1 has tau=0.02s between steps (no longer used here).
		fail_degrees: The angle at which the environment terminates the episode.
		fail_position: The position at which the environment terminates the episode, whether it's positive or negative.
	Return: None
	"""
	def __init__(self, plan_duration = 2, threshold = 0.9, tau = 0.02, fail_degrees = 15, fail_position = 2.4):
		super().__init__()
		self.actions_per_step = plan_duration
		self.fail_angle = fail_degrees/180*pi
		self.fail_position = fail_position
		self.threshold = threshold

		self.actions = []
		return


	def set_param(self, param):
		if not hasattr('plan_duration', 'param'):
			param.actions_per_step = 2
		if not hasattr('threshold', 'param'):
			param.threshold = 0.9
		if not hasattr('fail_degrees', 'param'):
			param.fail_degrees = 15
		if not hasattr('fail_position', 'param'):
			param.fail_position = 2.4

		self.actions_per_step = param.plan_duration
		self.fail_angle = self.fail_degrees/180*pi
		self.fail_position = param.fail_position
		self.threshold = param.threshold

		self.actions = []
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

		if len(self.actions) == 0:
			self.select_actions(state)

		action = self.actions.pop(0)
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
		if abs(angle) > self.threshold*self.fail_angle:
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
