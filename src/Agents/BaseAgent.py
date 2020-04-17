class BaseAgent:
	def set_param(self, param):
		raise NotImplementedError('Expected `set_param` to be implemented')

	def start(self, observation):
		raise NotImplementedError('Expected `start` to be implemented')

	def step(self, reward, observation):
		raise NotImplementedError('Expected `step` to be implemented')

	def end(self, reward):
		raise NotImplementedError('Expected `end` to be implemented')

	def save(self, filepath):
		raise NotImplementedError('Expected `save` to be implemented')

	def load(self, filepath):
		# It's fine to not implement this, but it's used for offline learning.
		raise NotImplementedError('Expected `load` to be implemented')
