class BaseAgent:
	def set_param(self, param):
		raise NotImplementedError('Expected `set_param` to be implemented')

	def start(self, observation):
		raise NotImplementedError('Expected `start` to be implemented')

	def step(self, reward, observation):
		raise NotImplementedError('Expected `step` to be implemented')

	def end(self, reward):
		raise NotImplementedError('Expected `end` to be implemented')
