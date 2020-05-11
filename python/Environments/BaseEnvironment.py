class BaseEnvironment:
	def set_param(self, param):
		raise NotImplementedError('Expected `set_param` to be implemented')

	def start(self):
		raise NotImplementedError('Expected `init` to be implemented')

	def step(self, action):
		raise NotImplementedError('Expected `init` to be implemented')

	def num_action(self):
		raise NotImplementedError('Expected `num_action` to be implemented')

	def state_dim(self):
		raise NotImplementedError('Expected `state_dim` to be implemented')

	def state_range(self):
		raise NotImplementedError('Expected `state_range` to be implemented')
