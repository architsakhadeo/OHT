from concurrent import futures

import grpc
import json

import Remote.remote_pb2 as remote_pb2
import Remote.remote_pb2_grpc as remote_pb2_grpc

class RemoteEnvironment(remote_pb2_grpc.EnvironmentServicer):
	def __init__(self, environment):
		self.environment = environment

	def Initialize(self, attr, context):
		self.environment.set_param(json.loads(attr.attributes.attributes))
		self.attributes = attr.attributes.attributes
		# TODO: use attr.run.run to load the random seed or whatever
		return remote_pb2.Empty()

	def Start(self, empty, context):
		state = self.environment.start()
		return remote_pb2.State(values = state)

	def Step(self, action, context):
		(state, reward, done) = self.environment.step(action.action)
		return remote_pb2.StepResult(
			state = remote_pb2.State(values = state),
			reward = reward,
			terminal = done
			)

	def GetAttributes(self, empty, context):
		return remote_pb2.Attributes(attributes = self.attributes) # TODO this is something different, provided by the enviornment



def serve(environment):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    remote_pb2_grpc.add_EnvironmentServicer_to_server(
        RemoteEnvironment(environment), server)
    server.add_insecure_port('[::]:8080')
    server.start()
    server.wait_for_termination()
