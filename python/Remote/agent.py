from concurrent import futures

import grpc
import json

import Remote.remote_pb2 as remote_pb2
import Remote.remote_pb2_grpc as remote_pb2_grpc

class RemoteAgent(remote_pb2_grpc.AgentServicer):
	def __init__(self, agent):
		self.agent = agent

	def Initialize(self, attr, context):
		self.agent.set_param(json.loads(attr.experiment.attributes))
		# TODO: use attr.run.run to load the random seed or whatever
		return remote_pb2.Empty()

	def Start(self, state, context):
		action = self.agent.start(state.values)
		return remote_pb2.Action(action = action)

	def Step(self, result, context):
		if result.terminal:
			self.agent.end(result.reward)
			return remote_pb2.Empty()

		(action, unused) = self.agent.step(result.reward, result.state.values)
		return remote_pb2.Action(action = action)



def serve(agent):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    remote_pb2_grpc.add_AgentServicer_to_server(
        RemoteAgent(agent), server)
    server.add_insecure_port('[::]:8081')
    server.start()
    server.wait_for_termination()
