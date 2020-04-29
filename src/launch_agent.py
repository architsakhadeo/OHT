import logging

from Remote.agent import serve
from Agents.ExpectedSarsaLambda import ExpectedSarsaTileCodingContinuing

logging.basicConfig()
serve(ExpectedSarsaTileCodingContinuing())
