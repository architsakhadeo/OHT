import logging

from Remote.environment import serve
from Environments.ContinuingCartpoleEnvironment import CartpoleEnvironmentContinuing

logging.basicConfig()
serve(CartpoleEnvironmentContinuing())
