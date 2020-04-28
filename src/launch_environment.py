import logging

from Remote.environment import serve
from Environments.CartpoleEnvironment import CartpoleEnvironment

logging.basicConfig()
serve(CartpoleEnvironment())
