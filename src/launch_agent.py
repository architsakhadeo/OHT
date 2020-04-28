import logging

from Remote.agent import serve
from Agents.HandCoded import HandCoded

logging.basicConfig()
serve(HandCoded())
