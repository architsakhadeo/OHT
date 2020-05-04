import include_parent_folder
import time
from utils.collect_config import Sweeper
from utils.collect_parser import CollectInput
from Experiments.control import Experiment

t0 = time.time()
ci = CollectInput()
parsers = ci.control_experiment_input()
json_name = parsers.json.lower()

sweeper = Sweeper('../Parameters/{}.json'.format(json_name.lower()), "env_model_param")
config = sweeper.parse(parsers.sweeper_idx)

exp = Experiment(config, parsers)
exp.env_simulator_training(file_path=None)