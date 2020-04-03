import argparse


class CollectInput(object):
    def __init__(self):
        self.parser = None

    def control_experiment_input(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--domain', default="CP", type=str)
        self.parser.add_argument('--sweeper_idx', default=0, type=int)
        self.parser.add_argument('--run_idx', default=0, type=int)
        return self.parser.parse_args()
