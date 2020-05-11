import argparse


class CollectInput(object):
    def __init__(self):
        self.parser = None

    def control_experiment_input(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--json', default="CCP", type=str)
        self.parser.add_argument('--sweeper_idx', default=0, type=int)
        self.parser.add_argument('--run_idx', default=0, type=int)
        return self.parser.parse_args()

    def write_jobs(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--json', default="CCP", type=str)
        self.parser.add_argument('--sweeper', default="control", type=str)
        self.parser.add_argument('--lines', default=1, type=int)
        return self.parser.parse_args()
