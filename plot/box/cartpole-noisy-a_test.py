import os
import sys
cwd = os.getcwd()
sys.path.insert(0, cwd+'/../..')
from plot.box.utils_plot import *
from plot.box.paths_cartpoleNoisyA_test import *

def sweep_model():
    k10_far_cms = {
        "normal": cpn01_k10_far_reward002,
        "0.01 terminal": cpn01_k10_far_reward002_risk001,
        "0.1 terminal": cpn01_k10_far_reward002_risk01,
        "chosen data": cpn01_k10_far_reward002_withT,
        "0.01 terminal chosen data": cpn01_k10_far_reward002_withT_risk001,
        "0.1 terminal chosen data": cpn01_k10_far_reward002_withT_risk01,
    }
    te = {"true": cpn01_true}
    plot_generation(te, k10_far_cms, ranges, "total-reward", "../img/test_10k_k10_far_model", outer=10, sparse_reward=-1, max_len=1000)
    # plot_each_run(te, k10_far_cms, "total-reward", "../img/test_10k_k10_far_model_run", outer=10, sparse_reward=-1, max_len=1000)

def data_density():
    datasets = {
        "eps 1": "../../data/hyperparam_v4/cartpole-noisy-action_test/noise_0/offline_data/esarsa/step500k_env/fixed_eps1/param_0/",
        "training": "../../data/hyperparam_v4/cartpole-noisy-action_test/noise_0/offline_data/esarsa/step500k_env/learning/param_0",
        "eps 0": "../../data/hyperparam_v4/cartpole-noisy-action_test/noise_0/offline_data/esarsa/step500k_env/fixed_eps0/param_0",
    }
    dimension = {
        0: "cart position",
        1: "cart velocity",
        2: "pole angle",
        3: "pole angular velocity",
    }
    group = {"cart": [0, 1], "pole": [2, 3]}
    key="new state"
    for i in range(1):
        run = "traces-{}".format(i)
        plot_dataset(datasets, key, dimension, group, run, "../img/data_density", setlim=[0, 10000])

def termination_type():
    datasets = {
        "eps 1": "../../data/hyperparam_v4/cartpole-noisy-action_test/noise_0.1perc/offline_data/esarsa/step500k_env/fixed_eps1/param_0/",
        "training": "../../data/hyperparam_v4/cartpole-noisy-action_test/noise_0.1perc/offline_data/esarsa/step500k_env/learning/param_0",
        "eps 0": "../../data/hyperparam_v4/cartpole-noisy-action_test/noise_0.1perc/offline_data/esarsa/step500k_env/fixed_eps0/param_0",
    }
    types = ["pos", "ang"]
    for i in range(1):
        run = "traces-{}".format(i)
        # plot_termination(datasets, types, run, "../img/data_termination_scatter")
        plot_termination_perc(datasets, types, run, "../img/data_termination")


if __name__ == '__main__':
    ranges = [0, 0.05, 0.1, 0.2, 0.5, 0.7, 0.9]
    # sweep_model()
    # data_density()
    termination_type()