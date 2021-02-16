import os
import sys
cwd = os.getcwd()
sys.path.insert(0, cwd+'/../..')
from plot.box.utils_plot import *
from plot.box.paths_puddle import *

def top_param():
    calibration = {
        "random data": pd_random,
        "return -320": pd_return320,
        "return -40": pd_return40,
    }
    random = pd_rnd
    te = {"true": pd_true}
    plot_compare_top(te, calibration, None, random, "total-return", "../img/puddle_top", outer=10)#, ylim=[-150, -10])

def sweep_model():
    k3_close_cms = {
        "random data": pd_random,
        "return -320": pd_return320,
        "return -40": pd_return40,
    }
    te = {"true": pd_true}
    plot_generation(te, k3_close_cms, ranges, "total-return", "../img/puddle_model", outer=10, sparse_reward=-1, max_len=1000)
    # plot_each_run(te, cms, "total-reward", "../img/v2_model_run", outer=10, sparse_reward=-1, max_len=1000)

def data_density():
    datasets = {
        "random": "../../data/hyperparam_v4/puddle/offline_data/esarsa/step10k_env/random/param_0/",
        "return -320": "../../data/hyperparam_v4/puddle/offline_data/esarsa/step10k_env/return-320/param_0/",
        "return -40": "../../data/hyperparam_v4/puddle/offline_data/esarsa/step10k_env/return-40/param_0/",
    }
    dimension = {
        0: "x",
        1: "y",
    }
    group = {"xy": [0, 1]}
    key="new state"
    for i in range(10):
        run = "traces-{}".format(i)
        plot_dataset(datasets, key, dimension, group, run, "../img/puddle_data_density")


if __name__ == '__main__':
    ranges = [0, 0.05, 0.1, 0.2, 0.5, 0.7, 0.9]
    top_param()
    sweep_model()
    data_density()
