import os
import sys
cwd = os.getcwd()
sys.path.insert(0, cwd+'/../..')
from plot.box.utils_plot import *
from plot.box.paths_cartpoleNoisyA import *

def sweep_model():
    k3_close_cms = {
        "random data": rwd1_cpn01_10k_k3_close_random,
        "reward 0.995": rwd1_cpn01_10k_k3_close_reward0995,
        "reward 0.995 far": rwd1_cpn01_10k_k3_far_reward0995,
    }
    te = {"true": rwd1_cpn01_true}
    # plot_generation(te, k3_close_cms, ranges, "total-reward", "../img/rwd1_10k_k3_close_model", outer=10, max_len=1000)
    plot_generation(te, k3_close_cms, ranges, "total-return", "../img/rwd1_10k_k3_close_model", outer=10, max_len=1000)

def data_density():
    datasets = {
        "random": "../../data/hyperparam_v4/cartpole-noisy-action_rwd1/noise_0.1perc/offline_data/esarsa/step10k_env/random/param_0",
        "rwd0995": "../../data/hyperparam_v4/cartpole-noisy-action_rwd1/noise_0.1perc/offline_data/esarsa/step10k_env/reward0.995/param_0",
    }
    dimension = {
        0: "cart position",
        1: "cart velocity",
        2: "pole angle",
        3: "pole angular velocity",
    }
    group = {"cart": [0, 1], "pole": [2, 3]}
    key="new state"
    for i in range(10):
        run = "traces-{}".format(i)
        plot_dataset(datasets, key, dimension, group, run, "../img/data_density")


if __name__ == '__main__':
    ranges = [0, 0.05, 0.1, 0.2, 0.5, 0.7, 0.9]
    sweep_model()
    # compare_random()
    # data_density()
