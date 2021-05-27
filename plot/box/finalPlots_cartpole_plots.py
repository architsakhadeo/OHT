import os
import sys
cwd = os.getcwd()
sys.path.insert(0, cwd+'/../..')
from plot.box.utils_plot import *
from plot.box.paths_cartpole_finalPlots import *

def top_param():
    # PLOT 7

    calibration = {
        "Random policy": cart_knnlaplace_random_10k_plot7,
        "Medium policy": cart_knnlaplace_suboptim_10k_plot7,
        "Near-optimal policy": cart_knnlaplace_optim_10k_plot7,
        # "learning policy": cart_knnlaplace_learningpolicy_10k_plot7
    }
    random = cart_rnd
    true = {"true": cart_true}
    plot_compare_top(true, calibration, None, random, "cartpole-failures", "../img/finalPlots/cartpole/plot4/plot4_cartpole",
                     outer=30, res_scale=-1, ylabel="Return per episode", ylim=[[-418, 5408]], right_ax=[],label_ncol=4)


def sweep_model():
    calibration = {
        "random policy (KNN)": cart_random_knn,
        "optimal policy (KNN)": cart_optim_knn
    }
    te = {"true": cart_true}
    plot_generation(te, calibration, ranges, "totals", "../img/acrobot_model", outer=30, sparse_reward=-1, max_len=1000, res_scale=-1)
    #plot_each_run(te, calibration, "totals", "../img/acrobot_model", outer=30, sparse_reward=-1, max_len=1000)

def data_density():
    datasets = {
        "15k": "../../data/hyperparam_v5/acrobot/offline_data/true_restarts/esarsa/step15k/optimalfixed_eps0/param_0/",
        "10k": "../../data/hyperparam_v5/acrobot/offline_data/true_restarts/esarsa/step10k/optimalfixed_eps0/param_0/",
        "5k": "../../data/hyperparam_v5/acrobot/offline_data/true_restarts/esarsa/step5k/optimalfixed_eps0/param_0/",
    }
    dimension = {
        0: "0",
        1: "1",
        2: "2",
        3: "3",
        4: "4",
        5: "5",
    }
    group = {"01": [0, 1], "23": [2, 3], "45": [4, 5]}
    key="new state"
    for i in range(10):
        run = "traces-{}".format(i)
        plot_dataset(datasets, key, dimension, group, run, "../img/data_density")

if __name__ == '__main__':
    ranges = [0]
    top_param()
    #sweep_model()
    # data_density()
    #top_param()
