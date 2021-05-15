import os
import sys
cwd = os.getcwd()
sys.path.insert(0, cwd+'/../..')
from plot.box.utils_plot import *
from plot.box.paths_puddle import *

def top_param():
    calibration = {
        "optimal (knn)": pd_optim_knn,
        "average (knn)": pd_suboptim_knn,
        "bad (knn)": pd_subsuboptim_knn,
        "optimal (network)": pd_optim_network,
        "average (network)": pd_suboptim_network,
        "bad (network)": pd_subsuboptim_network,
    }
    random = pd_rnd
    te = {"true": pd_true}
    #cem = {"calibration (cem)": pd_cemOffline}
    #plot_compare_top(te, calibration, None, random, "totals", "../img/puddle_top_lognegate_ylim", outer=30, yscale="log", res_scale=-1, ylim=[-100, 0])
    plot_compare_top(te, calibration, None, random, "totals", "../img/puddle_top_ylim", None, outer=30, ylim=[-100, 0])

def sweep_model():
    calibration = {
        "bad (network)": pd_subsuboptim_network,
        "average (network)": pd_suboptim_network,
        "optimal (network)": pd_optim_network,
        "bad (knn)": pd_subsuboptim_knn,
        "average (knn)": pd_suboptim_knn,
        "optimal (knn)": pd_optim_knn
    }
    random = pd_rnd
    te = {"true": pd_true}
    plot_generation(te, calibration, ranges, "totals", "../img/puddle_model_ylim", ylim = [-100, -20], outer=30, sparse_reward=-1, max_len=1000)
    #plot_each_run(te, calibration, "totals", "../img/puddleworld_run", outer=30, sparse_reward=-1, max_len=1000)

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
    ranges = [0]
    #top_param()
    sweep_model()
    #data_density()
