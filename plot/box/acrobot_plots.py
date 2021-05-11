import os
import sys
cwd = os.getcwd()
sys.path.insert(0, cwd+'/../..')
from plot.box.utils_plot import *
from plot.box.paths_acrobot import *

def top_param():
    calibration = {
        "knn": ac_knn,
        "knn(laplace)": ac_laplace_knn,
        "network": ac_network,
        "network(laplace)": ac_laplace_network,
        "network(scaled)": ac_scale_network,
        "network(scaled+laplace)": ac_scale_laplace_network,
    }
    random = ac_rnd
    te = {"true": ac_true_temp}
    plot_compare_top(te, calibration, None, random, "totals", "../img/acrobot_top", outer=30, res_scale=-1)#, ylim=[-100, 0])

def sweep_model():
    calibration = {
        "calibration model": ac_offline,
    }
    te = {"true": ac_true}
    #plot_generation(te, calibration, ranges, "totals", "../img/acrobot_model", outer=30, sparse_reward=-1, max_len=1000, res_scale=-1)
    plot_each_run(te, calibration, "totals", "../img/acrobot_run", outer=30, sparse_reward=-1, max_len=1000)


if __name__ == '__main__':
    ranges = [0, 0.05, 0.1, 0.2, 0.5]
    top_param()
    #sweep_model()
