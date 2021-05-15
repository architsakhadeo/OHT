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
        #"calibration model": ac_offline
    }
    random = ac_rnd
    te = {"true": ac_true_temp}
    #te = {"true": ac_true}
    #cem = {"calibration (cem)": ac_cemOffline}
    #plot_compare_top(te, calibration, None, random, "totals", "../img/acrobot_top", outer=30, res_scale=-1)#, ylim=[-100, 0])
    #plot_compare_top(te, calibration, None, random, "totals", "../img/acrobot_top", cem, outer=30, res_scale=-1)
    plot_compare_top(te, calibration, None, random, "totals", "../img/acrobot_top", outer=30, res_scale=-1)
    

def sweep_model():
    calibration = {
        "bad (network)": ac_subsuboptim_network,
        "average (network)": ac_suboptim_network,
        "optimal (network)": ac_optim_network,
        "bad (knn)": ac_subsuboptim_knn,
        "average (knn)": ac_suboptim_knn,
        "optimal (knn)": ac_optim_knn
    }
    te = {"true": ac_true}
    plot_generation(te, calibration, ranges, "totals", "../img/acrobot_model", outer=30, sparse_reward=-1, max_len=1000, res_scale=-1)
    #plot_each_run(te, calibration, "totals", "../img/acrobot_model", outer=30, sparse_reward=-1, max_len=1000)


if __name__ == '__main__':
    ranges = [0]
    #top_param()
    sweep_model()
