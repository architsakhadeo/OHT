import os
import sys
import numpy
cwd = os.getcwd()
sys.path.insert(0, cwd+'/../..')
from plot.box.utils_data import *
from plot.box.utils_plot import *
from plot.box.paths_final import *

def cartpole():
    calibration = {
        # "test": ["../../data/hyperparam_rs_t200_v2/cartpole-noisy-action/noise_1perc/offline_learning/knn-ens/randomInit/farTrans/k3/timeout1000/esarsa/step10k_env/data_eps0/drop0.3/ensembleseed0"],
        "test": ["../../data/hyperparam_v2/cartpole-noisy-nonSparse/noise_1perc/offline_learning/knn-ens/randomInit/farTrans/k3/timeout1000/esarsa/step10k_env/data_eps0/drop0/ensembleseed0/"],
    }
    random = cpn1_rnd
    te = {"true": ["../../data/hyperparam_v2/cartpole-noisy-nonSparse/noise_1perc/online_learning/esarsa-adam/step50k/sweep/"]}
    fqi = {"fqi": cpn1_fqi}
    plot_compare_top(te, calibration, fqi, random, "total-reward", "../img/test", outer=10)
    plot_each_run(te, calibration, "total-reward", "../img/test_run", outer=10)


cartpole()
