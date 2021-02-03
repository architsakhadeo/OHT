import os
import sys
import numpy
cwd = os.getcwd()
sys.path.insert(0, cwd+'/../..')
from plot.box.utils_data import *
from plot.box.utils_plot import *
from plot.box.paths_final import *


def arcrobot():
    calibration = {
        "k1_notimeout": k1_notimeout,
        "k1_timeout1000": k1_timeout1000,
        "k3ensemble_notimeout": k3ensemble_notimeout,
        "k3ensemble_timeout1000": k3ensemble_timeout1000,
        "k3ensemble_adversarial_notimeout": k3ensemble_adversarial_notimeout,
        "k3ensemble_adverarial_timeout1000": k3ensemble_adverarial_timeout1000
    }
    random = ac_rnd
    te = {"true": ac_true_env}
    fqi = {"fqi": ac_fqi}
    plot_compare_top(te, calibration, fqi, random, "episode", "../img/final_acrobot_violin_log", ylim=[50,200], yscale="log", res_scale=-1, outer=10)

def cartpole_rs():
    calibration = {
        "trueStart_adversarialTrans_t1000": RS_trueStart_farTrans_time1000,
        "distStart_closeTrans_t200": RS_distStart_closeTrans_time200,
    }
    random = cpn1_rnd
    te = {"true": cpn1_true_env}
    fqi = {"fqi": RS_cpn1_fqi}
    plot_compare_top(te, calibration, fqi, random, "reward", "../img/final_cartpole_rs", outer=10)

def cartpole():
    calibration = {
        # "trueStart_adversarialTrans_t1000": trueStart_farTrans_time1000,
        # # "distStart_adversariaTrans_t200": distStart_farTrans_time200,
        # "distStart_closeTrans_t200": distStart_closeTrans_time200,

        # "calibration": trueStart_farTrans_time1000,
        # "with random start": RS_trueStart_farTrans_time1000,

        "far trans": v2_trueStart_farTrans_time1000,
    }
    random = cpn1_rnd
    te = {"true": v2_cpn1_true_env}
    fqi = {"fqi": cpn1_fqi}
    plot_compare_top(te, calibration, fqi, random, "reward", "../img/v2_top_param_cartpole", outer=10)

def cartpole_ablation():
    calibration = {
        # "trueStart_adversarialTrans_t1000": trueStart_farTrans_time1000, #
        # # "trueStart_adversarialTrans_t0": trueStart_farTrans_time0,
        # "noAdversarial_t1000": trueStart_closeTrans_time1000, #
        # # "noAdversarial_t0": trueStart_closeTrans_time0,
        # "noEnsemble_t1000": trueStart_noEnsemble_time1000, #
        # # "noEnsemble_t0": trueStart_noEnsemble_time0,

        "calibration": trueStart_farTrans_time1000, #
        "no Adversarial": trueStart_closeTrans_time1000, #
        "no Ensemble": trueStart_noEnsemble_time1000, #
    }
    random = cpn1_rnd
    te = {"true": cpn1_true_env}
    fqi = {"fqi": cpn1_fqi}
    # plot_compare_top(te, calibration, fqi, random, "reward", "../img/ablation_cartpole", outer=10)
    plot_each_run(te, calibration, "reward", "../img/ablation_cartpole", outer=10)

def cartpole_size():
    calibration = {
        "10k": trueStart_farTrans_time1000,
        "5k": trueStart_farTrans_time1000_5k,
        "2k": trueStart_farTrans_time1000_2k,
        "1k": trueStart_farTrans_time1000_1k,
    }
    random = cpn1_rnd
    te = {"true": cpn1_true_env}
    fqi = {"fqi": cpn1_fqi}
    # plot_compare_top(te, calibration, fqi, random, "reward", "../img/datset_size_cartpole", outer=10)
    plot_each_run(te, calibration, "reward", "../img/dataset_size_cartpole", outer=10)


# arcrobot()
# cartpole_rs()
cartpole()
# cartpole_ablation()
# cartpole_size()