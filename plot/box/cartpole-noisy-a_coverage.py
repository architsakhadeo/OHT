import os
import sys
cwd = os.getcwd()
sys.path.insert(0, cwd+'/../..')
from plot.box.utils_plot import *
from plot.box.paths_cartpoleNoisyA import *

def sweep_model_rs():

    cms = {
        "trueStart_adversarialTrans_t1000": RS_trueStart_farTrans_time1000,
        "distStart_adversarialTrans_t200": RS_distStart_farTrans_time200,
        "distStart_closeTrans_t200": RS_distStart_closeTrans_time200,
    }
    te = {"true": cpn1_true_env}
    plot_generation(te, cms, ranges, "reward", "../img/sweep_model_RS")

def sweep_model_er():

    cms = {
        "trueStart_adversarialTrans_t1000": ER_trueStart_farTrans_time1000,
        "trueStart_closeTrans_t1000": ER_trueStart_closeTrans_time1000,
    }
    te = {"true": cpn1_true_env}
    plot_generation(te, cms, ranges, "reward", "../img/sweep_model_ER")
    plot_each_run(te, cms, "reward", "../img/check_avg_ablation_ER")

"""
no random restart
"""
def sweep_model():
    cms = {
        "trueStart_adversariaTrans_t1000": trueStart_farTrans_time1000,
        # "distStart_adversariaTrans_t200": distStart_farTrans_time200,
        # "distStart_closeTrans_t200": distStart_closeTrans_time200,
        # "trueStart_adversarialTrans_t1000": trueStart_farTrans_time1000, #
        # "trueStart_adversarialTrans_t0": trueStart_farTrans_time0,
        # "noAdversarial_t1000": trueStart_closeTrans_time1000, #
        # "noAdversarial_t0": trueStart_closeTrans_time0,
        # "noEnsemble_t1000": trueStart_noEnsemble_time1000, #
        # "noEnsemble_t0": trueStart_noEnsemble_time0,
        #
        # "k5_t200": trueStart_farTrans_time200_k5,
        # "k5_t1000": trueStart_farTrans_time1000_k5,

    }
    te = {"true": cpn1_true_env}
    plot_generation(te, cms, ranges, "reward", "../img/sweep_model_random_break_tie")

def check_run_model():
    cms = {
        # "no random start": trueStart_farTrans_time1000,
        # "with random start": RS_trueStart_farTrans_time1000,
        "far trans": v2_trueStart_farTrans_time1000,
        "close trans": v2_trueStart_closeTrans_time1000,
    }
    te = {"true": v2_cpn1_true_env}
    plot_generation(te, cms, ranges, "reward", "../img/v2_model", outer=10, sparse_reward=-1, max_len=1000)
    plot_each_run(te, cms, "reward", "../img/v2_model_run", outer=10, sparse_reward=-1, max_len=1000)
    # plot_generation(te, cms, ranges, "sparseReward", "../img/v2_model", outer=10, sparse_reward=-1, max_len=1000)
    # plot_each_run(te, cms, "sparseReward", "../img/v2_model_run", outer=10, sparse_reward=-1, max_len=1000)


def check_run_ablation():
    cms = {
        "calibration model": trueStart_farTrans_time1000, #
        # "distStart_adversariaTrans_t200": distStart_farTrans_time200,
        # "distStart_closeTrans_t200": distStart_closeTrans_time200,

        # "trueStart_adversarialTrans_t0": trueStart_farTrans_time0,
        "no Adversarial": trueStart_closeTrans_time1000, #
        # # "noAdversarial_t0": trueStart_closeTrans_time0,
        "no Ensemble": trueStart_noEnsemble_time1000, #
        # # "noEnsemble_t0": trueStart_noEnsemble_time0,
    }
    te = {"true": cpn1_true_env}
    plot_each_run(te, cms, "reward", "../img/check_avg_ablation_noRS")

def check_run_size():
    cms = {
        "10k": trueStart_farTrans_time1000,
        "5k": trueStart_farTrans_time1000_5k,
        "2k": trueStart_farTrans_time1000_2k,
        "1k": trueStart_farTrans_time1000_1k,
    }
    te = {"true": cpn1_true_env}
    plot_each_run(te, cms, "reward", "../img/check_avg_size_noRS")

if __name__ == '__main__':
    ranges = [0, 0.05, 0.1, 0.2, 0.5, 0.7, 0.9]
    # noise1()
    # sweep_model_rs()
    # sweep_model_er()
    # sweep_model()
    check_run_model()
    # check_run_ablation()
    # check_run_size()