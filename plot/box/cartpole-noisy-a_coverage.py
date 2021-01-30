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
    plot_generation(te, cms, ranges, "../img/sweep_model_RS")

    """
    no random restart
    """
def sweep_model():
    cms = {
        "trueStart_adversariaTrans_t1000": trueStart_farTrans_time1000,
        # "distStart_adversariaTrans_t200": distStart_farTrans_time200,
        # "distStart_closeTrans_t200": distStart_closeTrans_time200,
        "trueStart_adversarialTrans_t1000": trueStart_farTrans_time1000, #
        "trueStart_adversarialTrans_t0": trueStart_farTrans_time0,
        "noAdversarial_t1000": trueStart_closeTrans_time1000, #
        "noAdversarial_t0": trueStart_closeTrans_time0,
        "noEnsemble_t1000": trueStart_noEnsemble_time1000, #
        "noEnsemble_t0": trueStart_noEnsemble_time0,
    }
    te = {"true": cpn1_true_env}
    plot_generation(te, cms, ranges, "../img/sweep_model_noRS")

def check_run():
    cms = {
        "trueStart_adversarialTrans_t1000": trueStart_farTrans_time1000, #
        # "distStart_adversariaTrans_t200": distStart_farTrans_time200,
        # "distStart_closeTrans_t200": distStart_closeTrans_time200,

        "trueStart_adversarialTrans_t0": trueStart_farTrans_time0,
        # "noAdversarial_t1000": trueStart_closeTrans_time1000, #
        # # "noAdversarial_t0": trueStart_closeTrans_time0,
        # "noEnsemble_t1000": trueStart_noEnsemble_time1000, #
        # # "noEnsemble_t0": trueStart_noEnsemble_time0,
    }
    te = {"true": cpn1_true_env}
    plot_each_run(te, cms, "../img/check_run_noRS")

if __name__ == '__main__':
    ranges = [0, 0.05, 0.1, 0.2, 0.5, 0.7, 0.9]
    # noise1()
    # sweep_model_rs()
    # sweep_model()
    check_run()
