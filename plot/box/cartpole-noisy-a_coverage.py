import os
import sys
cwd = os.getcwd()
sys.path.insert(0, cwd+'/../..')
from plot.box.utils_plot import *
from plot.box.paths_cartpoleNoisyA import *

def noise1():
    eps0 = {
        "timeout1000": ns1_timeout1000,
    }
    te = {"true": ns1_true_env}

    plot_generation(te, eps0, ranges, "../img/ns1_eps0")

def sweep_model():

    cms = {
        "RS_distStart_adversarialTrans_t200": RS_farStart_farTrans_t200,
        "RS_trueStart_adversarialTrans_t1000": RS_trueStart_farTrans_t1000,
    }
    te = {"true": RS_ns1_true_env}
    plot_generation(te, cms, ranges, "../img/sweep_model")

    """
    no random restart
    """
    # cms = {
    #     "trueStart_adversariaTrans_t1000": trueStart_farTrans_time1000,
    #     # "distStart_adversariaTrans_t200": distStart_farTrans_time200,
    #     "distStart_closeTrans_t200": distStart_closeTrans_time200,
    # }
    # te = {"true": cpn1_true_env}


if __name__ == '__main__':
    ranges = [0, 0.1, 0.2, 0.5, 0.7, 0.9]
    # noise1()
    sweep_model()
