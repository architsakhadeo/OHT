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
        "baseline": ns1_timeout1000,
        "farStart_timeout200": ns1_timeout200_farStart,
        "trueStart_timeout200": ns1_timeout200,
        "farStart_timeout1000": ns1_timeout1000_farStart,
    }
    te = {"true": ns1_true_env}
    plot_generation(te, cms, ranges, "../img/sweep_model")


if __name__ == '__main__':
    ranges = [0, 0.1, 0.2, 0.5, 0.7, 0.9]
    # noise1()
    sweep_model()
