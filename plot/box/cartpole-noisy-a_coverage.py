import os
import sys
cwd = os.getcwd()
sys.path.insert(0, cwd+'/../..')
from plot.box.utils_plot import *
from plot.box.paths_cartpoleNoisyA import *


def discuss_plots():
    eps01_20k = {
        # "cm": noise50_data20k_eps01
    }
    te = {"true": noise50_true}

    plot_generation(te, eps0_20k, ranges, "../img/ns50_eps0_data20k")



ranges = [0, 0.1, 0.5]

discuss_plots()