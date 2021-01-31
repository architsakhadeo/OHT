import os
import sys
cwd = os.getcwd()
sys.path.insert(0, cwd+'/../..')
from plot.box.utils_plot import *
from plot.box.paths_cartpole import *


def sweep_model():
    cms = {
        # "timeout200": ns1_timeout200,
        # "timeout1000": ns1_timeout1000,
    }
    te = {"true": noise0_trueenv}
    plot_generation(te, cms, ranges, "reward", "../img/sweep_model")


ranges = [0, 0.1, 0.5, 0.9]

# sweep_model()