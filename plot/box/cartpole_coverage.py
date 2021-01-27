import os
import sys
cwd = os.getcwd()
sys.path.insert(0, cwd+'/../..')
from plot.box.utils_plot import *
from plot.box.paths_cartpole import *


def sweep_model():
    cms = {
        # "k5_p02_t1000": data10k_eps03_k5_p02_t1000,
    }
    te = {"true": trueenv}
    plot_generation(te, cms, ranges, "../img/sweep_model")


ranges = [0, 0.1, 0.5]

# sweep_model()
