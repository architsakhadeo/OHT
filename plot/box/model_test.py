import os
import sys
cwd = os.getcwd()
sys.path.insert(0, cwd+'/../..')
from plot.box.utils_plot import *
from plot.box.paths_model_test import *


def sweep_model():
    cms = {
        "baseline": data10k_eps01_k5_p02_t0,
        # "noisydata_10": noisystate_n10,
        "noisydata_50": noisystate_n50,
        # "startfurthest_t0": startfurthes_t0,
        "startfurthest_t200": startfurthes_t200,
        # "timeout100": timeout_t100,
        "timeout200": timeout_t200,
        # "timeout500": timeout_t500,
        # "transfurthest_t0": transfurthest_t0,
        # "transfurthest_t200": transfurthest_t200,
    }
    te = {"true": trueenv}
    plot_generation(te, cms, ranges_top, "../img/model_compare_top30")
    plot_generation(te, cms, ranges, "../img/model_compare_top60")
    plot_generation(te, cms, ranges_all, "../img/model_compare_all")

ranges_top = [0, 0.1, 0.2, 0.3]
ranges = [0, 0.2, 0.4, 0.6]
ranges_all = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

sweep_model()