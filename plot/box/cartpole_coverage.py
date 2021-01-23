import os
import sys
cwd = os.getcwd()
sys.path.insert(0, cwd+'/../..')
from plot.box.utils_plot import *
from plot.box.paths_cartpole import *


def sweep_model():
    cms = {
        "k1_p0_t0": data10k_eps1_k1_p0_t0,
        "k5_p02_t0": data10k_eps1_k5_p02_t0,
        "k5_p02_t200": data10k_eps1_k5_p02_t200,
        # "k5_p02_t1000": data10k_eps03_k5_p02_t1000,
    }
    te = {"true": trueenv}
    plot_generation(te, cms, ranges, "../img/sweep_model")

def sweep_coverage():
    cms = {
            "eps0": data10k_eps0_k5_p02_t0,
            "eps0.1": data10k_eps01_k5_p02_t0,
            "eps0.3": data10k_eps03_k5_p02_t0,
            "eps1": data10k_eps1_k5_p02_t0,
    }
    te = {"true": trueenv}
    plot_generation(te, cms, ranges, "../img/coverage_data10k")

def sweep_datasize():
    eps0_cms = {
        "2.5k": data2d5k_eps0_k5_p02_t0,
        "5k": data5k_eps0_k5_p02_t0,
        "10k": data10k_eps0_k5_p02_t0,
        "20k": data20k_eps0_k5_p02_t0,
    }
    eps01_cms = {
        "2.5k": data2d5k_eps01_k5_p02_t0,
        "5k": data5k_eps01_k5_p02_t0,
        "10k": data10k_eps01_k5_p02_t0,
        "20k": data20k_eps01_k5_p02_t0,
    }
    eps03_cms = {
        "2.5k": data2d5k_eps03_k5_p02_t0,
        "5k": data5k_eps03_k5_p02_t0,
        "10k": data10k_eps03_k5_p02_t0,
        "20k": data20k_eps03_k5_p02_t0,
    }
    eps1_cms = {
        "2.5k": data2d5k_eps1_k5_p02_t0,
        "5k": data5k_eps1_k5_p02_t0,
        "10k": data10k_eps1_k5_p02_t0,
        "20k": data20k_eps1_k5_p02_t0,
    }
    te = {"true": trueenv}

    # plot_generation(te, eps0_cms, "../img/datasize_eps0")
    # plot_generation(te, eps01_cms, "../img/datasize_eps0.1")
    # plot_generation(te, eps03_cms, "../img/datasize_eps0.3")
    plot_generation(te, eps1_cms, ranges, "../img/datasize_eps1")

ranges = [0, 0.3, 0.6, 0.9]

sweep_model()
sweep_coverage()
sweep_datasize()