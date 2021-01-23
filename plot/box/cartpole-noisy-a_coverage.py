import os
import sys
cwd = os.getcwd()
sys.path.insert(0, cwd+'/../..')
from plot.box.utils_plot import *
from plot.box.paths_cartpoleNoisyA import *


def sweep_noise():
    data10k_cms = {
        "noise2": noise2_data10k_eps1,
        # "noise10": noise10_data10k_eps1,
        "noise50": noise50_data10k_eps1,
    }
    data20k_cms = {
        "noise2": noise2_data20k_eps1,
        # "noise10": noise10_data10k_eps1,
        "noise50": noise50_data20k_eps1,
    }
    te = {"true": noise0_trueenv}
    plot_generation(te, data10k_cms, ranges, "../img/noisy_sweepNoise_data10k_eps1")
    plot_generation(te, data20k_cms, ranges, "../img/noisy_sweepNoise_data20k_eps1")


def sweep_coverage():
    d10k_n50_cms = {
        "eps0.1": noise50_data10k_eps01,
        "eps1": noise50_data10k_eps1,
    }
    d10k_n10_cms = {
        "eps0.1": noise10_data10k_eps01,
        "eps1": noise10_data10k_eps1,
    }
    d10k_n2_cms = {
        "eps0.1": noise2_data10k_eps01,
        "eps1": noise2_data10k_eps1,
    }

    d20k_n50_cms = {
        "eps0.1": noise50_data20k_eps01,
        "eps1": noise50_data20k_eps1,
    }
    d20k_n10_cms = {
        "eps0.1": noise10_data20k_eps01,
        "eps1": noise10_data20k_eps1,
    }
    d20k_n2_cms = {
        "eps0.1": noise2_data20k_eps01,
        "eps1": noise2_data20k_eps1,
    }
    n50_te = {"true": noise50_true}
    n10_te = {"true": noise10_true}
    n2_te = {"true": noise2_true}
    # plot_generation(n2_te, d10k_n2_cms, ranges, "../img/noisy_sweepCoverage_data10k_noise2")
    # # plot_generation(n10_te, d10k_n10_cms, ranges, "../img/noisy_sweepCoverage_data10k_noise10")
    # plot_generation(n50_te, d10k_n50_cms, ranges, "../img/noisy_sweepCoverage_data10k_noise50")

    plot_generation(n2_te, d20k_n2_cms, ranges, "../img/noisy_sweepCoverage_data20k_noise2")
    # plot_generation(n10_te, d20k_n10_cms, ranges, "../img/noisy_sweepCoverage_data10k_noise10")
    plot_generation(n50_te, d20k_n50_cms, ranges, "../img/noisy_sweepCoverage_data20k_noise50")


def sweep_datasize():
    n50_eps01_cms = {
        "10k": noise50_data10k_eps01,
        "20k": noise50_data20k_eps01,
    }
    n50_eps1_cms = {
        "10k": noise50_data10k_eps1,
        "20k": noise50_data20k_eps1,
    }
    n2_eps01_cms = {
        "10k": noise2_data10k_eps01,
        "20k": noise2_data20k_eps01,
    }
    n2_eps1_cms = {
        "10k": noise2_data10k_eps1,
        "20k": noise2_data20k_eps1,
    }

    n50_te = {"true": noise50_true}

    plot_generation(n50_te, n50_eps01_cms, ranges, "../img/noisy_sweepSize_noise50_eps0.1")
    plot_generation(n50_te, n50_eps1_cms, ranges, "../img/noisy_sweepSize_noise50_eps1")

ranges = [0, 0.2, 0.4, 0.6, 0.8]
# ranges = [0, 0.1, 0.2, 0.3]

sweep_noise()
sweep_coverage()
sweep_datasize()