import os
import sys
import numpy
cwd = os.getcwd()
sys.path.insert(0, cwd+'/../..')
from plot.box.utils_data import *
from plot.box.utils_plot import *
from plot.box.paths_acrobot import *

def plot_generation(te, cms, title, ylim=None):

    te_data = loading_pessimistic(te)
    # te_rank = ranking_allruns(te_data)["true"]
    te_data = te_data["true"]

    te_thrd = []
    for perc in ranges:
        te_thrd.append(percentile_avgeraged_run(te_data, perc))

    cms_data = loading_pessimistic(cms)
    filtered = {}
    models_rank = ranking_allruns(cms_data)
    for model in cms_data.keys():
        ranks = models_rank[model]

        filtered[model] = []
        for perc in ranges:
            target = percentile(ranks, perc)
            data = [te_data[item[0]][item[1]] for item in target]
            filtered[model].append(data)

    plot_boxs(filtered, te_thrd, labels, title, ylim=ylim)


def sweep_model():
    cms = {
        "k1_p0": data10k_eps10_k1_p0,
        "k5_p20": data10k_eps10_k5_p20_ens,
    }
    te = {"true": true_env}
    plot_generation(te, cms, "../../img/sweep_model")

def sweep_coverage():
    cms = {
            "eps0": data10k_eps0_k5_p02_t0,
            "eps0.1": data10k_eps01_k5_p02_t0,
            "eps0.3": data10k_eps03_k5_p02_t0,
            "eps1": data10k_eps1_k5_p02_t0,
    }
    te = {"true": trueenv}
    plot_generation(te, cms, "../img/coverage_data10k")

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
    plot_generation(te, eps1_cms, "../img/datasize_eps1")

ranges = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
labels = ["0%\n(top param)", "10%", "20%", "30%", "40%", "50%"]
#ranges = [[0, 0.3], [0.3, 0.7], [0.7, 1.0]]

sweep_model()
# sweep_coverage()
#sweep_datasize()