import os
import sys
import numpy
cwd = os.getcwd()
sys.path.insert(0, cwd+'/../..')
from plot.box.utils_data import *
from plot.box.utils_plot import *
from plot.box.paths_final import *

def plot_compare_top(te, cms, fqi, rand_lst, title, ylim=None, source="reward", yscale="linear"):
    ranges = [0]
    # true env data dictionary
    te_data = loading_pessimistic(te, source)
    te_data = average_run(te_data["true"])
    # print(te_data)

    # fqi data
    # # best each run
    # fqi_data_all = loading_pessimistic(fqi, source) # 30 runs in total, but different parameters
    # fqi_rank = ranking_allruns(fqi_data_all)["fqi"]
    # fqi_data = []
    # for rk in fqi_rank.keys():
    #     fqi_data.append(fqi_rank[rk][0][1])
    # # all performance
    fqi_data_all = loading_pessimistic(fqi, source)["fqi"] # 30 runs in total, but different parameters
    # fqi_rank = ranking_allruns(fqi_data_all)["fqi"]
    fqi_data = []
    for rk in fqi_data_all.keys():
        for pk in fqi_data_all[rk].keys():
            fqi_data.append(fqi_data_all[rk][pk])


    # random data list
    rand_data = performance_by_param(rand_lst, te_data)

    # top true env data performance
    te_thrd = []
    for perc in ranges:
        te_thrd.append(percentile_avgeraged_run(te_data, perc))

    filtered = {"random": [rand_data], "fqi": [fqi_data]}
    #filtered = {"random": [rand_data]}
    cms_data = loading_pessimistic(cms, source)
    models_rank = ranking_allruns(cms_data)
    for model in cms_data.keys():
        ranks = models_rank[model]

        filtered[model] = []
        for perc in ranges:
            target = percentile_worst(ranks, perc, te_data)
            # data = [te_data[item[1]] for item in target]
            data = [item[2] for item in target]
            filtered[model].append(data)
    # print(filtered)
    plot_violins(filtered, te_thrd, ranges, title, ylim=ylim, yscale=yscale)#, baseline=[fqi_data, "fqi"])
    #plot_boxs(filtered, te_thrd, ranges, title, ylim=ylim, yscale=yscale)

def performance_by_param(rand_lst, data):
    perf = []
    for i in rand_lst:
        pk = "param_{}".format(i)
        perf.append(data[pk])
    return perf

def arcrobot():
    calibration = {
        "k1_notimeout": k1_notimeout,
        "k1_timeout1000": k1_timeout1000,
        "k3ensemble_notimeout": k3ensemble_notimeout,
        "k3ensemble_timeout1000": k3ensemble_timeout1000,
        "k3ensemble_adversarial_notimeout": k3ensemble_adversarial_notimeout,
        "k3ensemble_adverarial_timeout1000": k3ensemble_adverarial_timeout1000
    }
    random = ac_rnd
    te = {"true": ac_true_env}
    fqi = {"fqi": ac_fqi}
    plot_compare_top(te, calibration, fqi, random, "../img/final_acrobot_violin_log", source="episode", ylim=[50,200], yscale="log")

def cartpole():
    calibration = {
        "trueStart_adversarialTrans_t1000": trueStart_farTrans_time1000,
        # "distStart_adversariaTrans_t200": distStart_farTrans_time200,
        "distStart_closeTrans_t200": distStart_closeTrans_time200,
    }
    random = cpn1_rnd
    te = {"true": cpn1_true_env}
    fqi = {"fqi": cpn1_fqi}
    plot_compare_top(te, calibration, fqi, random, "../img/final_cartpole")

def cartpole_rs():
    calibration = {
        "trueStart_adversarialTrans_t1000": RS_trueStart_farTrans_time1000,
        "distStart_closeTrans_t200": RS_distStart_closeTrans_time200,
    }
    random = cpn1_rnd
    te = {"true": cpn1_true_env}
    fqi = {"fqi": RS_cpn1_fqi}
    plot_compare_top(te, calibration, fqi, random, "../img/final_cartpole_rs")

def cartpole_ablation():
    calibration = {
        "trueStart_adversarialTrans_t1000": trueStart_farTrans_time1000, #
        "trueStart_adversarialTrans_t0": trueStart_farTrans_time0,
        "noAdversarial_t1000": trueStart_closeTrans_time1000, #
        "noAdversarial_t0": trueStart_closeTrans_time0,
        "noEnsemble_t1000": trueStart_noEnsemble_time1000, #
        "noEnsemble_t0": trueStart_noEnsemble_time0,
    }
    random = cpn1_rnd
    te = {"true": cpn1_true_env}
    fqi = {"fqi": cpn1_fqi}
    plot_compare_top(te, calibration, fqi, random, "../img/ablation_cartpole")


# arcrobot()
# cartpole()
# cartpole_rs()
cartpole_ablation()