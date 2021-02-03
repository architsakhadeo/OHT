import os
import sys
import numpy
cwd = os.getcwd()
sys.path.insert(0, cwd+'/../..')
from plot.box.utils_data import *
from plot.box.utils_plot import *
from plot.box.paths_final import *

def plot_compare_top(te, cms, cem, fqi, rand_lst, source, title,
                     ylim=None, yscale="linear", res_scale=1, outer=None, sparse_reward=None, max_len=np.inf):
    ranges = [0]
    # true env data dictionary
    te_data = loading_average(te, source, outer=outer, sparse_reward=sparse_reward, max_len=max_len)
    te_data = average_run(te_data["true"])
    # print(te_data)

    # fqi data
    # # best each run
    # fqi_data_all = loading_average(fqi, source) # 30 runs in total, but different parameters
    # fqi_rank = ranking_allruns(fqi_data_all)["fqi"]
    # fqi_data = []
    # for rk in fqi_rank.keys():
    #     fqi_data.append(fqi_rank[rk][0][1])
    # # all performance
    fqi_data_all = loading_average(fqi, source, outer=outer, sparse_reward=sparse_reward, max_len=max_len)["fqi"] # 30 runs in total, but different parameters
    # fqi_rank = ranking_allruns(fqi_data_all)["fqi"]
    fqi_data = []
    for rk in fqi_data_all.keys():
        for pk in fqi_data_all[rk].keys():
            fqi_data.append(fqi_data_all[rk][pk])
    

    cem_data_all = loading_average(cem, source, outer=outer, sparse_reward=sparse_reward, max_len=max_len)["cem"] # 30 runs in total, but different parameters
    # cem_rank = ranking_allruns(cem_data_all)["cem"]
    cem_data = []
    for rk in cem_data_all.keys():
        for pk in cem_data_all[rk].keys():
            cem_data.append(cem_data_all[rk][pk])

    # random data list
    rand_data = performance_by_param(rand_lst, te_data)

    # top true env data performance
    te_thrd = []
    for perc in ranges:
        te_thrd.append(percentile_avgeraged_run(te_data, perc))

    filtered = {"random": [rand_data], "fqi": [fqi_data], "cem": [cem_data]}
    #filtered = {"random": [rand_data]}
    cms_data = loading_average(cms, source, outer=outer, sparse_reward=sparse_reward, max_len=max_len)
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
    plot_violins(filtered, te_thrd, ranges, title, ylim=ylim, yscale=yscale, res_scale=res_scale)
    #plot_boxs(filtered, te_thrd, ranges, title, ylim=ylim, yscale=yscale, res_scale=res_scale)

def performance_by_param(rand_lst, data):
    perf = []
    for i in rand_lst:
        pk = "param_{}".format(i)
        perf.append(data[pk])
    return perf

def arcrobot():
    calibration = {
        #"k1_notimeout": k1_notimeout,
        #"k1_timeout1000": k1_timeout1000,
        #"k3ensemble_notimeout": k3ensemble_notimeout,
        #k3ensemble_timeout1000": k3ensemble_timeout1000,
        #"k3ensemble_adversarial_notimeout": k3ensemble_adversarial_notimeout,
        #"k3ensemble_adverarial_timeout1000": k3ensemble_adverarial_timeout1000,
        "calibration model with inner runs": k3_adversarial_timeout1000_subruns 
    }
    cem = {"cem": ac_CEM}
    random = ac_rnd
    te = {"true": ac_true_env}
    fqi = {"fqi": ac_fqi}
    plot_compare_top(te, calibration, fqi, random, "episode", "../img/final_acrobot_violin_log", ylim=[50,200], yscale="log", res_scale=-1, outer=10)

def cartpole_rs():
    calibration = {
        "trueStart_adversarialTrans_t1000": RS_trueStart_farTrans_time1000,
        "distStart_closeTrans_t200": RS_distStart_closeTrans_time200,
    }
    random = cpn1_rnd
    te = {"true": cpn1_true_env}
    fqi = {"fqi": RS_cpn1_fqi}
    plot_compare_top(te, calibration, fqi, random, "reward", "../img/final_cartpole_rs", outer=10)

def cartpole():
    calibration = {
        # "trueStart_adversarialTrans_t1000": trueStart_farTrans_time1000,
        # # "distStart_adversariaTrans_t200": distStart_farTrans_time200,
        # "distStart_closeTrans_t200": distStart_closeTrans_time200,

        # "calibration": trueStart_farTrans_time1000,
        # "with random start": RS_trueStart_farTrans_time1000,

        "far trans": v2_trueStart_farTrans_time1000,
    }
    random = cpn1_rnd
    te = {"true": v2_cpn1_true_env}
    fqi = {"fqi": cpn1_fqi}
    plot_compare_top(te, calibration, fqi, random, "reward", "../img/v2_top_param_cartpole", outer=10)

def cartpole_ablation():
    calibration = {
        # "trueStart_adversarialTrans_t1000": trueStart_farTrans_time1000, #
        # # "trueStart_adversarialTrans_t0": trueStart_farTrans_time0,
        # "noAdversarial_t1000": trueStart_closeTrans_time1000, #
        # # "noAdversarial_t0": trueStart_closeTrans_time0,
        # "noEnsemble_t1000": trueStart_noEnsemble_time1000, #
        # # "noEnsemble_t0": trueStart_noEnsemble_time0,

        "calibration": trueStart_farTrans_time1000, #
        "no Adversarial": trueStart_closeTrans_time1000, #
        "no Ensemble": trueStart_noEnsemble_time1000, #
    }
    random = cpn1_rnd
    te = {"true": cpn1_true_env}
    fqi = {"fqi": cpn1_fqi}
    # plot_compare_top(te, calibration, fqi, random, "reward", "../img/ablation_cartpole", outer=10)
    plot_each_run(te, calibration, "reward", "../img/ablation_cartpole", outer=10)

def cartpole_size():
    calibration = {
        "10k": trueStart_farTrans_time1000,
        "5k": trueStart_farTrans_time1000_5k,
        "2k": trueStart_farTrans_time1000_2k,
        "1k": trueStart_farTrans_time1000_1k,
    }
    random = cpn1_rnd
    te = {"true": cpn1_true_env}
    fqi = {"fqi": cpn1_fqi}
    # plot_compare_top(te, calibration, fqi, random, "reward", "../img/datset_size_cartpole", outer=10)
    plot_each_run(te, calibration, "reward", "../img/dataset_size_cartpole", outer=10)


arcrobot()
# cartpole_rs()
#cartpole()
# cartpole_ablation()
# cartpole_size()