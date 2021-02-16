import os
import sys
import numpy
cwd = os.getcwd()
sys.path.insert(0, cwd+'/../..')
from plot.box.utils_data import *
from plot.box.utils_plot import *
from plot.box.paths_final import *


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

        # "wo random restart": v2_trueStart_farTrans_time1000,
        # "random restart + dataset timeout200": v2_RSt200_trueStart_farTrans_time1000,
        # "random restart, eps 0": v2_RS_trueStart_farTrans_time1000,
        # "eps0.1": v3_RSA_trueStart_farTrans_time1000_eps01,
        # "random s-a restart, eps 1": v3_RSA_trueStart_farTrans_time1000_eps1,
        "simplest, eps 1": v3_trueStart_closeTrans_time1000_eps1_noEns,
    }
    random = cpn1_rnd
    te = {"true": v2_cpn1_true_env}
    fqi = {"fqi": v2_fqi}
    plot_compare_top(te, calibration, fqi, random, "total-reward", "../img/cartpole_simplest", outer=10)
    plot_each_run(te, calibration, "total-reward", "../img/cartpole_simplest_run", outer=10)

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

def final_plots_gs():
    def ac_good_cov():
        calibration = {
            "calibration (grid search)": AC_eps0
        }
        random = ac_rnd
        cem = {"calibration (cem)": AC_cem}
        te = {"true": AC_true}
        fqi = {"fqi": AC_fqi_eps0}
        # plot_compare_top(te, calibration, fqi, random, "total-episode", "../img/final_ac_good_cov", ylim=[80,300], yscale="log", res_scale=-1, outer=10)
        plot_compare_top(te, calibration, fqi, random, "total-episode", "../img/final_ac_good_cov", yscale="log", res_scale=-1, outer=10, cem=cem)

    def ac_study_cov():
        calibration = {
            "eps 0": AC_eps0,
            "eps 0.25": AC_eps025,
            "eps 1": AC_eps1,
        }
        random = ac_rnd
        te = {"true": AC_true}
        fqi = {
            "eps 0": AC_fqi_eps0,
            "eps 0.25": AC_fqi_eps025,
            "eps 1": AC_fqi_eps1,
        }
        # plot_compare_top(te, calibration, fqi, random, "total-episode", "../img/final_ac_study_cov", ylim=[50,200], yscale="log", res_scale=-1, outer=10)
        plot_generation(te, calibration, [0, 0.1, 0.2, 0.5, 0.7, 0.9], "total-episode", "../img/final_ac_study_cov", ylim=[80,300], yscale="log", res_scale=-1, outer=10)

    def cp_good_cov():
        calibration = {
            "calibration (grid search)": CP_eps1
        }
        random = cpn1_rnd
        cem = {"calibration (cem)": CP_cem}
        te = {"true": CP_true}
        fqi = {"fqi": CP_fqi_eps1}
        plot_compare_top(te, calibration, fqi, random, "total-reward", "../img/final_cp_good_cov", outer=10, cem=cem)

    def cp_study_cov():
        calibration = {
            "eps 0": CP_eps0,
            "eps 0.25": CP_eps025,
            "eps 1": CP_eps1,
        }
        random = cpn1_rnd
        te = {"true": CP_true}
        fqi = {
            "eps 0": CP_fqi_eps0,
            "eps 0.25": CP_fqi_eps025,
            "eps 1": CP_fqi_eps1,
        }
        plot_generation(te, calibration, [0, 0.1, 0.2, 0.5, 0.7, 0.9], "total-reward", "../img/final_cp_study_cov", outer=10, sparse_reward=-1, max_len=1000)

    def ac_fqi_cov():
        random = ac_rnd
        te = {"true": AC_true}

        cms_eps0 = {
            "calibration": AC_eps0,
        }
        fqi_eps0 = {"fqi": AC_fqi_eps0}
        # plot_compare_top(te, cms_eps0, fqi_eps0, random, "total-episode", "../img/ac_fqi_cov_eps0", ylim=[50,200], yscale="linear", res_scale=-1, outer=10)
        plot_compare_top(te, cms_eps0, fqi_eps0, random, "total-episode", "../img/ac_fqi_cov_eps0", yscale="linear", res_scale=-1, outer=10)

        cms_eps1 = {
            "calibration": AC_eps1,
        }
        fqi_eps1 = {"fqi": AC_fqi_eps1}
        # plot_compare_top(te, cms_eps1, fqi_eps1, random, "total-episode", "../img/ac_fqi_cov_eps1", ylim=[50,200], yscale="linear", res_scale=-1, outer=10)
        plot_compare_top(te, cms_eps1, fqi_eps1, random, "total-episode", "../img/ac_fqi_cov_eps1", yscale="linear", res_scale=-1, outer=10)

        cms_eps025 = {
            "calibration": AC_eps025,
        }
        fqi_eps025 = {"fqi": AC_fqi_eps025}
        # plot_compare_top(te, cms_eps025, fqi_eps025, random, "total-episode", "../img/ac_fqi_cov_eps0.25", ylim=[50,200], yscale="linear", res_scale=-1, outer=10)
        plot_compare_top(te, cms_eps025, fqi_eps025, random, "total-episode", "../img/ac_fqi_cov_eps0.25", yscale="linear", res_scale=-1, outer=10)

    def cp_fqi_cov():
        random = cpn1_rnd
        te = {"true": CP_true}

        cms_eps0 = {
            "calibration": CP_eps0,
        }
        fqi_eps0 = {"fqi": CP_fqi_eps0}
        plot_compare_top(te, cms_eps0, fqi_eps0, random, "total-reward", "../img/cp_fqi_cov_eps0", outer=10)

        cms_eps1 = {
            "calibration": CP_eps1,
        }
        fqi_eps1 = {"fqi": CP_fqi_eps1}
        plot_compare_top(te, cms_eps1, fqi_eps1, random, "total-reward", "../img/cp_fqi_cov_eps1", outer=10)

        cms_eps025 = {
            "calibration": CP_eps025,
        }
        fqi_eps025 = {"fqi": CP_fqi_eps025}
        plot_compare_top(te, cms_eps025, fqi_eps025, random, "total-reward", "../img/cp_fqi_cov_eps0.25", outer=10)

    ac_good_cov()
    ac_study_cov()
    cp_good_cov()
    cp_study_cov()
    ac_fqi_cov()
    cp_fqi_cov()

# arcrobot()
# cartpole_rs()
# cartpole()
# cartpole_ablation()
# cartpole_size()

final_plots_gs()
