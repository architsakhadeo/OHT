import os
import sys
cwd = os.getcwd()
sys.path.insert(0, cwd+'/../..')
from plot.box.utils_plot import *
from plot.box.paths_cartpole import *


def plot_compare_fqi(te, fqi_dct, source, title, outer, res_scale, sparse_reward=None, max_len=np.inf, ylim=[]):
    # top true env data performance
    ranges = [0]
    te_thrd = []
    # true env data dictionary
    te_data = loading_average(te, source, outer=outer, sparse_reward=sparse_reward, max_len=max_len)
    te_data = average_run(te_data["true"])
    for perc in ranges:
        te_thrd.append(percentile_avgeraged_run(te_data, perc))

    filtered = {}
    for item in fqi_dct.items():
        print(item)
        k, v = item
        fqi = {k:v}
        fqi_data_all = loading_average(fqi, source, outer=outer, sparse_reward=sparse_reward, max_len=max_len)[k] # 30 runs in total, but different parameters
        total_fqi_data = {}
        fqi_data = []
        for rk in fqi_data_all.keys():
            for pk in fqi_data_all[rk].keys():
                if pk not in total_fqi_data:
                    total_fqi_data[pk] = [fqi_data_all[rk][pk]]
                else:
                    total_fqi_data[pk].append(fqi_data_all[rk][pk])
        for pk in total_fqi_data.keys():
            fqi_data.append(np.mean(total_fqi_data[pk]))

        filtered[k] = [fqi_data]
    plot_boxs(filtered, te_thrd, ranges, title, ylim=ylim)


def acrobot():
    ac_true_temp = ["../../data/hyperparam_v5/acrobot/online_learning/esarsa/step15k/sweep/"]
    ac_fqi_5k = ["../../data/hyperparam_v7/acrobot/offline_learning/random_restarts/fqi/fqi-adam/alpha_hidden_epsilon/step5k_env/optimalfixed_eps0/lambda1e-3/lockat_baseline_online/"]
    ac_fqi_15k = ["../../data/hyperparam_v7/acrobot/offline_learning/random_restarts/fqi/fqi-adam/alpha_hidden_epsilon/step15k_env/optimalfixed_eps0/lambda1e-3/lockat_baseline_online/"]
    ac_fqi_30k = ["../../data/hyperparam_v7/acrobot/offline_learning/random_restarts/fqi/fqi-adam/alpha_hidden_epsilon/step30k_env/optimalfixed_eps0/earlystop/lambda1e-5/lockat_baseline_online/"]
    ac_fqi_30k_mix = ["../../data/hyperparam_v7/acrobot/offline_learning/random_restarts/fqi/fqi-adam/alpha_hidden_epsilon/step30k_env/mixed_eps0/earlystop/lambda1e-5/lockat_baseline_online/"]

    te = {"true": ac_true_temp}
    fqi = {
        "fqi_5k": ac_fqi_5k,
        "fqi_15k": ac_fqi_15k,
        "fqi_30k": ac_fqi_30k,
        "fqi_30k_mix": ac_fqi_30k_mix
    }
    plot_compare_fqi(te, fqi, "totals", "../img/acrobot_fqi_nn", outer=30, res_scale=-1, ylim=[-3000, 0])

def puddle():
    pdrand_true = ["../../data/hyperparam_v5/puddlerand/online_learning/esarsa/step30k/sweep"]
    pdrand_fqi_5k = ["../../data/hyperparam_v7/puddlerand/offline_learning/random_restarts/fqi/fqi-adam/alpha_hidden_epsilon/step5k_env/optimalfixed_eps0/lambda1e-3/lockat_baseline_online/"]
    pdrand_fqi_15k = ["../../data/hyperparam_v7/puddlerand/offline_learning/random_restarts/fqi/fqi-adam/alpha_hidden_epsilon/step15k_env/optimalfixed_eps0/lambda1e-3/lockat_baseline_online/"]
    pdrand_fqi_30k = ["../../data/hyperparam_v7/puddlerand/offline_learning/random_restarts/fqi/fqi-adam/alpha_hidden_epsilon/step30k_env/optimalfixed_eps0/earlystop/lambda1e-3/lockat_baseline_online/"]
    pdrand_fqi_30k_mix = ["../../data/hyperparam_v7/puddlerand/offline_learning/random_restarts/fqi/fqi-adam/alpha_hidden_epsilon/step30k_env/mixed_eps0/earlystop/lambda1e-3/lockat_baseline_online/"]

    te = {"true": pdrand_true}
    fqi = {
        "fqi_5k": pdrand_fqi_5k,
        "fqi_15k": pdrand_fqi_15k,
        "fqi_30k": pdrand_fqi_30k,
        "fqi_30k_mix": pdrand_fqi_30k_mix
    }
    plot_compare_fqi(te, fqi, "totals", "../img/pdrand_fqi_nn", outer=30, res_scale=-1, ylim=[-40000, 0])

if __name__ == '__main__':
    acrobot()
    puddle()