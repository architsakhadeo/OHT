import os
import math
import numpy as np
import pandas as pd

"""
input: 
    models_data: {
        modelname: {
            run number 0: {parameter0: worst auc over all ensemble paths, ...}
            run number 1: {parameter0: worst auc over all ensemble paths, ...}
            ...
        }
    }
return:
    ranked: {
        modelname: {
            run number 0: ((param_1st: auc), (param_2nd: auc), ...)
            run number 1: ((param_1st: auc), (param_2nd: auc), ...)
            ...
        }
    }
"""
def ranking_allruns(models_data):
    ranked = {}
    for model in models_data.keys():
        ranked[model] = {}
        for rk in models_data[model].keys():
            ranked[model][rk] = sorting(models_data[model][rk])
    return ranked
def sorting(pvs):
    return (sorted(pvs.items(), key=lambda item:item[1]))[::-1]
    # st = [(k, v) for k, v in sorted(pvs.items(), key=lambda item: item[1])]
    # return st

"""
input:
    ranked: {
            run number 0: ((param_1st: auc), (param_2nd: auc), ...)
            run number 1: ((param_1st: auc), (param_2nd: auc), ...)
            ...
    }
    perc: percentile
    metric: {
            param_1st: auc
            param_end: auc
            ...
    }
return:
    worst_per_run: [
        [run number, param, true perf of param]
    ]
"""
def percentile_worst(ranked, perc, metric):
    filtered = []
    for rk in ranked.keys():
        idx = max(min(math.ceil(len(ranked[rk]) * perc), len(ranked[rk])-1), 1)
        # target = ranked[rk][idx]
        # filtered.append([rk, target[0], target[1]])  # run number, parameter, performance
        filtered += [[rk, kv[0], kv[1]] for kv in ranked[rk][0: idx]] # run number, parameter, performance
    worst_per_run = []
    for rk in ranked.keys():
        min_pk = None
        min_true_auc = np.inf
        for item in filtered:
            if item[0] == rk:
                if metric[item[1]] < min_true_auc:
                    min_true_auc = metric[item[1]]
                    min_pk = item[1]
        worst_per_run.append([rk, min_pk, min_true_auc])
    return worst_per_run
# def percentile(ranked, low, high, mode="pessimistic"):
#     filtered = []
#     for rk in ranked.keys():
#         l = int(len(ranked[rk]) * low)
#         h = int(len(ranked[rk]) * high)
#         filtered += [[rk, kv[0], kv[1]] for kv in ranked[rk][l: h]] # run number, parameter, performance
#
#     if mode=="pessimistic":
#         worst_per_run = []
#         for rk in ranked.keys():
#             temp = {}
#             for item in filtered:
#                 if item[0] == rk:
#                     temp[item[1]] = item[2]
#             sorted = sorting(temp)
#             worst_per_run.append([rk, sorted[-1][0], sorted[-1][1]])
#         return worst_per_run
#     elif mode=="optimistic":
#         best_per_run = []
#         for rk in ranked.keys():
#             temp = {}
#             for item in filtered:
#                 if item[0] == rk:
#                     temp[item[1]] = item[2]
#             sorted = sorting(temp)
#             best_per_run.append([rk, sorted[0][0], sorted[0][1]])
#         return best_per_run
#     else:
#         return filtered

def percentile_avgeraged_run(group_by_param, perc):
    # group_by_param = average_run(data)
    sorted_group = sorting(group_by_param)
    h = min(int(len(sorted_group) * perc), len(sorted_group)-1)
    return sorted_group[h][1]

def average_run(data):
    group_by_param = {}
    all_runs = list(data.keys())
    all_params = list(data[all_runs[0]].keys())
    for pk in all_params:
        group_by_param[pk] = []
        for rk in all_runs:
            # print(rk, len(list(data[rk].keys())))
            group_by_param[pk].append(data[rk][pk])
    for pk in all_params:
        group_by_param[pk] = np.array(group_by_param[pk]).mean()
    return group_by_param

# def average_param(data):
#     sum = 0
#     for pk in data.keys():
#         sum += data[pk]
#     return float(sum) / len(list(data.keys()))

"""
input:
    path: ensemble_paths of one ensemble model [path_ens1, path_ens2, ...]
return 
    data: {
            run number 0: {parameter0: [auc_ens1, auc_ens2, ...]}
            run number 1: {parameter0: [auc_ens1, auc_ens2, ...]}
          }
"""
def load_rewards(paths):
    data = {}
    for path in paths: # each ensemble seed
        params = os.listdir(path)
        for param in params: # each param
            pp = os.path.join(path, param)
            p_key = param#int(param.split("_")[1])

            temp = os.listdir(pp)
            runs = []
            for t in temp:
                if "rewards" in t:
                    runs.append(t)

            all_runs = {}
            for run in runs:
                r_per_step = pd.read_csv(os.path.join(pp, run))['rewards']
                # r_per_step = r_per_step[:50000]
                all_runs["run"+run.split("-")[1].split(".")[0]] = np.mean(np.array(r_per_step)) # {run number: auc / total step}

            for rk in all_runs.keys():
                if rk not in data.keys():
                    data[rk] = {p_key: []}
                if p_key not in data[rk].keys():
                    data[rk][p_key] = []
                data[rk][p_key].append(all_runs[rk])
    return data

# Load from episodes-x.csv
def load_sparseRewards(paths, reward=-1):
    data = {}
    for path in paths: # each ensemble seed
        params = os.listdir(path)
        for param in params: # each param
            pp = os.path.join(path, param)
            p_key = param#int(param.split("_")[1])

            temp = os.listdir(pp)
            runs = []
            for t in temp:
                if "episodes" in t:
                    runs.append(t)

            all_runs = {}
            for run in runs:
                # print(os.path.join(pp, run))
                ep_len = pd.read_csv(os.path.join(pp, run))['episode lengths']
                ep_len = np.array(ep_len).reshape((-1))
                all_runs["run"+run.split("-")[1].split(".")[0]] = len(ep_len)*reward / float(ep_len.sum()) # {run number: auc / total step}

            for rk in all_runs.keys():
                if rk not in data.keys():
                    data[rk] = {p_key: []}
                if p_key not in data[rk].keys():
                    data[rk][p_key] = []
                data[rk][p_key].append(all_runs[rk])
    return data

def load_epSteps(paths):
    data = {}
    for path in paths: # each ensemble seed
        params = os.listdir(path)
        for param in params: # each param
            pp = os.path.join(path, param)
            p_key = param#int(param.split("_")[1])

            temp = os.listdir(pp)
            runs = []
            for t in temp:
                if "episodes" in t:
                    runs.append(t)

            all_runs = {}
            for run in runs:
                r_per_step = pd.read_csv(os.path.join(pp, run))['episode lengths']
                negative_r_per_step = np.array(r_per_step)*-1.0
                # all_runs[int(run.split("-")[1].split(".")[0])] = np.mean(np.array(r_per_step)) # {run number: auc / total step}
                all_runs["run"+run.split("-")[1].split(".")[0]] = np.mean(negative_r_per_step) # {run number: auc / total step}

            for rk in all_runs.keys():
                if rk not in data.keys():
                    data[rk] = {p_key: []}
                if p_key not in data[rk].keys():
                    data[rk][p_key] = []
                data[rk][p_key].append(all_runs[rk])
    return data

"""
input:
    path: ensemble_paths of models
        {
            modelname: [path_ens1, path_ens2, ...]
        }
return:
    data: 
    { modelname: 
        {
            run number 0: {parameter0: worst auc over all ensemble paths, ...}
            run number 1: {parameter0: worst auc over all ensemble paths, ...}
        }
    }
"""
def loading_pessimistic(models_paths, source="reward"):
    models_data = {}
    for model in models_paths.keys():
        paths = models_paths[model]
        print("Loading data: {}: {}".format(model, paths[0]))
        if source == "reward":
            data = load_rewards(paths)
            # data = load_sparseRewards(paths)
        elif source == "episode":
            # Acrobot code (please do not delete)
            data = load_epSteps(paths)
        for rk in data.keys():
            if rk in data.keys():
                for pk in data[rk].keys():
                    if pk in data[rk].keys():
                        data[rk][pk] = np.array(data[rk][pk]).min()
                    else:
                        print("param doesn't exist", rk, pk)
            else:
                print("run doesn't exist", rk)

        models_data[model] = data
    return models_data