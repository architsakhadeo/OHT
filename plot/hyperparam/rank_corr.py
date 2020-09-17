import matplotlib.pyplot as plt
import numpy as np

def corr_group(true_order, offline_order_all, envs):
    algs = list(offline_order_all.keys())
    fig, axs = plt.subplots(1, len(algs))
    for i in range(len(envs)):
        env = envs[i]
        od_true = []
        od_offl = []
        for k in true_order.keys():
            od_true.append(true_order[k])
            print(offline_order_all[env].keys())
            od_offl.append(offline_order_all[env][k])
        axs[i].scatter(od_true, od_offl)
        axs[i].plot([i for i in range(len(od_true))], "--", color="gray")
        axs[i].set_title(env)
    plt.show()

def text2order(slist):
    rank = {}
    for idx in range(len(slist)-1, -1, -1):
        key = slist[idx].split(" sum=")[0]
        rank[key] = idx
    return rank

def corr_from_log(offline_path, offline_key, online_key):
    offlog = offline_path + "/img/ordered_sum.txt"
    group = {}
    with open(offlog, "r") as off:
        off_lines = off.readlines()
    for idx in range(len(off_lines)):
        content = off_lines[idx].strip().strip(":").strip()
        print(content)
        if (content in offline_key) or (content == online_key):
            start = idx
            key = content
        if content == "" or idx == len(off_lines)-1:
            end = idx
            if key not in group.keys():
                group[key] = off_lines[start: end]
                print(key, "added. Content from {}, {}. {} line".format(start, end, end-start))
            elif key in group.keys():
                input(key + " exists. \nCurrent start and end are: {}, {}".format(start, end))

    true_order = text2order(group[online_key])
    offline_order = {}
    for ofk in offline_key:
        offline_order[ofk] = text2order(group[ofk])
    corr_group(true_order, offline_order, envs = offline_key)

def dqn_corr(offline_key, online_key):
    offline_paths = [
        "../../data/hyperparam/cartpole/offline_learning/dqn-adam/step1k_env",
        "../../data/hyperparam/cartpole/offline_learning/dqn-adam/step10k_env",
        "../../data/hyperparam/cartpole/offline_learning/dqn-adam/step20k_env",
    ]
    # online_path = "../../data/hyperparam/cartpole/online_learning/dqn-adam/step50k"
    for op in offline_paths:
        corr_from_log(op, offline_key, online_key)

def esarsa_corr(offline_key, online_key):
    offline_paths = [
        "../../data/hyperparam/cartpole/offline_learning/esarsa-adam/step1k_env",
        "../../data/hyperparam/cartpole/offline_learning/esarsa-adam/step10k_env",
        "../../data/hyperparam/cartpole/offline_learning/esarsa-adam/step20k_env",
    ]
    # online_path = "../../data/hyperparam/cartpole/online_learning/dqn-adam/step50k"
    for op in offline_paths:
        corr_from_log(op, offline_key, online_key)


def main():
    offlines = [
        "lockat_baseline",
        "lockat_halfbaseline",
        "lockat_quarterbaseline",
        "lockat_-0.1",
        "lockat_random",
    ]
    online = "sweep"
    dqn_corr(offlines, online)
    esarsa_corr(offlines, online)

if __name__ == '__main__':
    main()