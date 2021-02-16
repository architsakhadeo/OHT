import os
import sys
cwd = os.getcwd()
sys.path.insert(0, cwd+'/../..')
from plot.box.utils_plot import *
from plot.box.paths_cartpoleNoisyA import *

def compare_random():
    calibration = {
        "random data": cpn01_k10_far_random,
        "reward -0.02": cpn01_k10_far_reward02,
        "reward -0.01": cpn01_k10_far_reward01,
        "reward -0.004": cpn01_k10_far_reward004,
        "reward -0.002": cpn01_k10_far_reward002,
    }
    random = cpn1_rnd
    te = {"true": cpn01_true}
    plot_compare_top(te, calibration, None, random, "total-reward", "../img/compare_random", outer=10)

def sweep_model():
    k3_close_cms = {
        "random data": cpn01_5k_k3_close_random,
        "reward -0.02": cpn01_5k_k3_close_reward02,
        "reward -0.01": cpn01_5k_k3_close_reward01,
        "reward -0.004": cpn01_5k_k3_close_reward004,
        "reward -0.002": cpn01_5k_k3_close_reward002,
    }
    k10_far_cms = {
        "random data": cpn01_k10_far_random,
        "reward -0.02": cpn01_k10_far_reward02,
        "reward -0.01": cpn01_k10_far_reward01,
        "reward -0.004": cpn01_k10_far_reward004,
        "reward -0.002": cpn01_k10_far_reward002,
    }
    te = {"true": cpn01_true}
    plot_generation(te, k3_close_cms, ranges, "total-reward", "../img/5k_k3_close_model", outer=10, sparse_reward=-1, max_len=1000)
    plot_generation(te, k10_far_cms, ranges, "total-reward", "../img/10k_k10_far_model", outer=10, sparse_reward=-1, max_len=1000)
    # plot_each_run(te, cms, "total-reward", "../img/v2_model_run", outer=10, sparse_reward=-1, max_len=1000)

def data_density():
    datasets = {
        "random": "../../data/hyperparam_v4/cartpole-noisy-action/noise_0.1perc/offline_data/esarsa/step10k_env/random/param_0",
        "reward -0.02": "../../data/hyperparam_v4/cartpole-noisy-action/noise_0.1perc/offline_data/esarsa/step10k_env/reward-0.02/param_0",
        "reward -0.01": "../../data/hyperparam_v4/cartpole-noisy-action/noise_0.1perc/offline_data/esarsa/step10k_env/reward-0.01/param_0",
        "reward -0.004": "../../data/hyperparam_v4/cartpole-noisy-action/noise_0.1perc/offline_data/esarsa/step10k_env/reward-0.004/param_0",
        "reward -0.002": "../../data/hyperparam_v4/cartpole-noisy-action/noise_0.1perc/offline_data/esarsa/step10k_env/reward-0.002/param_0",
    }
    dimension = {
        0: "cart position",
        1: "cart velocity",
        2: "pole angle",
        3: "pole angular velocity",
    }
    group = {"cart": [0, 1], "pole": [2, 3]}
    key="new state"
    for i in range(10):
        run = "traces-{}".format(i)
        plot_dataset(datasets, key, dimension, group, run, "../img/data_density")

def performance_dataset():
    datasets = {
        "random": "../../data/hyperparam_v4/cartpole-noisy-action/noise_0.1perc/offline_data/esarsa/step10k_env/random/param_0",
        "reward -0.02": "../../data/hyperparam_v4/cartpole-noisy-action/noise_0.1perc/offline_data/esarsa/step10k_env/reward-0.02/param_0",
        "reward -0.01": "../../data/hyperparam_v4/cartpole-noisy-action/noise_0.1perc/offline_data/esarsa/step10k_env/reward-0.01/param_0",
        "reward -0.004": "../../data/hyperparam_v4/cartpole-noisy-action/noise_0.1perc/offline_data/esarsa/step10k_env/reward-0.004/param_0",
        "reward -0.002": "../../data/hyperparam_v4/cartpole-noisy-action/noise_0.1perc/offline_data/esarsa/step10k_env/reward-0.002/param_0",
        "reward -0.002 eps 0.1": "../../data/hyperparam_v4/cartpole-noisy-action/noise_0.1perc/offline_data/esarsa/step10k_env/reward-0.002_eps0.1/param_0",
    }
    fig, ax = plt.subplots()
    all_cases = []
    for case in datasets:
        path = datasets[case]
        all_runs = os.listdir(path)
        all_len = []
        for r in all_runs:
            if "trace" in r:
                termins = pd.read_csv(os.path.join(path, r), error_bad_lines=False)["terminal"]
                count = len(np.where(termins==1)[0])
                count = count if count != 0 else 1
                ep_len = len(termins) / float(count)
                print("{:.2f}".format(ep_len))
                all_len.append(ep_len)
        print("{} policy: averaged length {:.2f}".format(case, np.array(all_len).mean()))
        all_cases.append(all_len)
    vp = ax.violinplot(all_cases)
    # plt.ylim(0, 1000)
    plt.xticks(list(range(1, len(datasets.keys())+1)), datasets.keys())
    plt.show()


if __name__ == '__main__':
    ranges = [0, 0.05, 0.1, 0.2, 0.5, 0.7, 0.9]
    # sweep_model()
    # compare_random()
    # data_density()
    performance_dataset()
