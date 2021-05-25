import os
import sys
cwd = os.getcwd()
sys.path.insert(0, cwd+'/../..')
from plot.box.utils_plot import *
from plot.box.paths_puddle import *

def top_param():
    calibration = {
        "knn": pdrand_knn_5k,
        "knn(laplace)": pdrand_laplace_knn_test1_5k,
        # "network": pdrand_network_5k,
        # "network(laplace)": pdrand_laplace_network_5k,
        "network(scaled)": pdrand_scale_network_5k,
        "network(scaled+laplace)": pdrand_scale_laplace_network_5k,
    }
    random = pdrand_rnd
    te = {"true": pdrand_true}
    # plot_compare_top(te, calibration, None, random, "totals", "../img/puddlerand_top_zoomin", outer=30, ylim=[-60, -25]) #None)#

    calibration = {
        "15k knn": pdrand_knn_15k,
        # "10k knn": pdrand_knn_10k,
        "5k knn": pdrand_knn_5k,
        "2.5k knn": pdrand_knn_2p5k,
        # "1k knn": pdrand_knn_1k,
        # "500 knn": pdrand_knn_500,
    }
    # plot_compare_top(te, calibration, None, random, "totals", "../img/puddlerand_knn_size", outer=30, ylim=[-45, -25]) #None)#
    calibration = {
        # "15k knn(laplace)": pdrand_laplace_knn_test1_15k,
        # "10k knn(laplace)": pdrand_laplace_knn_test1_10k,
        "5k knn(laplace)": pdrand_laplace_knn_test1_5k,
        "2.5k knn(laplace)": pdrand_laplace_knn_test1_2p5k,
        "1k knn(laplace)": pdrand_laplace_knn_test1_1k,
        "500 knn(laplace)": pdrand_laplace_knn_test1_500,
    }
    plot_compare_top(te, calibration, None, [], "totals", "../img/puddlerand_knn_laplace_size", outer=30, ylim=[-37.5, -27.5]) #None)#
    calibration = {
        # "15k network(laplace)": pdrand_scale_laplace_network_15k,
        # "10k network(laplace)": pdrand_scale_laplace_network_10k,
        "5k network(laplace)": pdrand_scale_laplace_network_5k,
        "2.5k network(laplace)": pdrand_scale_laplace_network_2p5k,
        "1k network(laplace)": pdrand_scale_laplace_network_1k,
        "500 network(laplace)": pdrand_scale_laplace_network_500,
    }
    # plot_compare_top(te, calibration, None, random, "totals", "../img/puddlerand_network_laplace_size", outer=30, ylim=[-7000, -25]) #None)#
    calibration = {
        "15k network": pdrand_scale_network_15k,
        # "10k network": pdrand_scale_network_10k,
        "5k network": pdrand_scale_network_5k,
        "2.5k network": pdrand_scale_network_2p5k,
        # "1k network": pdrand_scale_network_1k,
        # "500 network": pdrand_scale_network_500,
    }
    # plot_compare_top(te, calibration, None, random, "totals", "../img/puddlerand_network_size", outer=30, ylim=[-7000, -25]) #None)#

# def sweep_model():
#     k3_close_cms = {
#     }
#     te = {"true": pdrand_true}
#     plot_generation(te, k3_close_cms, ranges, "total-return", "../img/puddlehard_model", outer=30, sparse_reward=-1, max_len=400)
#     # plot_each_run(te, cms, "total-reward", "../img/v2_model_run", outer=10, sparse_reward=-1, max_len=1000)

def data_density():
    datasets = {
        "return -45": "../../data/hyperparam_v5/puddlehard/offline_data/esarsa/step10k_env/return-45/param_0/",
    }
    dimension = {
        0: "x",
        1: "y",
    }
    group = {"xy": [0, 1]}
    key="new state"
    for i in range(10):
        run = "traces-{}".format(i)
        plot_dataset(datasets, key, dimension, group, run, "../img/puddlehard_data_density")

def return_per_ep():
    from numpy import genfromtxt
    paths = [
        "../../data/hyperparam_v5/puddlerand/offline_learning/knn/learning/k3_laplace/timeout20k/dqn/step5k_env/data_optimal/drop0/sweep_rep1/"
    ]
    for path in paths:
        plt.figure()
        params = os.listdir(path)
        for parm in params:
            files = os.listdir(os.path.join(path, parm))
            rtns = []
            for f in files:
                if "returns" in f:
                    rtn = genfromtxt(os.path.join(path, parm, f), delimiter=',')
                    rtns.append(rtn[1:20])
            rtns = np.array(rtns)
            print(rtns.mean(axis=0), rtns.mean(axis=0).sum())
            plt.plot(rtns.mean(axis=0))
        plt.savefig("{}.png".format(paths.index(path)))

def dqn():
    calibration = {
        "5k knn(laplace)": pdrand_laplace_knn_test1_5k_dqn,
    }
    random = np.array(pdrand_rnd) % 24
    te = {"true": pdrand_true_dqn}
    plot_compare_top(te, calibration, None, random, "totals", "../img/puddlerand_dqn", outer=30, res_scale=-1)

if __name__ == '__main__':
    ranges = [0, 0.05, 0.1, 0.2, 0.5, 0.7, 0.9]
    top_param()
    # dqn()
    # sweep_model()
    # data_density()
    # return_per_ep()