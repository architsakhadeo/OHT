import os
import sys
cwd = os.getcwd()
sys.path.insert(0, cwd+'/../..')
from plot.box.utils_plot import *
from plot.box.paths_puddle import *

def top_param():
    calibration = {
        "knn": pdrand_knn,
        "knn(laplace)": pdrand_laplace_knn_test1,
        "network": pdrand_network,
        "network(laplace)": pdrand_laplace_network,
        "network(scaled)": pdrand_scale_network,
        "network(scaled+laplace)": pdrand_scale_laplace_network,
    }
    random = pdrand_rnd
    te = {"true": pdrand_true}
    plot_compare_top(te, calibration, None, random, "totals", "../img/puddlerand_top_zoomin", outer=30, ylim=[-60, -25]) #None)#

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
        "../../data/hyperparam_v5/puddlerand/online_learning/dqn/temp"
    ]
    for path in paths:
        plt.figure()
        params = os.listdir(path)
        for parm in params:
            files = os.listdir(os.path.join(path, parm))
            rtns = []
            total_ep = []
            for f in files:
                if "returns" in f:
                    rtn = genfromtxt(os.path.join(path, parm, f), delimiter=',')
                    if len(rtn)>30:
                        rtns.append(rtn[1:31])
                        total_ep.append(len(rtn)-1)
            rtns = np.array(rtns)
            # print(rtns)
            # print(total_ep)
            # print()
            if len(rtns)>1:
                print(os.path.join(path, parm), rtns.mean(axis=0), rtns.mean(axis=0).sum(), np.array(total_ep).mean())
            plt.plot(rtns.mean(axis=0))
        plt.savefig("{}.png".format(paths.index(path)))



if __name__ == '__main__':
    ranges = [0, 0.05, 0.1, 0.2, 0.5, 0.7, 0.9]
    # top_param()
    # sweep_model()
    # data_density()
    return_per_ep()