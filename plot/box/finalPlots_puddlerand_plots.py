import os
import sys
cwd = os.getcwd()
sys.path.insert(0, cwd+'/../..')
from plot.box.utils_plot import *
from plot.box.paths_puddlerand_finalPlots import *

def top_param():
    # PLOT 1
    calibration = {
        "Calibration-KNN": pr_knnlaplace_optim_5k,
        # "Calibration (raw)": pr_knnraw_optim_5k,
        "Calibration-NN": pr_networkscaledlaplace_optim_5k,
        # "NN Calibration (raw)": pr_networkscaledraw_optim_5k,
    }
    random = pr_rnd
    true = {"true": pr_true}
    fqi = {"FQI": pr_fqi_nn}
    #cem = {"cem": pr_cem}
    # plot_compare_top(true, calibration, fqi, random, "totals", "../img/finalPlots/puddlerand/plot1/plot1_models",
    #                  outer=30, ylim=[[-100, -20], []], ylabel="Return per episode", right_ax=["FQI", "Random"],
    #                  label_ncol=6)

    # PLOT 2
    calibration = {
        "Size = 5000": pr_knnlaplace_avg_5k_new,
        # "Size = 2500": pr_knnlaplace_avg_2500_new,
        "Size = 1000": pr_knnlaplace_avg_1k_new,
        "Size = 500": pr_knnlaplace_avg_500_new,
    }
    true = {"true": pr_true}
    # plot_compare_top(true, calibration, None, [], "totals", "../img/finalPlots/puddlerand/plot2/plot2_size",
    #                  outer=30, ylim=[[-34, -27]], ylabel="Return per episode", right_ax=[],
    #                  label_ncol=3, plot="bar")

    calibration = {
        "Optimal policy": pr_knnlaplace_optim_5k_new,
        "Medium policy": pr_knnlaplace_avg_5k_new,
        "Naive policy": pr_knnlaplace_bad_5k_new,
    }
    # plot_compare_top(true, calibration, None, [], "totals", "../img/finalPlots/puddlerand/plot2/plot2_policy",
    #                  outer=30, ylim=[[-34, -27]], ylabel="Return per episode", right_ax=[],
    #                  label_ncol=3, plot="bar")

    # PLOT Agents
    calibration = {
        "Esarsa": pr_knnlaplace_optim_5k,
        "DQN": pr_dqn_knnlaplace_optim,
        "AC": pr_actorcritic_knnlaplace_optim,
    }
    true = {
        "Esarsa": pr_true,
        "DQN": pr_dqn,
        "AC": pr_actorcritic,
    }
    plot_compare_agents(true, calibration, None, [], "totals", "../img/finalPlots/puddlerand/plot_agents",
                        outer=30, ylim=[[-3000, 100]], ylabel="Return per episode", right_ax=[],
                        label_ncol=3)

def sweep_model():
    '''
    # PLOT 2
    calibration = {
        "size = 500": pr_knnlaplace_optim_500_plot2,
        "size = 1000": pr_knnlaplace_optim_1k_plot2,
        "size = 2500": pr_knnlaplace_optim_2500_plot2,
        "size = 5000": pr_knnlaplace_optim_5k_plot2
    }
    true = {"true": pr_true}
    plot_generation(true, calibration, ranges, "totals", "../img/finalPlots/puddlerand/plot2/plot2_boxplot", outer=30, sparse_reward=-1, max_len=1000)
    '''
    
    # PLOT 3
    calibration = {
        "optimal policy": pr_knnlaplace_optim_5k_plot3,
        "average policy": pr_knnlaplace_suboptim_5k_plot3,
        "Naive policy": pr_knnlaplace_subsuboptim_5k_plot3
    }
    true = {"true": pr_true}
    plot_generation(true, calibration, ranges, "totals", "../img/finalPlots/puddlerand/plot3/plot3_boxplot", outer=30, sparse_reward=-1, max_len=1000)
    


def data_density():
    datasets = {
        "random": "../../data/hyperparam_v4/puddle/offline_data/esarsa/step10k_env/random/param_0/",
        "return -320": "../../data/hyperparam_v4/puddle/offline_data/esarsa/step10k_env/return-320/param_0/",
        "return -40": "../../data/hyperparam_v4/puddle/offline_data/esarsa/step10k_env/return-40/param_0/",
    }
    dimension = {
        0: "x",
        1: "y",
    }
    group = {"xy": [0, 1]}
    key="new state"
    for i in range(10):
        run = "traces-{}".format(i)
        plot_dataset(datasets, key, dimension, group, run, "../img/puddle_data_density")


if __name__ == '__main__':
    ranges = [0]
    top_param()
    # sweep_model()
    #data_density()
