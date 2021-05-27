import os
import sys
cwd = os.getcwd()
sys.path.insert(0, cwd+'/../..')
from plot.box.utils_plot import *
from plot.box.paths_acrobot_finalPlots import *

def top_param():
    # # Test code
    # calibration = {
    #     "Calibration-KNN": ac_knnlaplace_optim_5k,
    #     "Calibration-NN": ac_networkscaledlaplace_optim_5k,
    # }
    # random = ac_rnd
    # true = {"true": ac_true}
    # plot_compare_top(true, calibration, None, [], "totals", "../img/finalPlots/test",
    #                  outer=30, res_scale=-1, ylim=[[100, 250], []], ylabel="Step per episode", right_ax=["Calibration-NN", "FQI", "Random"],
    #                  label_ncol=7)

    # PLOT 1
    calibration = {
        "Calibration-KNN": ac_knnlaplace_optim_5k,
        # "KNN (raw)": ac_knnraw_optim_5k,
        # "NN (raw)": ac_networkscaledraw_optim_5k,
        "Calibration-NN": ac_networkscaledlaplace_optim_5k,
    }
    random = ac_rnd
    true = {"true": ac_true}
    fqi = {"FQI": ac_fqi_tc}
    #cem = {"cem": ac_cem}
    # plot_compare_top(true, calibration, fqi, random, "totals", "../img/finalPlots/acrobot/plot1/plot1_models",
    #                  outer=30, res_scale=-1, ylim=[[100, 250], []], ylabel="Step per episode", right_ax=["Calibration-NN", "FQI", "Random"],
    #                  label_ncol=6)


    # PLOT 2
    calibration = {
        "Size = 5000": ac_knnlaplace_avg_5k_new,
        # "Size = 2500": ac_knnlaplace_avg_2500_new,
        "Size = 1000": ac_knnlaplace_avg_1k_new,
        "Size = 500": ac_knnlaplace_avg_500_new,
    }
    true = {"true": ac_true}
    # plot_compare_top(true, calibration, None, [], "totals", "../img/finalPlots/acrobot/plot2/plot2_size",
    #                  outer=30, res_scale=-1, ylim=[[110, 115]], ylabel="Step per episode (Median)", right_ax=[],
    #                  label_ncol=3, plot="bar")


    calibration = {
        "Optimal policy": ac_knnlaplace_optim_5k_new,
        "Medium policy": ac_knnlaplace_avg_5k_new,
        "Naive policy": ac_knnlaplace_bad_5k_new,
    }
    # plot_compare_top(true, calibration, None, [], "totals", "../img/finalPlots/acrobot/plot2/plot2_policy",
    #                  outer=30, res_scale=-1, ylim=[[110, 123]], ylabel="Step per episode (Median)", right_ax=[],
    #                  label_ncol=3, plot="bar")

    # PLOT 3
    paths = {
        "Calibration": acshift_knnlaplace_optim_5k,
        # "Calibration (50k)": acshift_knnlaplace_optim_5k_50kstep,
        "Esarsa transfer (true)": acshift_esarsa_true_trans,
        "Esarsa transfer (calibration)": acshift_esarsa_calibration_trans,
    }
    fqi = {"FQI": acshift_fqi_tc_optim_5k}
    true = {"true": acshift_true}
    # plot_learning_perform(paths, "totals", "../img/finalPlots/acrobot/plot3/plot3_shift", res_scale=-1, yscale="log", #ylim=[0, 15000],
    #                       ylabel="Step per episode", right_ax=[],
    #                       label_ncol=2,
    #                       fqi=fqi, true_perf=true)

    # PLOT Agents
    calibration = {
        "Esarsa": ac_knnlaplace_optim_5k,
        "DQN": ac_dqn_knnlaplace_optim,
        "AC": ac_actorcritic_knnlaplace_optim,
    }
    true = {
        "Esarsa": ac_true,
        "DQN": ac_dqn,
        "AC": ac_actorcritic,
    }
    plot_compare_agents(true, calibration, None, [], "totals", "../img/finalPlots/acrobot/plot_agents",
                        outer=30, res_scale=-1, ylim=[[0, 500]], ylabel="Step per episode", right_ax=[],
                        label_ncol=3)

    # info = {
    #
    #     "Optimal policy": {"color": c_dict["Optimal policy"], "style": "-"},
    #     "Medium policy": {"color": c_dict["Medium policy"], "style": "-"},
    #     "Naive policy": {"color": c_dict["Naive policy"], "style": "-"},
    #
    #     "Size = 5000": {"color": c_dict["Size = 5000"], "style": "-"},
    #     "Size = 2500": {"color": c_dict["Size = 2500"], "style": "-"},
    #     "Size = 1000": {"color": c_dict["Size = 1000"], "style": "-"},
    #     "Size = 500": {"color": c_dict["Size = 500"], "style": "-"},
    # }
    # draw_label(info, "../img/finalPlots/acrobot/plot2/plot2_labels", 5)

def sweep_model():
    '''
    # PLOT 2
    calibration = {
        "size = 500": ac_knnlaplace_optim_500_plot2,
        "size = 1000": ac_knnlaplace_optim_1k_plot2,
        "size = 2500": ac_knnlaplace_optim_2500_plot2,
        "size = 5000": ac_knnlaplace_optim_5k_plot2
    }
    true = {"true": ac_true}
    plot_generation(true, calibration, ranges, "totals", "../img/finalPlots/acrobot/plot2/plot2_boxplot_testing", outer=30, sparse_reward=-1, max_len=1000, res_scale=-1)
    '''
    '''
    # PLOT 3
    calibration = {
        "optimal policy": ac_knnlaplace_optim_5k_plot3,
        "average policy": ac_knnlaplace_suboptim_5k_plot3,
        "Naive policy": ac_knnlaplace_subsuboptim_5k_plot3
    }
    true = {"true": ac_true}
    plot_generation(true, calibration, ranges, "totals", "../img/finalPlots/acrobot/plot3/plot3_boxplot_testing", outer=30, sparse_reward=-1, max_len=1000, res_scale=-1)
    '''


def data_density():
    datasets = {
        "15k": "../../data/hyperparam_v5/acrobot/offline_data/true_restarts/esarsa/step15k/optimalfixed_eps0/param_0/",
        "10k": "../../data/hyperparam_v5/acrobot/offline_data/true_restarts/esarsa/step10k/optimalfixed_eps0/param_0/",
        "5k": "../../data/hyperparam_v5/acrobot/offline_data/true_restarts/esarsa/step5k/optimalfixed_eps0/param_0/",
    }
    dimension = {
        0: "0",
        1: "1",
        2: "2",
        3: "3",
        4: "4",
        5: "5",
    }
    group = {"01": [0, 1], "23": [2, 3], "45": [4, 5]}
    key="new state"
    for i in range(10):
        run = "traces-{}".format(i)
        plot_dataset(datasets, key, dimension, group, run, "../img/data_density")

if __name__ == '__main__':
    ranges = [0]
    top_param()
    #sweep_model()
    # data_density()
    #top_param()
