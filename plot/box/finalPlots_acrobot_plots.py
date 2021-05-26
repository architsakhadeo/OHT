import os
import sys
cwd = os.getcwd()
sys.path.insert(0, cwd+'/../..')
from plot.box.utils_plot import *
from plot.box.paths_acrobot_finalPlots import *

def top_param():
    # PLOT 1
    calibration = {
        "Calibration": ac_knnlaplace_optim_5k,
        "Calibration (raw)": ac_knnraw_optim_5k,
        "NN Calibration (raw)": ac_networkscaledraw_optim_5k,
        "NN Calibration (laplace)": ac_networkscaledlaplace_optim_5k,
    }
    random = ac_rnd
    true = {"true": ac_true}
    fqi = {"FQI": ac_fqi_tc}
    #cem = {"cem": ac_cem}
    # plot_compare_top(true, calibration, fqi, random, "totals", "../img/finalPlots/acrobot/plot1/plot1_models",
    #                  outer=30, res_scale=-1, ylim=[[100, 250], []], ylabel="Step per episode", right_ax=["NN Calibration (laplace)", "FQI", "Random selection"],
    #                  label_ncol=7)


    # PLOT 2
    calibration = {
        "Size = 5000": ac_knnlaplace_avg_5k_new,
        "Size = 2500": ac_knnlaplace_avg_2500_new,
        "Size = 1000": ac_knnlaplace_avg_1k_new,
        "Size = 500": ac_knnlaplace_avg_500_new,
    }
    true = {"true": ac_true}
    plot_compare_top(true, calibration, None, [], "totals", "../img/finalPlots/acrobot/plot2/plot2_size",
                     outer=30, res_scale=-1, ylim=[], ylabel="Step per episode", right_ax=[],
                     label_ncol=3)

    calibration = {
        "Optimal policy": ac_knnlaplace_optim_5k_new,
        "Average policy": ac_knnlaplace_avg_5k_new,
        "Bad policy": ac_knnlaplace_bad_5k_new,
    }
    plot_compare_top(true, calibration, None, [], "totals", "../img/finalPlots/acrobot/plot2/plot2_policy",
                     outer=30, res_scale=-1, ylim=[], ylabel="Step per episode", right_ax=[],
                     label_ncol=3)

    info = {
        "true performance": {"color": "black", "style": "--"},
        "Optimal policy": {"color": c_dict["Optimal policy"], "style": "-"},
        "Average policy": {"color": c_dict["Average policy"], "style": "-"},
        "Bad policy": {"color": c_dict["Bad policy"], "style": "-"},

        "Size = 5000": {"color": c_dict["Size = 5000"], "style": "-"},
        "Size = 2500": {"color": c_dict["Size = 2500"], "style": "-"},
        "Size = 1000": {"color": c_dict["Size = 1000"], "style": "-"},
        "Size = 500": {"color": c_dict["Size = 500"], "style": "-"},
    }
    draw_label(info, "../img/finalPlots/acrobot/plot2/plot2_labels", 5)

    # PLOT 3
    paths = {
        "Calibration": ac_knnlaplace_optim_5k,
        "Esarsa transfer (true)": acshift_esarsa_true_trans,
        "Esarsa transfer (calibration)": acshift_esarsa_calibration_trans,
    }
    fqi = {"FQI": acshift_fqi_tc_optim_5k}
    # plot_learning_perform(paths, "totals", "../img/finalPlots/acrobot/plot3/plot3_shift", res_scale=-1,
    #                       ylabel="Step per episode", right_ax=[],
    #                       label_ncol=5,
    #                       fqi=fqi)



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
        "bad policy": ac_knnlaplace_subsuboptim_5k_plot3
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
