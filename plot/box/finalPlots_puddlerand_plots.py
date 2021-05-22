import os
import sys
cwd = os.getcwd()
sys.path.insert(0, cwd+'/../..')
from plot.box.utils_plot import *
from plot.box.paths_puddlerand_finalPlots import *

def top_param():
    
    '''
    # PLOT 1

    calibration = {
        "KNN (laplace)": pr_knnlaplace_optim_5k_plot1
    }
    random = pr_rnd
    true = {"true": pr_true}
    #fqi = {"fqi": pr_fqi}
    #cem = {"cem": pr_cem}
    plot_compare_top(true, calibration, None, random, "totals", "../img/finalPlots/puddlerand/plot1/plot1_boxplot", outer=30)#, ylim=[-200, 0])
    '''
    
    '''
    # PLOT 4

    calibration = {
        "KNN (laplace)": pr_knnlaplace_optim_5k_plot4,
        "network (raw)": pr_networkscaledraw_optim_5k_plot4
    }
    random = pr_rnd
    true = {"true": pr_true}
    #fqi = {"fqi": pr_fqi}
    #cem = {"cem": pr_cem}
    plot_compare_top(true, calibration, None, random, "totals", "../img/finalPlots/puddlerand/plot4/plot4_scaled_raw_boxplot", outer=30)#, ylim=[-200, 0])
    '''

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
        "bad policy": pr_knnlaplace_subsuboptim_5k_plot3
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
    #top_param()
    sweep_model()
    #data_density()
