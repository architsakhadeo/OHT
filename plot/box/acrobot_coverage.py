import os
import sys
import numpy
cwd = os.getcwd()
sys.path.insert(0, cwd+'/../..')
from plot.box.utils_data import *
from plot.box.utils_plot import *
from plot.box.paths_acrobot import *

# def plot_generation(te, cms, title, ylim=None):
#
#     te_data = loading_pessimistic(te)
#     # te_rank = ranking_allruns(te_data)["true"]
#     te_data = te_data["true"]
#
#     te_thrd = []
#     for perc in ranges:
#         te_thrd.append(percentile_avgeraged_run(te_data, perc))
#
#     cms_data = loading_pessimistic(cms)
#     filtered = {}
#     models_rank = ranking_allruns(cms_data)
#     for model in cms_data.keys():
#         ranks = models_rank[model]
#
#         filtered[model] = []
#         for perc in ranges:
#             target = percentile(ranks, perc)
#             data = [te_data[item[0]][item[1]] for item in target]
#             filtered[model].append(data)
#
#     plot_boxs(filtered, te_thrd, labels, title, ylim=ylim)


def sweep_model():
    cms = [{
        "k1": data2500_eps0_k1_p0,
        "k3": data2500_eps0_k3_p0,
        "k5": data2500_eps0_k5_p0,
    }, {
        "k1": data2500_eps10_k1_p0,
        "k3": data2500_eps10_k3_p0,
        "k5": data2500_eps10_k5_p0,
    }, {
        "k1": data2500_eps25_k1_p0,
        "k3": data2500_eps25_k3_p0,
        "k5": data2500_eps25_k5_p0,
    }, {
        "k1": data2500_eps50_k1_p0,
        "k3": data2500_eps50_k3_p0,
        "k5": data2500_eps50_k5_p0,
    }, {
        "k1": data2500_eps75_k1_p0,
        "k3": data2500_eps75_k3_p0,
        "k5": data2500_eps75_k5_p0,
    }, {
        "k1": data2500_eps100_k1_p0,
        "k3": data2500_eps100_k3_p0,
        "k5": data2500_eps100_k5_p0,
    }, {
        "k1": data5k_eps0_k1_p0,
        "k3": data5k_eps0_k3_p0,
        "k5": data5k_eps0_k5_p0,
    }, {
        "k1": data5k_eps10_k1_p0,
        "k3": data5k_eps10_k3_p0,
        "k5": data5k_eps10_k5_p0,
    }, {
        "k1": data5k_eps25_k1_p0,
        "k3": data5k_eps25_k3_p0,
        "k5": data5k_eps25_k5_p0,
    }, {
        "k1": data5k_eps50_k1_p0,
        "k3": data5k_eps50_k3_p0,
        "k5": data5k_eps50_k5_p0,
    }, {
        "k1": data5k_eps75_k1_p0,
        "k3": data5k_eps75_k3_p0,
        "k5": data5k_eps75_k5_p0,
    }, {
        "k1": data5k_eps100_k1_p0,
        "k3": data5k_eps100_k3_p0,
        "k5": data5k_eps100_k5_p0,
    }, {
        "k1": data10k_eps0_k1_p0,
        "k3": data10k_eps0_k3_p0,
        "k5": data10k_eps0_k5_p0,
    }, {
        "k1": data10k_eps10_k1_p0,
        "k3": data10k_eps10_k3_p0,
        "k5": data10k_eps10_k5_p0,
    }, {
        "k1": data10k_eps25_k1_p0,
        "k3": data10k_eps25_k3_p0,
        "k5": data10k_eps25_k5_p0,
    }, {
        "k1": data10k_eps50_k1_p0,
        "k3": data10k_eps50_k3_p0,
        "k5": data10k_eps50_k5_p0,
    }, {
        "k1": data10k_eps75_k1_p0,
        "k3": data10k_eps75_k3_p0,
        "k5": data10k_eps75_k5_p0,
    }, {
        "k1": data10k_eps100_k1_p0,
        "k3": data10k_eps100_k3_p0,
        "k5": data10k_eps100_k5_p0,
    }]
    te = {"true": true_env}
    listOfPlots = ["../../img/sweep_model/step2500/epsilon0/sweepmodel_step2500_epsilon0",
                   "../../img/sweep_model/step2500/epsilon10/sweepmodel_step2500_epsilon10",
                   "../../img/sweep_model/step2500/epsilon25/sweepmodel_step2500_epsilon25",
                   "../../img/sweep_model/step2500/epsilon50/sweepmodel_step2500_epsilon50",
                   "../../img/sweep_model/step2500/epsilon75/sweepmodel_step2500_epsilon75",
                   "../../img/sweep_model/step2500/epsilon100/sweepmodel_step2500_epsilon100",
                   "../../img/sweep_model/step5k/epsilon0/sweepmodel_step5k_epsilon0",
                   "../../img/sweep_model/step5k/epsilon10/sweepmodel_step5k_epsilon10",
                   "../../img/sweep_model/step5k/epsilon25/sweepmodel_step5k_epsilon25",
                   "../../img/sweep_model/step5k/epsilon50/sweepmodel_step5k_epsilon50",
                   "../../img/sweep_model/step5k/epsilon75/sweepmodel_step5k_epsilon75",
                   "../../img/sweep_model/step5k/epsilon100/sweepmodel_step5k_epsilon100",
                   "../../img/sweep_model/step10k/epsilon0/sweepmodel_step10k_epsilon0",
                   "../../img/sweep_model/step10k/epsilon10/sweepmodel_step10k_epsilon10",
                   "../../img/sweep_model/step10k/epsilon25/sweepmodel_step10k_epsilon25",
                   "../../img/sweep_model/step10k/epsilon50/sweepmodel_step10k_epsilon50",
                   "../../img/sweep_model/step10k/epsilon75/sweepmodel_step10k_epsilon75",
                   "../../img/sweep_model/step10k/epsilon100/sweepmodel_step10k_epsilon100",
                   ]
    for i in range(len(listOfPlots)):
        print(i)
        plot_generation(te, cms[i], ranges, listOfPlots[i])
    

def sweep_coverage():
    '''
    cms = [{
        "eps0": data2500_eps0_k1_p0,
        "eps10": data2500_eps10_k1_p0,
        "eps25": data2500_eps25_k1_p0,
        "eps50": data2500_eps50_k1_p0,
        "eps75": data2500_eps75_k1_p0,
        "eps100": data2500_eps100_k1_p0,
    }, {
        "eps0": data2500_eps0_k3_p0,
        "eps10": data2500_eps10_k3_p0,
        "eps25": data2500_eps25_k3_p0,
        "eps50": data2500_eps50_k3_p0,
        "eps75": data2500_eps75_k3_p0,
        "eps100": data2500_eps100_k3_p0,
    }, {
        "eps0": data2500_eps0_k5_p0,
        "eps10": data2500_eps10_k5_p0,
        "eps25": data2500_eps25_k5_p0,
        "eps50": data2500_eps50_k5_p0,
        "eps75": data2500_eps75_k5_p0,
        "eps100": data2500_eps100_k5_p0,
    }, {
        "eps0": data5k_eps0_k1_p0,
        "eps10": data5k_eps10_k1_p0,
        "eps25": data5k_eps25_k1_p0,
        "eps50": data5k_eps50_k1_p0,
        "eps75": data5k_eps75_k1_p0,
        "eps100": data5k_eps100_k1_p0,
    }, {
        "eps0": data5k_eps0_k3_p0,
        "eps10": data5k_eps10_k3_p0,
        "eps25": data5k_eps25_k3_p0,
        "eps50": data5k_eps50_k3_p0,
        "eps75": data5k_eps75_k3_p0,
        "eps100": data5k_eps100_k3_p0,
    }, {
        "eps0": data5k_eps0_k5_p0,
        "eps10": data5k_eps10_k5_p0,
        "eps25": data5k_eps25_k5_p0,
        "eps50": data5k_eps50_k5_p0,
        "eps75": data5k_eps75_k5_p0,
        "eps100": data5k_eps100_k5_p0,
    }, {
        "eps0": data10k_eps0_k1_p0,
        "eps10": data10k_eps10_k1_p0,
        "eps25": data10k_eps25_k1_p0,
        "eps50": data10k_eps50_k1_p0,
        "eps75": data10k_eps75_k1_p0,
        "eps100": data10k_eps100_k1_p0,
    }, {
        "eps0": data10k_eps0_k3_p0,
        "eps10": data10k_eps10_k3_p0,
        "eps25": data10k_eps25_k3_p0,
        "eps50": data10k_eps50_k3_p0,
        "eps75": data10k_eps75_k3_p0,
        "eps100": data10k_eps100_k3_p0,
    }, {
        "eps0": data10k_eps0_k5_p0,
        "eps10": data10k_eps10_k5_p0,
        "eps25": data10k_eps25_k5_p0,
        "eps50": data10k_eps50_k5_p0,
        "eps75": data10k_eps75_k5_p0,
        "eps100": data10k_eps100_k5_p0,
    }]
    te = {"true": true_env}
    listOfPlots = ["../../img/sweep_coverage/step2500/k1/sweepcoverage_step2500_k1",
                   "../../img/sweep_coverage/step2500/k3/sweepcoverage_step2500_k3",
                   "../../img/sweep_coverage/step2500/k5/sweepcoverage_step2500_k5",
                   "../../img/sweep_coverage/step5k/k1/sweepcoverage_step5k_k1",
                   "../../img/sweep_coverage/step5k/k3/sweepcoverage_step5k_k3",
                   "../../img/sweep_coverage/step5k/k5/sweepcoverage_step5k_k5",
                   "../../img/sweep_coverage/step10k/k1/sweepcoverage_step10k_k1",
                   "../../img/sweep_coverage/step10k/k3/sweepcoverage_step10k_k3",
                   "../../img/sweep_coverage/step10k/k5/sweepcoverage_step10k_k5"
                   ]
    '''
    cms = [{
        "eps0": data2500_eps0_k1_p0,
        "eps10": data2500_eps10_k1_p0,
        "eps25": data2500_eps25_k1_p0,
        "eps100": data2500_eps100_k1_p0,
    }, {
        "eps0": data2500_eps0_k3_p0,
        "eps10": data2500_eps10_k3_p0,
        "eps25": data2500_eps25_k3_p0,
        "eps100": data2500_eps100_k3_p0,
    }, {
        "eps0": data2500_eps0_k5_p0,
        "eps10": data2500_eps10_k5_p0,
        "eps25": data2500_eps25_k5_p0,
        "eps100": data2500_eps100_k5_p0,
    }, {
        "eps0": data5k_eps0_k1_p0,
        "eps10": data5k_eps10_k1_p0,
        "eps25": data5k_eps25_k1_p0,
        "eps100": data5k_eps100_k1_p0,
    }, {
        "eps0": data5k_eps0_k3_p0,
        "eps10": data5k_eps10_k3_p0,
        "eps25": data5k_eps25_k3_p0,
        "eps100": data5k_eps100_k3_p0,
    }, {
        "eps0": data5k_eps0_k5_p0,
        "eps10": data5k_eps10_k5_p0,
        "eps25": data5k_eps25_k5_p0,
        "eps100": data5k_eps100_k5_p0,
    }, {
        "eps0": data10k_eps0_k1_p0,
        "eps10": data10k_eps10_k1_p0,
        "eps25": data10k_eps25_k1_p0,
        "eps100": data10k_eps100_k1_p0,
    }, {
        "eps0": data10k_eps0_k3_p0,
        "eps10": data10k_eps10_k3_p0,
        "eps25": data10k_eps25_k3_p0,
        "eps100": data10k_eps100_k3_p0,
    }, {
        "eps0": data10k_eps0_k5_p0,
        "eps10": data10k_eps10_k5_p0,
        "eps25": data10k_eps25_k5_p0,
        "eps100": data10k_eps100_k5_p0,
    }]
    te = {"true": true_env}
    listOfPlots = ["../../img/sweep_coverage/step2500/k1/sweepcoverage_step2500_k1",
                   "../../img/sweep_coverage/step2500/k3/sweepcoverage_step2500_k3",
                   "../../img/sweep_coverage/step2500/k5/sweepcoverage_step2500_k5",
                   "../../img/sweep_coverage/step5k/k1/sweepcoverage_step5k_k1",
                   "../../img/sweep_coverage/step5k/k3/sweepcoverage_step5k_k3",
                   "../../img/sweep_coverage/step5k/k5/sweepcoverage_step5k_k5",
                   "../../img/sweep_coverage/step10k/k1/sweepcoverage_step10k_k1",
                   "../../img/sweep_coverage/step10k/k3/sweepcoverage_step10k_k3",
                   "../../img/sweep_coverage/step10k/k5/sweepcoverage_step10k_k5"
                   ]
    for i in range(len(listOfPlots)):
        print(i)
        plot_generation(te, cms[i], ranges, listOfPlots[i])

def sweep_datasize():
    cms = [{
        "2.5k": data2500_eps0_k1_p0,
        "5k": data5k_eps0_k1_p0,
        "10k": data10k_eps0_k1_p0,
    }, {
        "2.5k": data2500_eps0_k3_p0,
        "5k": data5k_eps0_k3_p0,
        "10k": data10k_eps0_k3_p0,
    }, {
        "2.5k": data2500_eps0_k5_p0,
        "5k": data5k_eps0_k5_p0,
        "10k": data10k_eps0_k5_p0,
    },{
        "2.5k": data2500_eps10_k1_p0,
        "5k": data5k_eps10_k1_p0,
        "10k": data10k_eps10_k1_p0,
    }, {
        "2.5k": data2500_eps10_k3_p0,
        "5k": data5k_eps10_k3_p0,
        "10k": data10k_eps10_k3_p0,
    }, {
        "2.5k": data2500_eps10_k5_p0,
        "5k": data5k_eps10_k5_p0,
        "10k": data10k_eps10_k5_p0,
    }, {
        "2.5k": data2500_eps25_k1_p0,
        "5k": data5k_eps25_k1_p0,
        "10k": data10k_eps25_k1_p0,
    }, {
        "2.5k": data2500_eps25_k3_p0,
        "5k": data5k_eps25_k3_p0,
        "10k": data10k_eps25_k3_p0,
    }, {
        "2.5k": data2500_eps25_k5_p0,
        "5k": data5k_eps25_k5_p0,
        "10k": data10k_eps25_k5_p0,
    }, {
        "2.5k": data2500_eps50_k1_p0,
        "5k": data5k_eps50_k1_p0,
        "10k": data10k_eps50_k1_p0,
    }, {
        "2.5k": data2500_eps50_k3_p0,
        "5k": data5k_eps50_k3_p0,
        "10k": data10k_eps50_k3_p0,
    }, {
        "2.5k": data2500_eps50_k5_p0,
        "5k": data5k_eps50_k5_p0,
        "10k": data10k_eps50_k5_p0,
    }, {
        "2.5k": data2500_eps75_k1_p0,
        "5k": data5k_eps75_k1_p0,
        "10k": data10k_eps75_k1_p0,
    }, {
        "2.5k": data2500_eps75_k3_p0,
        "5k": data5k_eps75_k3_p0,
        "10k": data10k_eps75_k3_p0,
    }, {
        "2.5k": data2500_eps75_k5_p0,
        "5k": data5k_eps75_k5_p0,
        "10k": data10k_eps75_k5_p0,
    }, {
        "2.5k": data2500_eps100_k1_p0,
        "5k": data5k_eps100_k1_p0,
        "10k": data10k_eps100_k1_p0,
    }, {
        "2.5k": data2500_eps100_k3_p0,
        "5k": data5k_eps100_k3_p0,
        "10k": data10k_eps100_k3_p0,
    }, {
        "2.5k": data2500_eps100_k5_p0,
        "5k": data5k_eps100_k5_p0,
        "10k": data10k_eps100_k5_p0,
    }]
    te = {"true": true_env}
    listOfPlots = ["../../img/sweep_datasize/k1/epsilon0/sweepdatasize_k1_epsilon0",
                   "../../img/sweep_datasize/k3/epsilon0/sweepdatasize_k3_epsilon0",
                   "../../img/sweep_datasize/k5/epsilon0/sweepdatasize_k5_epsilon0",
                   "../../img/sweep_datasize/k1/epsilon10/sweepdatasize_k1_epsilon10",
                   "../../img/sweep_datasize/k3/epsilon10/sweepdatasize_k3_epsilon10",
                   "../../img/sweep_datasize/k5/epsilon10/sweepdatasize_k5_epsilon10",
                   "../../img/sweep_datasize/k1/epsilon25/sweepdatasize_k1_epsilon25",
                   "../../img/sweep_datasize/k3/epsilon25/sweepdatasize_k3_epsilon25",
                   "../../img/sweep_datasize/k5/epsilon25/sweepdatasize_k5_epsilon25",
                   "../../img/sweep_datasize/k1/epsilon50/sweepdatasize_k1_epsilon50",
                   "../../img/sweep_datasize/k3/epsilon50/sweepdatasize_k3_epsilon50",
                   "../../img/sweep_datasize/k5/epsilon50/sweepdatasize_k5_epsilon50",
                   "../../img/sweep_datasize/k1/epsilon75/sweepdatasize_k1_epsilon75",                   
                   "../../img/sweep_datasize/k3/epsilon75/sweepdatasize_k3_epsilon75",
                   "../../img/sweep_datasize/k5/epsilon75/sweepdatasize_k5_epsilon75",
                   "../../img/sweep_datasize/k1/epsilon100/sweepdatasize_k1_epsilon100",
                   "../../img/sweep_datasize/k3/epsilon100/sweepdatasize_k3_epsilon100",
                    "../../img/sweep_datasize/k5/epsilon100/sweepdatasize_k5_epsilon100"
                   ]
    for i in range(len(listOfPlots)):
        print(i)
        plot_generation(te, cms[i], ranges, listOfPlots[i])

ranges = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
labels = ["0%\n(top param)", "10%", "20%", "30%", "40%", "50%"]
#ranges = [[0, 0.3], [0.3, 0.7], [0.7, 1.0]]

#sweep_model()
#sweep_coverage()
sweep_datasize()