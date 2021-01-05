#!/usr/bin/env python
# coding: utf-8

# # Plot Comparison Between Algorithms

# Expects the input data to contain CSV files containing episode lengths



import os
import numpy as np
import itertools as it
from loadFromEpisodeLengths import load_data
from loadFromEpisodeLengths import convert_data
from loadFromEpisodeLengths import transform_data
from stats import getMean, getMedian, getBest, getWorst, getConfidenceIntervalOfMean, getRegion
import matplotlib.pyplot as plt

def plotMean(xAxis, data, color, alg):
    mean = getMean(data)
    plt.plot(xAxis, mean, label=alg+'-mean', color=color)

# def plotMedian(xAxis, data, color):
#     median = getMedian(data)
#     plt.plot(xAxis, median, label=alg+'-median', color=color)
#
# def plotBest(xAxis, data, transformation, color):
#     best = getBest(data, transformation)
#     plt.plot(xAxis, best, label=alg+'-best', color=color)
#
# def plotWorst(xAxis, data, transformation, color):
#     worst = getWorst(data,  transformation)
#     plt.plot(xAxis, worst, label=alg+'-worst', color=color)
#
# def plotMeanAndConfidenceInterval(xAxis, data, confidence, color):
#     plotMean(xAxis, data, color=color)
#     lowerBound, upperBound = getConfidenceIntervalOfMean(data, confidence)
#     plt.fill_between(xAxis, lowerBound, upperBound, alpha=0.25, color=color)
#
# def plotMeanAndPercentileRegions(xAxis, data, lower, upper, transformation, color):
#     plotMean(xAxis, data, color)
#     lowerRun, upperRun = getRegion(data, lower, upper, transformation)
#     plt.fill_between(xAxis, lowerRun, upperRun, alpha=0.25, color=color)


def plot_img(dataPath, savePath, label, keyWords):
    algorithms = dataPath.keys()
    Data = {}
    for key in algorithms:
        if os.path.isdir(dataPath[key]) == True:
            Data[key] = load_data(dataPath[key])

    convertedData = {}

    for alg, data in Data.items():
        convertedData[alg], totalTimesteps = convert_data(alg, data)
    del Data

    plottingData = {}
    transformation = 'Average-Rewards'
    window = 2500

    for alg, data in convertedData.items():
        plottingData[alg] = transform_data(alg, data, totalTimesteps, transformation, window)

    # print('Data will be plotted for', ', '.join([k for k in plottingData.keys()]))
    # print('The stored failure timesteps are transformed to: ', transformation)
    del convertedData

    plt.figure(figsize=[10, 6])
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    count = 0
    auc = {}
    for alg in sorted(plottingData):
        data = plottingData[alg]
        lenRun = len(data[0])
        xAxis = np.array([i for i in range(1,lenRun+1)])

        if transformation == 'Average-Rewards':
            xAxis += (window-1)


        color = colors[count]
        count += 1

        plotMean(xAxis, data, color, alg)

        #plotMedian(xAxis, data, color=color)

        #plotBest(xAxis, data, transformation=transformation, color=color)

        #plotWorst(xAxis, data, transformation=transformation, color=color)

        #plotMeanAndConfidenceInterval(xAxis, data, confidence=0.95, color=color)

        #plotMeanAndPercentileRegions(xAxis, data, lower=0.025, upper=0.975, transformation=transformation, color=color)

        temp = keyWords[:]
        temp.append(alg)
        k = ",".join(sorted(temp))
        auc[k] = np.sum(data[:, data.shape[1]//2*(-1): ]) / len(data)

    del plottingData, temp, alg, count, colors

    plt.plot(np.ones(lenRun)*(-1.0/200), "--", color='grey')

    plt.xlabel('Timesteps', labelpad=35)
    plt.ylabel(transformation, rotation=0, labelpad=45)
    plt.rcParams['figure.figsize'] = [8, 5.33]
    plt.legend(loc=0)
    plt.yticks()
    plt.xticks()
    plt.tight_layout()

    yrange = -0.02
    if yrange:
        plt.ylim(yrange, 0)

    imgname = savePath+'/change='+label.strip("=")+"_"+"_".join(keyWords)
    if yrange is not None:
        imgname += "_ylim"+str(yrange)+".png"
    plt.savefig(imgname)
    plt.close()
    plt.clf()

    return auc


def plot_single_label(label, aucRec):
    savePath = "../img/label="+label+"/"

    if not os.path.isdir(savePath):
        os.makedirs(savePath)
    temp = keyWordsAll.copy()
    temp.pop(label)

    keys, values = zip(*temp.items())
    permutations_dicts = [dict(zip(keys, v)) for v in it.product(*values)]
    for keyWordsDict in permutations_dicts:
        keyWords = []
        for key,value in keyWordsDict.items():
            if type(value) == list:
                value = [str(k) for k in value]
                value = ",".join(value).strip("[").strip("]")
            keyWords.append(key+"="+str(value))
        dataPath = {}
        for path in os.listdir(basePath):
            add = True
            for kw in keyWords:
                if kw not in path:
                    add = False

            if add:
                lb = label+path.split(label)[1].split("_")[0]
                dataPath[lb] = basePath + path
        auc = plot_img(dataPath, savePath, label, keyWords)
        aucRec.update(auc)
        print("done:", label, keyWords)
    return aucRec

def summary_all():
    aucRec = {}
    for label in keyWordsAll.keys():
        aucRec = plot_single_label(label, aucRec)

    sort_auc = [[k, v] for k, v in sorted(aucRec.items(), key=lambda item: item[1])]
    with open("../img/auc.txt", "w") as f:
        for auc in sort_auc:
            f.write("{:<80} auc={:.8f} \n".format(auc[0], auc[1]))

# basePath = '../data/dqn/adam-step100k_sweep/'
# keyWordsAll = {
#     "alpha": [1e-3, 1e-4, 1e-5, 1e-6],
#     "dqn-hidden": [
#         [64, 64],
#         [128, 128],
#         [256, 256]
#     ],
#     "dqn-sync":[32, 256, 1024],
#     "dqn-batch":[16, 32, 64],
#     "buffer-size":[500, 2500, 5000]
# }

# basePath = '../data/dqn/adam-step10m_sweep/'
# keyWordsAll = {
#     "alpha": [3e-5, 1e-5, 3e-6],
#     "dqn-hidden": [
#         [64, 64],
#         [128, 128]
#     ],
#     "dqn-sync":[32],
#     "dqn-batch":[32, 64],
#     "buffer-size":[2500]
# }
basePath = '../data/dqn/temp/'
keyWordsAll = {
    "alpha": [3e-5],
    "dqn-hidden": [
        [64, 64]
    ],
    "dqn-sync":[32],
    "dqn-batch":[64],
    "buffer-size":[2500]
}

# summary_all()
plot_single_label("dqn-hidden", {})