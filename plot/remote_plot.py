import os
import matplotlib.pyplot as plt
import numpy as np
from stats import getMean, getMedian, getBest, getWorst, getConfidenceIntervalOfMean, getRegion

# Add color, linestyles as needed

def plotMean(xAxis, data, label=None):
    mean = getMean(data)
    l = alg if label is None else label
    plt.plot(xAxis, mean, label=l+'-mean')

def plotMedian(xAxis, data):
    median = getMedian(data)
    plt.plot(xAxis, median, label=alg+'-median')

def plotBest(xAxis, data, transformation, label=None):
    best = getBest(data, transformation)
    l = alg if label is None else label
    plt.plot(xAxis, best, label=l+'-best')

def plotWorst(xAxis, data, transformation):
    worst = getWorst(data,  transformation)
    plt.plot(xAxis, worst, label=alg+'-worst')

def plotMeanAndConfidenceInterval(xAxis, data, confidence, label=None):
    plotMean(xAxis, data, label=label)
    lowerBound, upperBound = getConfidenceIntervalOfMean(data, confidence)
    plt.fill_between(xAxis, lowerBound, upperBound, alpha=0.25)

def plotMeanAndPercentileRegions(xAxis, data, lower, upper, transformation):
    plotMean(xAxis, data)
    lowerRun, upperRun = getRegion(data, lower, upper, transformation)
    plt.fill_between(xAxis, lowerRun, upperRun, alpha=0.25)

algorithmsToPlot = ['dqn']
basepath = '../data/'

rewardsData = {}

from load import load_data

for alg in os.listdir(basepath):
    if alg in algorithmsToPlot:
        sweep = os.listdir(basepath+alg+"/")
        rewardsData[alg] = {}
        for setting in sweep:
            rewardsData[alg][setting] = load_data(basepath+alg+"/"+setting)

print(rewardsData)
print('Data will be plotted for', ','.join([k for k in rewardsData.keys()]))
print('Loaded all the rewards from the csv files')

transformedData = {}

from load import transform_data

transformation = 'Returns'
window = 500

for alg in rewardsData.keys():
    transformedData[alg] = {}
    for setting, data in rewardsData[alg].items():
        transformedData[alg][setting] = transform_data(alg, data, transformation, window)

print(transformedData)
print('Data will be plotted for', ','.join([k for k in transformedData.keys()]))
print('All the stored rewards are transformed to: ', transformation)

for alg in transformedData.keys():
    for setting, data in transformedData[alg].items():
        lenRun = len(data[0])
        xAxis = np.array([i for i in range(1,lenRun+1)])


        if transformation == 'Average-Rewards':
            xAxis += (window-1)

        #plotMean(xAxis, data)

        #plotMedian(xAxis, data)

        # plotBest(xAxis, data, transformation=transformation)

        #plotWorst(xAxis, data, transformation=transformation)

        plotMeanAndConfidenceInterval(xAxis, data, confidence=0.9, label=setting.split("_")[0])

        #plotMeanAndPercentileRegions(xAxis, data, lower=0.5, upper=1.0, transformation=transformation)


    plt.ylim(-3000, 0)
    plt.xlabel('Timesteps', labelpad=35)
    plt.ylabel(transformation, rotation=0, labelpad=45)
    plt.rcParams['figure.figsize'] = [8, 5.33]
    plt.legend(loc=0)
    plt.yticks()
    plt.xticks()
    plt.tight_layout()

plt.savefig("temp.png")