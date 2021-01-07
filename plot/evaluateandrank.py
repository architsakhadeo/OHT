import os
import numpy as np
from loadFromEpisodeLengths import load_data
from loadFromEpisodeLengths import convert_data
from loadFromEpisodeLengths import transform_data


dirpath = '../data/renamedEsarsaModel/'
subdirs = os.listdir(dirpath)

def AUC(data):
    averageAcrossRuns = np.mean(data, axis=0)
    return np.sum(averageAcrossRuns)

def bottom50percentile(data):
    return np.sort(np.concatenate(data).ravel())[int(len(data)/2.0)]

performance = {}

for s in range(len(subdirs)):
    print('---------> ' + str(((s+1)*100.0)/len(subdirs)))
    data = load_data(dirpath+subdirs[s])
    convertedData, totalTimesteps = convert_data('', data)

    transformation = 'Average-Rewards'
    window = 2500
    alpha = 0.0004
    averaging_type='exponential-averaging'

    transformedData = transform_data('', convertedData, totalTimesteps, transformation, window, type=averaging_type, alpha=alpha)
    performance[subdirs[s]] = AUC(transformedData)
    #performance[subdirs[s]] = bottom50percentile(transformedData)
    print('-------------------------------------------------------')

print((sorted(performance.items(), key=lambda item:item[1]))[::-1])
