import os
import numpy as np
from loadFromEpisodeLengths import load_data
from loadFromEpisodeLengths import convert_data
from loadFromEpisodeLengths import transform_data


#dirpath = '../data/hyperparam/cartpole/offline_learning/knn/k3/esarsa-adam/step20k_env/lockat_baseline/'
dirpath = '../data/hyperparam/gridworld/online_learning/esarsa/1k/gridsearch_realenv/'
subdirs = os.listdir(dirpath)

def AUC(data):
    averageAcrossRuns = np.mean(data, axis=0)
    return np.sum(averageAcrossRuns[:])

def bottom10percentile(data):
    return np.sort(np.concatenate(data).ravel())[int(len(data)/10.0)]

performance = {}

for s in range(len(subdirs)):
    print('---------> ' + str(((s+1)*100.0)/len(subdirs)))
    data = load_data(dirpath+subdirs[s])
    convertedData, totalTimesteps = convert_data('', data)

    transformation = 'Returns'
    window = 100
    alpha = 0.0004
    averaging_type='exponential-averaging'

    transformedData = transform_data('', convertedData, totalTimesteps, transformation, window, type=averaging_type, alpha=alpha)
    performance[subdirs[s]] = AUC(transformedData)
    #performance[subdirs[s]] = bottom10percentile(transformedData)
    print('-------------------------------------------------------')

print((sorted(performance.items(), key=lambda item:item[1]))[::-1])
