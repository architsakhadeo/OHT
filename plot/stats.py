import numpy as np

def getMeanAndCI95(returns_list_files):
    numruns = returns_list_files.shape[0]

    mean_returns_list = np.mean(returns_list_files, axis=0)
    stddev_returns_list = np.std(returns_list_files, axis=0)

    CI95_returns_list = 1.96 * stddev_returns_list / np.sqrt(numruns)
    min_CI95_returns_list = mean_returns_list - CI95_returns_list
    max_CI95_returns_list = mean_returns_list + CI95_returns_list

    return mean_returns_list, min_CI95_returns_list, max_CI95_returns_list


'''
Calculate the median run based on final performance
'''
def getMedianBestWorstFinal(returns_list_files):
    numruns = returns_list_files.shape[0]

    indices = np.argsort(returns_list_files[:,-1])
    if numruns % 2 == 1:
        median_returns_list = returns_list_files[indices[int((numruns-1)/2)]]
    else:
        median_returns_list = (returns_list_files[indices[int(numruns/2)]] + returns_list_files[indices[int(numruns/2 - 1)]])/2.0

    best_returns_list = returns_list_files[indices[-1]]
    worst_returns_list = returns_list_files[indices[0]]

    return median_returns_list, best_returns_list, worst_returns_list


'''
Calculate median across every timestep, without reporting the median run
'''
def getMedianPerTimestep(returns_list_files):
    numruns = returns_list_files.shape[0]

    sorted_returns_list = np.sort(returns_list_files, axis=0)
    if numruns % 2 == 1:
        median_returns_list = sorted_returns_list[int((numruns-1)/2)]
    else:
        median_returns_list = (sorted_returns_list[int(numruns/2)] + sorted_returns_list[int(numruns/2 - 1)]) / 2.0

    return median_returns_list
