import numpy as np

def getMeanAndCI95(returns_list_files):
    numruns = returns_list_files.shape[0]

    mean_returns_list = np.mean(returns_list_files, axis=0)
    stddev_returns_list = np.std(returns_list_files, axis=0)

    CI95_returns_list = 1.96 * stddev_returns_list / np.sqrt(numruns)
    min_CI95_returns_list = mean_returns_list - CI95_returns_list
    max_CI95_returns_list = mean_returns_list + CI95_returns_list

    return mean_returns_list, min_CI95_returns_list, max_CI95_returns_list
