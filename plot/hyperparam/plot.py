import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Loads the episode lengths from the csv files into a dictionary and return the dictionary
def load_data(algpath, name='episodes'):
    Data = []
    dirFiles = os.listdir(algpath)
    # Files = np.array([i for i in dirFiles if 'episodes' in i])
    Files = np.array([i for i in dirFiles if name in i])

    for fileIndex in range(len(Files)):
        if name == "episodes":
            List = pd.read_csv(algpath+'/'+Files[fileIndex])
            Data.append(List['episode lengths'])
        elif name == "rewards":
            List = pd.read_csv(algpath+'/'+Files[fileIndex])
            Data.append(List['rewards'])
    return np.array(Data) if len(Data) !=1 else Data

def convert_data_ep(Data):
    convertedData = []
    for run in range(len(Data)):
        episodeLengthsData = Data[run].to_numpy()
        failureTimesteps = np.cumsum(episodeLengthsData)
        totalTimesteps = failureTimesteps[-1]
        # Not a failure on the last episode on the last timestep
        if episodeLengthsData[-1] != 0.0:
            failureTimesteps = failureTimesteps[:-1]
        failureTimesteps_DataFrame = pd.DataFrame({'failures': failureTimesteps})
        convertedData.append(failureTimesteps_DataFrame)
    return convertedData, totalTimesteps

def convert_data_reward(data):
    convertedData = np.zeros(data.shape)
    convertedData[:, 0] = data[:, 0]
    for s in range(1, data.shape[1]):
        convertedData[:, s] = data[:, s] + convertedData[:, s-1]
    return convertedData, data.shape[1]


def transform_data(failureTimesteps, totalTimesteps, transformation='Rewards', window=0):
    transformedData = []
    for run in range(len(failureTimesteps)):
        # Calculate rewards from failure timesteps
        indexing = (failureTimesteps[run] - 1).to_numpy().flatten()
        rewardsList = np.zeros(totalTimesteps)
        rewardsList[indexing] = -1.0
        # Keep the data to rewards
        if transformation == 'Rewards':
            tempData = pd.DataFrame({'rewards': rewardsList})
        # Returns are equal to sum of rewards
        elif transformation == 'Returns':
            returnsList = np.cumsum(rewardsList)
            tempData = pd.DataFrame({'returns': returnsList})
        # Failures are equal to negative returns
        elif transformation == 'Failures':
            returnsList = np.cumsum(rewardsList)
            failuresList = -1 * returnsList
            tempData = pd.DataFrame({'cummulativeFailures': failuresList})
        # Average rewards are calculated in a moving average manner
        # over a sliding window using the np.convolve method
        elif transformation == 'Average-Rewards':
            # Change this code carefully
            AverageRewardsList = np.convolve(rewardsList, np.ones(window)/window, 'valid')
            tempData = pd.DataFrame({'averageRewards': AverageRewardsList})
        transformedData.append(tempData)
    # Change DataFrames to numpy arrays
    for i in range(len(transformedData)):
        transformedData[i] = transformedData[i].to_numpy().flatten()
    transformedData = np.array(transformedData)
    return transformedData

def sliding_window(data, window=2500):
    new = np.zeros(data.shape)
    # for i in range(window):
    #     new[:, i] = np.mean(data[:, :window], axis=1)
    for i in range(window, len(data[0])):
        new[:, i] = np.mean(data[:, i-window: i+1], axis=1)
    return new[:, window:]

def avg_episode_data_failures(path):
    data = load_data(path, "rewards")
    convertedData, totalTimesteps = convert_data_reward(data)
    avg = np.mean(convertedData, axis=0)
    avg = -1 * avg
    return avg

def avg_episode_data(path):
    data = load_data(path, "rewards")
    # data = sliding_window(data)
    avg = np.mean(data, axis=0)
    return avg

def setting_from_json(file, keys):
    with open(file, "r") as f:
        lines = f.readlines()
    values = {}
    for l in lines:
        k,v = l.strip().split("=")
        if k in keys:
            values[k] = v
    return values

def sweep(basepath, label_keys):
    all_param = os.listdir(basepath)
    assert [p[:5] == "param" for p in all_param]
    best_auc = -1 * np.inf
    best_data = None
    best_label = None
    plt.figure()
    auc_rec = {}
    for param in all_param:
        folder = basepath + "/" + param
        setting = folder + "/log_json.txt"
        label = setting_from_json(setting, label_keys)
        label = " ".join([str(k)+"="+str(v) for k, v in label.items()])
        data = avg_episode_data(folder)
        plt.plot(data, label=label)
        auc = np.sum(data)
        auc_rec[label] = auc
        if auc > best_auc:
            best_auc = auc
            best_label = label
            best_data = data
    plt.title(basepath)
    plt.legend()
    # plt.ylim(np.mean(best_data)*5, 0)
    plt.ylim(-0.02, 0)
    plt.savefig("../../img/"+basepath.split("/")[-1]+".png")

    sort_auc = [[k, v] for k, v in sorted(auc_rec.items(), key=lambda item: item[1])]
    return {"data": best_data, "label": best_label}, sort_auc

def all_path(path_list, label_keys):
    best_list = []
    auc_allpath = {}
    for path in path_list:
        best_setting, sort_auc = sweep(path, label_keys)
        best_setting["label"] = path.split("/")[-1]+" "+best_setting["label"]
        best_list.append(best_setting)
        auc_allpath[path.split("/")[-1]] = sort_auc
    # plt.figure()
    # for data in best_list:
    #     plt.plot(data["data"], label=data["label"])
    # plt.legend()
    # plt.show()
    format_print_auc(auc_allpath)
    return

def format_print_auc(auc_dic):
    for key, rank in auc_dic.items():
        print(key, ":")
        for item in rank:
            label, auc = item
            print(label.split("=")[1], end=",\t")
        print("\n")

paths = [
    "../../data/hyperparam/cartpole/offline_learning/dqn-adam/step10k_env/lockat_baseline",
    "../../data/hyperparam/cartpole/offline_learning/dqn-adam/step10k_env/lockat_halfbaseline",
    "../../data/hyperparam/cartpole/offline_learning/dqn-adam/step10k_env/lockat_quarterbaseline",
    "../../data/hyperparam/cartpole/offline_learning/dqn-adam/step10k_env/lockat_-0.1",
    "../../data/hyperparam/cartpole/offline_learning/dqn-adam/step10k_env/lockat_random",
    "../../data/hyperparam/cartpole/online_learning/dqn-adam/step50k/sweep_lr",
]
# paths = [
#     "../../data/hyperparam/cartpole/offline_learning/dqn-adam/step1k_env/lockat_baseline",
#     "../../data/hyperparam/cartpole/offline_learning/dqn-adam/step1k_env/lockat_halfbaseline",
#     "../../data/hyperparam/cartpole/offline_learning/dqn-adam/step1k_env/lockat_quarterbaseline",
#     "../../data/hyperparam/cartpole/offline_learning/dqn-adam/step1k_env/lockat_-0.1",
#     "../../data/hyperparam/cartpole/offline_learning/dqn-adam/step1k_env/lockat_random",
#     "../../data/hyperparam/cartpole/online_learning/dqn-adam/step50k/sweep_lr",
# ]
keys = ["alpha"]

# paths = [
#     "../../data/hyperparam/cartpole/offline_learning/esarsa-adam/step10k_env/lockat_baseline",
#     "../../data/hyperparam/cartpole/offline_learning/esarsa-adam/step10k_env/lockat_halfbaseline",
#     "../../data/hyperparam/cartpole/offline_learning/esarsa-adam/step10k_env/lockat_quarterbaseline",
#     "../../data/hyperparam/cartpole/offline_learning/esarsa-adam/step10k_env/lockat_-0.1",
#     "../../data/hyperparam/cartpole/offline_learning/esarsa-adam/step10k_env/lockat_random",
#     "../../data/hyperparam/cartpole/online_learning/esarsa-adam/step50k/sweep_lr",
# ]
# keys = ["adaptive-alpha"]
all_path(paths, keys)