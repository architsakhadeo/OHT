import os
import numpy as np
import matplotlib.pyplot as plt
from loadFromEpisodeLengths import transform_data
from stats import getMean


def avgReward(r_array, window):
    avg_arr = np.zeros(r_array.shape)
    for c in range(r_array.shape[1]):
        avg_arr[:, c] = r_array[:, max(0, c-window): c+1].mean(axis=1)
    return avg_arr

def readLog(file):
    with open(file, "r") as f:
        content = f.readlines()
    rewards = np.zeros(int(content[-1].split("total steps ")[1].split(" episode steps")[0])+1)
    ep_lens = []
    for line in content[3: ]:
        splited = line.split("total steps ")
        if len(splited) == 2 and len(splited[1].split(" episode steps"))==2:
            t = int(splited[1].split(" episode steps")[0])
            ep_l = int(splited[1].split(" episode steps")[1])
            rewards[t] = -1
            ep_lens.append(ep_l)
    return rewards

def load_stdout(root):
    assert os.path.isdir(root)
    all_files = []
    for path, subdirs, files in os.walk(root):
        for name in files:
            if name in ["stdout"]:
                filename = os.path.join(path, name)
                all_files.append(filename)
    return all_files

def loadFromOnePath(path, alg, window):
    onlyfiles = load_stdout(path)
    all_rewards = []
    min_len = np.inf
    for file in onlyfiles:
        reward = readLog(file)
        all_rewards.append(reward)
        min_len = len(reward) if len(reward) < min_len else min_len
    for i in range(len(all_rewards)):
        all_rewards[i] = all_rewards[i][:min_len]
    all_rewards = np.array(all_rewards)
    plotData = np.array(avgReward(all_rewards, window))
    return plotData

def plotMean(xAxis, data, color, alg):
    mean = getMean(data)
    plt.plot(xAxis, mean, label=alg+'-mean', color=color)

def plotFromLog(all_paths, transformation):
    window = 2500
    plottingData = {}
    for alg in all_paths.keys():
        plottingData[alg] = loadFromOnePath(all_paths[alg], alg, window)

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for alg, data in plottingData.items():
        lenRun = len(data[0])
        xAxis = np.array([i for i in range(1,lenRun+1)])

        if transformation == 'Average-Rewards':
            xAxis += (window-1)

        if alg == 'delay0':
            color = colors[0]
        elif alg == 'delay1':
            color = colors[1]
        elif alg == 'delay2':
            color = colors[2]
        elif alg == 'delay3':
            color = colors[3]
        elif alg == 'delay4':
            color = colors[4]

        plotMean(xAxis, data, color, alg)

    plt.plot(np.ones(lenRun)*(-1.0/200), "--", color='grey')
    #plt.title('Rewards averaged with sliding window of 1000 timesteps across 100 runs', pad=25, fontsize=10)
    plt.xlabel('Timesteps', labelpad=35)
    plt.ylabel(transformation, rotation=0, labelpad=45)
    plt.rcParams['figure.figsize'] = [8, 5.33]
    plt.legend(loc=0)
    plt.yticks()
    plt.xticks()
    plt.tight_layout()
    plt.show()


all_paths = {
    "delay0": "../data/dqn/adam-step3m/outputs/delay0/",
    "delay1": "../data/dqn/adam-step3m/outputs/delay1/",
    "delay2": "../data/dqn/adam-step3m/outputs/delay2/",
    "delay3": "../data/dqn/adam-step3m/outputs/delay3/",
    "delay4": "../data/dqn/adam-step3m/outputs/delay4/",
}
plotFromLog(all_paths, transformation='Average-Rewards')