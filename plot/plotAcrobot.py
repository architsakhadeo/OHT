import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st

datapath = "../../../../../Downloads/transferabledata/new/data/hyperparam_ap/acrobot/online_learning/esarsa/step10k/best/"
labels = ["best", "10%", "20%", "30%", "40%", "50%"]
subdirs = ['param_18/', 'param_29/', 'param_38/', 'param_37/', 'param_30/', 'param_32/']

num_runs = 30

def findIndex(array, i):
    for j in range(len(array)):
        if i < array[j]:
            return j
        
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

for i in range(3):
    average = [0.0 for k in range(7500)]
    individual_runs = [[0.0 for k in range(7500)] for l in range(num_runs)]
    max = 0.0
    files = os.listdir(datapath+subdirs[i])
    for f in range(len(files)):
        if 'episodes' not in files[f]:
            continue
        content = open(datapath+subdirs[i]+files[f],'r').read().split('\n')
        content = content[1:-1]
        for j in range(len(content)):
            content[j] = int(content[j])
        if content[-1] > max:
            max = content[-1]

        cumsum_content = np.cumsum(content)
        
        for j in range(7500):
            index = findIndex(cumsum_content, j)
            average[j] += content[index]
            individual_runs[f][j] = content[index]
    
    for j in range(7500):
        average[j] /= (1.0*num_runs)
    
    #average = average[:-max]
    #average = average[:7500]
    #print(average)
    #print(sum(average)/7500.0)

    #print(individual_runs)
    npdata = np.array(individual_runs)
    
    mean = np.mean(npdata,axis=0)
    stderror = np.std(npdata,axis=0)/(num_runs**0.5)

    criticalValue = st.norm.ppf( (0.95 + 1.0) / 2.0 )
    confidenceInterval = criticalValue * stderror

    #lower = mean - stderror
    #upper = mean + stderror

    print(criticalValue)
    lower = mean - confidenceInterval
    upper = mean + confidenceInterval
    
    xAxis = [i for i in range(7500)]
    plt.plot(xAxis, mean, label=labels[i], color=colors[i])
    plt.fill_between(xAxis, lower, upper, alpha=0.25, color=colors[i])
    
    #plt.plot([i for i in range(7500)], average[:7500], label=labels[i])

plt.ylabel("Steps to\n success", rotation=0, labelpad=30)
plt.xlabel("Timesteps")
plt.legend()
plt.savefig("Acrobot_7500CI.png", bbox_inches='tight')
plt.show()
        

        

