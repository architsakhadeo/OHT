import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st


datapath = ["../data/best/puddleworldOptimal/", "../data/best/puddleworldSubOptimal/", "../data/best/puddleworldSubSubOptimal/"]
labels = ["optimal", "average", "bad"]

num_runs = 50

def findIndex(array, i):
    for j in range(len(array)):
        if i < array[j]:
            return j
        
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

for d in range(3):
    subdirs = os.listdir(datapath[d])
    for i in range(len(subdirs)):
        average = [0.0 for k in range(4400)]
        individual_runs = [[0.0 for k in range(4400)] for l in range(num_runs)]
        individual_runs_returns = [[0.0 for k in range(4400)] for l in range(num_runs)]
        average_returns = [0.0 for k in range(4400)]
        max = 0.0
        files = os.listdir(datapath[d]+subdirs[i])
        for f in range(len(files)):
            #print(files[f])
            if 'episodes' not in files[f]:
                continue
            #print(datapath)
            #print(subdirs, i, len(subdirs))
            #print(files, f, len(files))
            content = open(datapath[d]+subdirs[i]+'/'+files[f],'r').read().split('\n')
            returns_content = open(datapath[d]+subdirs[i]+'/'+files[f].replace('episodes','returns'),'r').read().split('\n')
            content = content[1:-1]
            returns_content = returns_content[1:-1]
            for j in range(len(content)):
                content[j] = int(content[j])
                returns_content[j] = float(returns_content[j])
            if content[-1] > max:
                max = content[-1]

            cumsum_content = np.cumsum(content)
            cumsum_content -= 1
            #print(cumsum_content)
            '''
            for j in range(4400):
                index = findIndex(cumsum_content, j)
                average[j] += content[index]
                individual_runs[f][j] = content[index]
            '''
            count = 0
            for t in cumsum_content:
                if t >= 4400:
                    continue
                individual_runs[f][t] = 1
                individual_runs_returns[f][t] = returns_content[count]
                count += 1
                average[t] += 1
                average_returns[t] += returns_content[count]
        
        for j in range(4400):
            average[j] /= (1.0*num_runs)
            average_returns[j] /= (1.0*num_runs)
        #average = average[:-max]
        #average = average[:7500]
        #print(average)
        #print(sum(average)/7500.0)

        #print(individual_runs)
        
        '''
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
        '''
        window_size = 200
        mean = [0 for i in range(int(len(average)/window_size))]
        mean_returns = [0 for i in range(int(len(average_returns)/window_size))]
        for m in range(len(mean)-1):
            mean[m] = window_size / np.sum(average[m*window_size: (m+1)*window_size])
            mean_returns[m] = np.sum(average_returns[m*window_size: (m+1)*window_size])




        xAxis = [window_size * i for i in range(int(len(average)/window_size))]
        #plt.plot(xAxis[:-2], mean[:-2], label=labels[d], color=colors[d])
        plt.plot(xAxis[:-1], np.array(mean_returns[:-1])/window_size, label=labels[d], color=colors[d])
        #plt.fill_between(xAxis, lower, upper, alpha=0.25, color=colors[i])
        
        #plt.plot([i for i in range(7500)], average[:7500], label=labels[i])

plt.ylabel("Average return\n per timestep", rotation=0, labelpad=30)
plt.xlabel("Timesteps")
plt.legend()
#plt.ylim([0, 100])
plt.savefig("../data/best/puddleworldPolicies_softmax.png", bbox_inches='tight')
plt.show()
        

        

