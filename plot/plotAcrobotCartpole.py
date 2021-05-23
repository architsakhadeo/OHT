import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st


#datapath = ["../data/best/acrobotOptimal/", "../data/best/acrobotSubOptimal/", "../data/best/acrobotSubSubOptimal/"]
datapath = ["../data/best/cartpoleOptimal/", "../data/best/cartpoleSubOptimal/", "../data/best/cartpoleSubSubOptimal/"]
labels = ["optimal", "average", "bad"]

num_runs = 50

def findIndex(array, i):
    for j in range(len(array)):
        if i < array[j]:
            return j
        
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

for d in range(2):
    subdirs = os.listdir(datapath[d])
    for i in range(len(subdirs)):
        average = [0.0 for k in range(5600)]
        individual_runs = [[0.0 for k in range(5600)] for l in range(num_runs)]
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
            content = content[1:-1]
            for j in range(len(content)):
                content[j] = int(content[j])
            if content[-1] > max:
                max = content[-1]

            cumsum_content = np.cumsum(content)
            cumsum_content -= 1
            #print(cumsum_content)
            '''
            for j in range(5600):
                index = findIndex(cumsum_content, j)
                average[j] += content[index]
                individual_runs[f][j] = content[index]
            '''

            for t in cumsum_content:
                if t >= 5600:
                    continue
                individual_runs[f][t] = 1
                average[t] += 1
        
        for j in range(5600):
            average[j] /= (1.0*num_runs)
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
        for m in range(len(mean)-1):
            mean[m] = window_size / np.sum(average[m*window_size: (m+1)*window_size])




        xAxis = [window_size * i for i in range(int(len(average)/window_size))]
        print(i, len(labels), len(mean), len(xAxis))
        plt.plot(xAxis[:-2], mean[:-2], label=labels[d], color=colors[d])
        #plt.fill_between(xAxis, lower, upper, alpha=0.25, color=colors[i])
        
        #plt.plot([i for i in range(7500)], average[:7500], label=labels[i])

plt.ylabel("Steps to\n failure", rotation=0, labelpad=30)
plt.xlabel("Timesteps")
plt.legend()
#plt.ylim([0, 750])
plt.savefig("../data/best/cartpolePolicies.png", bbox_inches='tight')
plt.show()
        

        

