import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st

dirs = ["../../../../../Downloads/puddle/step600k/sweep/param_21/"]
for dir in dirs:
    files = os.listdir(dir)
    min = 10000000
    for file in files:
        if 'returns' not in file:
            continue
        content = open(dir+file,'r').read().split('\n')
        content = content[1:-1]
        if len(content) < min:
            min = len(content)
    
    individual_runs = []
    for file in files:
        if 'returns' not in file:
            continue
        content = open(dir+file,'r').read().split('\n')
        content = content[1:-1]
        content = content[:min]
        for j in range(len(content)):
            content[j] = float(content[j])
        individual_runs.append(content)
    
    npdata = np.array(individual_runs)
    mean = np.mean(npdata, axis=0)
    
    plt.plot([i+1 for i in range(min)], mean)
    #plt.ylim([-750, 0])
    plt.savefig("../data/best/puddleworld" + str(dir.split('/')[-2]) + ".png", bbox_inches='tight')
            
#colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

plt.show()
