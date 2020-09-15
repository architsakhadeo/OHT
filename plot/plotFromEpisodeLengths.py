#!/usr/bin/env python
# coding: utf-8

# # Plot Comparison Between Algorithms

# Expects the input data to contain CSV files containing episode lengths


import os
import numpy as np
import matplotlib.pyplot as plt
from stats import getMean, getMedian, getBest, getWorst, getConfidenceIntervalOfMean, getRegion
from loadFromEpisodeLengths import load_data
from loadFromEpisodeLengths import convert_data
from loadFromEpisodeLengths import transform_data
#print(plt.rcParams['agg.path.chunksize'])
#plt.rcParams['agg.path.chunksize'] = 100000

# We need to read the CSV files (from a function in another file) to get the reward at each timestep for each run of each algorithm. Only the `dataPath` will be loaded.
# 
# `load_data` loads the CSV files containing episode lengths as a numpy array of Pandas DataFrames.
# 
# `dataPath` contains the exact path of the directories containing the CSV files. This path is relative to the `data` directory. It assumes every element will be path for a different algorithm. It will overwrite if two paths are for different parameter settings of the same algorithm.
# 
# Expects there to be more than 1 input CSV file.

#labels = ['esarsa-best-adam', 'dqn-1e-5', 'dqn-3e-5']
#dataPath = ['esarsa1/adaptive-alpha=3e-06_adaptive-stepsize=1_alpha=0.1_delays=0_enable-debug=0_epsilon=0.1_gamma=0.9_is-stepsize-adaptive=1_lambda=0.7_tiles=8_tilings=32/', 'dqn1/alpha=1e-05_buffer-size=2500_buffer-type=random_decreasing-epsilon=None_delays=0_dqn-adamBeta1=0.9_dqn-adamBeta2=0.999_dqn-adamEps=1e-08_dqn-batch=64_dqn-hidden=64,64_dqn-sync=32_e/', 'dqn2/alpha=3e-05_buffer-size=2500_buffer-type=random_decreasing-epsilon=None_delays=0_dqn-adamBeta1=0.9_dqn-adamBeta2=0.999_dqn-adamEps=1e-08_dqn-batch=64_dqn-hidden=64,64_dqn-sync=32_enable-debug=']
#dataPath = ['esarsa1/adaptive-alpha=3e-06_adaptive-stepsize=1_alpha=0.1_delays=0_enable-debug=0_epsilon=0.1_gamma=0.9_is-stepsize-adaptive=1_lambda=0.7_tiles=8_tilings=32/', 'dqn3/alpha=3e-05_buffer-size=2500_buffer-type=random_decreasing-epsilon=None_delays=0_dqn-adamBeta1=0.9_dqn-adamBeta2=0.999_dqn-adamEps=1e-08_dqn-batch=64_dqn-hidden=64,64_dqn-sync=32/']

labels = ['esarsa-3e-6', 'dqn-1e-5']
dataPath = ['esarsa1/adaptive-alpha=3e-06_adaptive-stepsize=1_alpha=0.1_delays=0_enable-debug=0_epsilon=0.1_gamma=0.9_is-stepsize-adaptive=1_lambda=0.7_tiles=8_tilings=32/', 'dqn4/alpha=1e-05_buffer-size=2500_buffer-type=random_decreasing-epsilon=None_delays=0_dqn-adamBeta1=0.9_dqn-adamBeta2=0.999_dqn-adamEps=1e-08_dqn-batch=64_dqn-hidden=64,64_dqn-sync=32_enable-debug=0_']

#labels = ['dqn-1e-5']
#dataPath = ['dqn4/alpha=1e-05_buffer-size=2500_buffer-type=random_decreasing-epsilon=None_delays=0_dqn-adamBeta1=0.9_dqn-adamBeta2=0.999_dqn-adamEps=1e-08_dqn-batch=64_dqn-hidden=64,64_dqn-sync=32_enable-debug=0_']

#labels = ['adam']
#dataPath = ['esarsa_adam/adaptive-alpha=0.001_alpha=0.1_delays=0_enable-debug=0_epsilon=0.05_gamma=0.95_is-stepsize-adaptive=1_lambda=0.8_tiles=4_tilings=32/']
#dataPath = ['esarsa_sgd/adaptive-alpha=0.001_alpha=0.1_delays=0_enable-debug=0_epsilon=0.05_gamma=0.95_is-stepsize-adaptive=0_lambda=0.8_tiles=4_tilings=32']#, 'esarsa_adam/adaptive-alpha=0.001_alpha=0.1_delays=0_enable-debug=0_epsilon=0.05_gamma=0.95_is-stepsize-adaptive=1_lambda=0.8_tiles=4_tilings=32/']

'''
dataPath = ['esarsa/alpha=0.1_delays=0_enable-debug=0_epsilon=0.05_gamma=0.95_lambda=0.8_tiles=4_tilings=32/',
            'esarsa/alpha=0.1_delays=1_enable-debug=0_epsilon=0.05_gamma=0.95_lambda=0.8_tiles=4_tilings=32/',
            'esarsa/alpha=0.1_delays=2_enable-debug=0_epsilon=0.05_gamma=0.95_lambda=0.8_tiles=4_tilings=32/',
            'esarsa/alpha=0.1_delays=3_enable-debug=0_epsilon=0.05_gamma=0.95_lambda=0.8_tiles=4_tilings=32/',
            'esarsa/alpha=0.1_delays=4_enable-debug=0_epsilon=0.05_gamma=0.95_lambda=0.8_tiles=4_tilings=32/',
            'esarsa/alpha=0.1_delays=5_enable-debug=0_epsilon=0.05_gamma=0.95_lambda=0.8_tiles=4_tilings=32/',
            'esarsa/alpha=0.1_delays=6_enable-debug=0_epsilon=0.05_gamma=0.95_lambda=0.8_tiles=4_tilings=32/',
            'esarsa/alpha=0.1_delays=7_enable-debug=0_epsilon=0.05_gamma=0.95_lambda=0.8_tiles=4_tilings=32/',
            'esarsa/alpha=0.1_delays=8_enable-debug=0_epsilon=0.05_gamma=0.95_lambda=0.8_tiles=4_tilings=32/',
            'esarsa/alpha=0.1_delays=9_enable-debug=0_epsilon=0.05_gamma=0.95_lambda=0.8_tiles=4_tilings=32/',
            'esarsa/alpha=0.1_delays=10_enable-debug=0_epsilon=0.05_gamma=0.95_lambda=0.8_tiles=4_tilings=32/',
            'esarsa/alpha=0.1_delays=11_enable-debug=0_epsilon=0.05_gamma=0.95_lambda=0.8_tiles=4_tilings=32/',
            'esarsa/alpha=0.1_delays=12_enable-debug=0_epsilon=0.05_gamma=0.95_lambda=0.8_tiles=4_tilings=32/',
            'esarsa/alpha=0.1_delays=13_enable-debug=0_epsilon=0.05_gamma=0.95_lambda=0.8_tiles=4_tilings=32/',
            'esarsa/alpha=0.1_delays=14_enable-debug=0_epsilon=0.05_gamma=0.95_lambda=0.8_tiles=4_tilings=32/',
            'esarsa/alpha=0.1_delays=15_enable-debug=0_epsilon=0.05_gamma=0.95_lambda=0.8_tiles=4_tilings=32/',
            'esarsa/alpha=0.1_delays=16_enable-debug=0_epsilon=0.05_gamma=0.95_lambda=0.8_tiles=4_tilings=32/',
            'esarsa/alpha=0.1_delays=17_enable-debug=0_epsilon=0.05_gamma=0.95_lambda=0.8_tiles=4_tilings=32/',
            'esarsa/alpha=0.1_delays=18_enable-debug=0_epsilon=0.05_gamma=0.95_lambda=0.8_tiles=4_tilings=32/',
            'esarsa/alpha=0.1_delays=19_enable-debug=0_epsilon=0.05_gamma=0.95_lambda=0.8_tiles=4_tilings=32/',
            'esarsa/alpha=0.1_delays=20_enable-debug=0_epsilon=0.05_gamma=0.95_lambda=0.8_tiles=4_tilings=32/'
           ]
'''
'''
dataPath = ['esarsa/alpha=0.1_delays=0_enable-debug=0_epsilon=0.05_gamma=0.95_lambda=0.8_tiles=4_tilings=32/',
            'esarsa/alpha=0.1_delays=1_enable-debug=0_epsilon=0.05_gamma=0.95_lambda=0.8_tiles=4_tilings=32/',
            'esarsa/alpha=0.1_delays=2_enable-debug=0_epsilon=0.05_gamma=0.95_lambda=0.8_tiles=4_tilings=32/',
            'esarsa/alpha=0.1_delays=3_enable-debug=0_epsilon=0.05_gamma=0.95_lambda=0.8_tiles=4_tilings=32/',
            'esarsa/alpha=0.1_delays=4_enable-debug=0_epsilon=0.05_gamma=0.95_lambda=0.8_tiles=4_tilings=32/'
            ]

'''
'''
dataPath = ['dqn/alpha=0.001_buffer-size=500_buffer-type=random_decreasing-epsilon=None_delays=0_dqn-batch=16_dqn-hidden=128,128_dqn-momentum=0.9_dqn-sync=32_enable-debug=0_epsilon=0.1_gamma=0.9_min-epsilon=0.1_state-len=4/',
            'dqn/alpha=0.001_buffer-size=500_buffer-type=random_decreasing-epsilon=None_delays=1_dqn-batch=16_dqn-hidden=128,128_dqn-momentum=0.9_dqn-sync=32_enable-debug=0_epsilon=0.1_gamma=0.9_min-epsilon=0.1_state-len=4/',
            'dqn/alpha=0.001_buffer-size=500_buffer-type=random_decreasing-epsilon=None_delays=2_dqn-batch=16_dqn-hidden=128,128_dqn-momentum=0.9_dqn-sync=32_enable-debug=0_epsilon=0.1_gamma=0.9_min-epsilon=0.1_state-len=4/',
            'dqn/alpha=0.001_buffer-size=500_buffer-type=random_decreasing-epsilon=None_delays=3_dqn-batch=16_dqn-hidden=128,128_dqn-momentum=0.9_dqn-sync=32_enable-debug=0_epsilon=0.1_gamma=0.9_min-epsilon=0.1_state-len=4/',
            'dqn/alpha=0.001_buffer-size=500_buffer-type=random_decreasing-epsilon=None_delays=4_dqn-batch=16_dqn-hidden=128,128_dqn-momentum=0.9_dqn-sync=32_enable-debug=0_epsilon=0.1_gamma=0.9_min-epsilon=0.1_state-len=4/'   
            ]
'''


basePath = '../data/'


for i in range(len(dataPath)):

    algorithms = [dataPath[i].split('/')[0] for i in range(len(dataPath))]
    print(algorithms)    
    Data = {}

    if os.path.isdir(basePath + dataPath[i]) == True:
        Data[algorithms[i]] = load_data(basePath+dataPath[i])
    

    print('Data will be stored for', ', '.join([k for k in Data.keys()]))
    print('Loaded the episode lengths data from the csv files')


    # `convert_data` converts the episode lengths (numpy array of Pandas DataFrames) into the absolute timesteps when failures occur (a python list of Pandas DataFrames)

    convertedData = {}

    for alg, data in Data.items():
        convertedData[alg], totalTimesteps = convert_data(alg, data)
        print(len(convertedData[alg]))

    print('Data will be stored for', ', '.join([k for k in convertedData.keys()]))
    print('The stored episode lengths are converted to absolute failure timesteps')


    # The rewards can be transformed into the following values of transformation =
    # 1. 'Returns'
    # 2. 'Failures'
    # 3. 'Average-Rewards'
    # 4. 'Rewards' (no change)
    # 
    # ----------------------------------------------------------------------------------------------
    # 
    # There is an additional parameter of window which can be any non-negative integer. It is used for the 'Average-Rewards' transformation to maintain a moving average over a sliding window. By default window is 0.
    # 
    # - If window is 500 and timesteps are 10000, then the first element is the average of the performances of timesteps from 1 - 500. The second element is the average of the performances of timesteps from 2 - 501. The last element is the average of the performances of timesteps from 9501 - 10000.
    # 
    # ----------------------------------------------------------------------------------------------
    # 
    # `transform_data` transforms the absolute failure timesteps (python list of Pandas DataFrames) into the respective `transformation` (a numpy array of numpy arrays) for plotting


    plottingData = {}


    transformation = 'Average-Rewards'
    window = 2500
    alpha = 0.01
    averaging_type='exponential-averaging'


    for alg, data in convertedData.items():
        plottingData[alg] = transform_data(alg, data, totalTimesteps, transformation, window, type=averaging_type, alpha=alpha)

    print('Data will be plotted for', ', '.join([k for k in plottingData.keys()]))
    print('The stored failure timesteps are transformed to: ', transformation)

    # Add color, linestyles as needed

    def plotMean(xAxis, data, color, label):
        mean = getMean(data)
        #plt.plot(xAxis, mean, label=alg+'-mean'+label, color=color)
        plt.plot(xAxis, mean, label=label, color=color)


    def plotMedian(xAxis, data, color, label):
        median = getMedian(data)
        plt.plot(xAxis, median, label=alg+'-median'+label, color=color)

    def plotBest(xAxis, data, transformation, color, label):
        best = getBest(data, transformation)
        plt.plot(xAxis, best, label=alg+'-best'+label, color=color)

    def plotWorst(xAxis, data, transformation, color, label):
        worst = getWorst(data,  transformation)
        plt.plot(xAxis, worst, label=alg+'-worst'+label, color=color)

    def plotMeanAndConfidenceInterval(xAxis, data, confidence, color, label):
        plotMean(xAxis, data, color, label)
        lowerBound, upperBound = getConfidenceIntervalOfMean(data, confidence)
        if transformation == 'Average-Rewards':    
            upperBound = np.clip(upperBound, a_min = None, a_max=0.0)
        plt.fill_between(xAxis, lowerBound, upperBound, alpha=0.25, color=color)

    def plotMeanAndPercentileRegions(xAxis, data, lower, upper, transformation, color, label):
        plotMean(xAxis, data, color, label)
        lowerRun, upperRun = getRegion(data, lower, upper, transformation)
        if transformation == 'Average-Rewards':    
            upperRun = np.clip(upperRun, a_min = None, a_max=0.0)
        plt.fill_between(xAxis, lowerRun, upperRun, alpha=0.25, color=color)


    # Here, we can plot the following statistics:
    # 
    # 1. Mean of all the runs
    # 
    # 2. Median run
    # 
    # 3. Run with the best performance (highest return, or equivalently least failures)
    # 
    # 4. Run with the worst performance (lowest return, or equivalently most failures)
    # 
    # 5. Mean along with the confidence interval (Currently, plots the mean along with 95% confidence interval, but should be changed to make it adaptive to any confidence interval)
    # 
    # 6. Mean along with percentile regions (Plots the mean and shades the region between the run with the lower percentile and the run with the upper percentile)
    # 
    # ----------------------------------------------------------------------------------------------
    # 
    # Details:
    # 
    # plotBest, plotWorst, plotMeanAndPercentileRegions sort the performances based on their final performance
    # 
    #                                    ----------------------------------------------------
    # 
    # Mean, Median, MeanAndConfidenceInterval are all symmetric plots so 'Failures' does not affect their plots
    #     
    # Best, Worst, MeanAndPercentileRegions are all asymmetric plots so 'Failures' affects their plots, and has to be treated in the following way:   
    # 
    #                                    ----------------------------------------------------
    # 
    # 1. plotBest for Returns will plot the run with the highest return (least failures)
    #    plotBest for Failures will plot the run with the least failures and not the highest failures
    # 
    # 2. plotWorst for Returns will plot the run with the lowest return (most failures)
    #    plotWorst for Failures will plot the run with the most failures and not the least failures
    # 
    # 3. plotMeanAndPercentileRegions for Returns uses the lower variable to select the run with the 'lower' percentile and uses the upper variable to select the run with the 'upper' percentile
    #    plotMeanAndPercentileRegions for Failures uses the lower variable along with some calculations to select the run with 'upper' percentile and uses the upper variable along with some calculations to select the run with the 'lower' percentile 
    #     
    # ----------------------------------------------------------------------------------------------
    # 
    # Caution:
    # - Jupyter notebooks (mostly) or matplotlib gives an error when displaying very dense plots. For example: plotting best and worst case for transformation of 'Rewards' for 'example' algorithm, or when trying to zoom into dense plots. Most of the plots for 'Rewards' and 'example' fail.


    # Details:
    # 
    # - X axis for 'Average-Rewards' will start from 'window' timesteps and end with the final timesteps
    # 
    # - Need to add color (shades), linestyle as per requirements
    # 
    # - Currently plot one at a time by commenting out the others otherwise, it displays different colors for all.
    # 


    # For saving figures
    #%matplotlib inline

    # For plotting in the jupyter notebook

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for alg, data in plottingData.items():
        lenRun = len(data[0][:1000000])
        #lenRun = len(data[0])
        xAxis = np.array([i for i in range(1,lenRun+1)])
        
        if transformation == 'Average-Rewards' and type=='sample-averaging':
            xAxis += (window-1)
        
        if alg == 'esarsa':
            color = colors[0]
        elif alg == 'hand':
            color = colors[1]
        elif alg == 'dqn':
            color = colors[2]
        
        #plotMean(xAxis, data, color=colors[i%len(colors)], label='delay-'+str(i))

        #plotMedian(xAxis, data, color=color, label='')
        
        #plotBest(xAxis, data, transformation=transformation, color=color, label='')
        
        #plotWorst(xAxis, data, transformation=transformation, color=color, label='')
        plotwindow = 1000000
        for j in range(0, len(xAxis), plotwindow):
            if j + plotwindow >= len(xAxis):
                temp = len(xAxis) - j
            else:
                temp = plotwindow
            tempxAxis = xAxis[j:j+temp]
            tempdata = data[:,j:j+temp]

            if j + plotwindow >= len(xAxis):
                plotMeanAndConfidenceInterval(tempxAxis, tempdata, confidence=0.95, color=colors[i], label=labels[i])
                #plotMeanAndConfidenceInterval(tempxAxis, tempdata, confidence=0.95, color=colors[i], label=labels[i])
                #plotMeanAndPercentileRegions(tempxAxis, tempdata, lower=0.0, upper=1.0, transformation=transformation, color=colors[i], label=labels[i])
                #plotBest(tempxAxis, tempdata, transformation=transformation, color=colors[i], label=labels[i])
            else:
                plotMeanAndConfidenceInterval(tempxAxis, tempdata, confidence=0.95, color=colors[i], label=None)
                #plotMeanAndConfidenceInterval(tempxAxis, tempdata, confidence=0.95, color=colors[i], label=None)
                #plotMeanAndPercentileRegions(tempxAxis, tempdata, lower=0.0, upper=1.0, transformation=transformation, color=colors[i], label=labels[i])
                #plotBest(tempxAxis, tempdata, transformation=transformation, color=colors[i], label=labels[i])
        
        #plotMeanAndPercentileRegions(xAxis, data, lower=0.025, upper=0.975, transformation=transformation, color=color, label='')

xAxis = np.array([i for i in range(1,1000000+1)])
plt.plot(xAxis, np.array([0 for i in range(len(xAxis))]), '--', color='black', linewidth=0.5)

plt.title('ESarsa-DQN-Adam ' + 'alpha='+str(alpha), pad=25, fontsize=10)
plt.xlabel('Timesteps', labelpad=35)
plt.ylabel(transformation, rotation=0, labelpad=45)
plt.rcParams['figure.figsize'] = [8, 5.33]
#plt.legend(loc=(-0.5, 1.1), prop={"size":8})
plt.legend(prop={"size":8})
plt.yticks()
plt.xticks()
bottom, top = plt.ylim()
plt.ylim(-0.02, top)
#plt.ylim(bottom, 50000)
plt.tight_layout()
#plt.show()
#plt.savefig('../img/dqn-10M-'+transformation+'.png',dpi=500, bbox_inches='tight')
plt.savefig('../img/comparison-'+str(averaging_type)+'-alpha='+str(alpha)+'-'+str(transformation)+'.png',dpi=500, bbox_inches='tight')
