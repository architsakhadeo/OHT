import csv
import os
import numpy as np
import pandas as pd

# Loads the rewards from the csv files into a dictionary and return the dictionary
def load_data(algpath):
	Data = []
	dirFiles = os.listdir(algpath)
	Files = np.array([i for i in dirFiles if 'rewards' in i])

	for fileIndex in range(len(Files)):
		List = pd.read_csv(algpath+'/'+Files[fileIndex])
		Data.append(List['rewards'])

	return Data


# Transforms the rewards to 'Rewards', 'Returns', 'Failures', 'Average-Rewards' 
def transform_data(alg, rewardsData, transformation='Rewards', window=0):
	
	transformedData = []

	for run in range(len(rewardsData)):
		if run % 10 == 0:
			print(run, alg)

		# Converts DataFrame rewards to numpy array rewards
		rewardsList = rewardsData[run].to_numpy().flatten()
		
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
			tempData = pd.DataFrame({'cumulativeFailures': failuresList})
		
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
