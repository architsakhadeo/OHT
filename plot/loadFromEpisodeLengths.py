import csv
import os
import numpy as np
import pandas as pd

# Loads the episode lengths from the csv files into a dictionary and return the dictionary
def load_data(algpath):
	Data = []
	dirFiles = os.listdir(algpath)
	Files = np.array([i for i in dirFiles if 'episodes' in i])

	for fileIndex in range(len(Files)):
		List = pd.read_csv(algpath+'/'+Files[fileIndex])
		Data.append(List['episode lengths'])
	return np.array(Data) if len(Data) !=1 else Data


# Converts episode lengths into failure timesteps
def convert_data(alg, Data):
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


# Transforms the failures timesteps to 'Rewards', 'Returns', 'Failures', 'Average-Rewards' 
# type = 'exponential-averaging' or 'sample-averaging'
def transform_data(alg, failureTimesteps, totalTimesteps, transformation='Rewards', window=0, type='exponential-averaging', alpha=0.9):
	
	transformedData = []

	for run in range(len(failureTimesteps)):
		# if run % 10 == 0:
		# 	print(run, alg)

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
			if type == 'sample-averaging':
				AverageRewardsList = np.convolve(rewardsList, np.ones(window)/window, 'valid')
			elif type == 'exponential-averaging':
				AverageRewardsList = [rewardsList[0]]
				o_n_minus_1 = 0
				for i in range(1, len(rewardsList)):
					o_n = o_n_minus_1  + alpha*(1 - o_n_minus_1)
					beta_n = alpha / (o_n)
					AverageRewardsList.append(AverageRewardsList[-1] + beta_n * (rewardsList[i] - AverageRewardsList[-1]))
					o_n_minus_1 = o_n
			tempData = pd.DataFrame({'averageRewards': AverageRewardsList})

		transformedData.append(tempData)

	# Change DataFrames to numpy arrays
	for i in range(len(transformedData)):
		transformedData[i] = transformedData[i].to_numpy().flatten()
	transformedData = np.array(transformedData)

	return transformedData
