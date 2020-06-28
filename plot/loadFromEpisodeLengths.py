import csv
import os
import numpy as np
import pandas as pd

# Loads the rewards from the csv files into a dictionary and return the dictionary
def load_data(algpath):
	Data = []
	dirFiles = os.listdir(algpath)
	Files = np.array([i for i in dirFiles if 'episodes' in i])

	for fileIndex in range(len(Files)):
		List = pd.read_csv(algpath+'/'+Files[fileIndex])
		Data.append(List['episode lengths'])

	return np.array(Data)


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


# Transforms the rewards to 'Rewards', 'Returns', 'Failures', 'Average-Rewards' 
def transform_data(alg, failureTimesteps, totalTimesteps, transformation='Rewards', window=0):
	
	transformedData = []

	for run in range(len(failureTimesteps)):
		if run % 10 == 0:
			print(run, alg)

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
