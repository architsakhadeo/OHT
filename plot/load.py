import csv
import os
import numpy as np
import pandas as pd

# Loads the rewards from the csv files into a dictionary and return the dictionary
def load_data(algpath):
	rewardsData = np.array([])
	dirFiles = os.listdir(algpath)
	rewardsFiles = np.array([i for i in dirFiles if 'rewards' in i])

	for fileIndex in range(len(rewardsFiles)):
		if fileIndex % 10 == 0:
			print(fileIndex, algpath)
		rewardsList = pd.read_csv(algpath+'/'+rewardsFiles[fileIndex])

		if rewardsData.size != 0:
			rewardsData = np.vstack((rewardsData, rewardsList.rewards))
		else:
			rewardsData = np.hstack((rewardsData, rewardsList.rewards))
	
	return rewardsData
	

# Transforms the rewards to 'Rewards', 'Returns', 'Failures', 'Average-Rewards' 
def transform_data(alg, rewardsData, transformation='Returns', window=0):
	transformedData = np.array([])

	for run in range(len(rewardsData)):
		if run % 10 == 0:
			print(run, alg)

		if transformation == 'Rewards':
			transformedRewards = rewardsData[run]


		# Returns are equal to sum of rewards
		if transformation == 'Returns':	
			returnsList = np.cumsum(rewardsData[run])
			transformedRewards = returnsList
		
		# Failures are equal to negative returns
		if transformation == 'Failures':
			returnsList = np.cumsum(rewardsData[run])
			failuresList = -1 * returnsList
			transformedRewards = failuresList
		
		# Average rewards are calculated in a moving average manner 
		# over a sliding window using the np.convolve method
		if transformation == 'Average-Rewards':
			rewardsList = rewardsData[run]

			# Change this code carefully
			averageRewardsList = np.convolve(rewardsList, np.ones(window)/window, 'valid')
			transformedRewards = averageRewardsList

		if transformedData.size != 0:
			transformedData = np.vstack((transformedData, transformedRewards))
		else:
			transformedData = np.hstack((transformedData, transformedRewards))

	return transformedData
