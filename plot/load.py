import csv
import os
import numpy as np

def load_data(alg_path):
	dir_files = os.listdir(alg_path)
	rewards_files = np.array([i for i in dir_files if 'rewards' in i])
	numruns = len(rewards_files)

	returns_list_files = np.array([])

	for file in rewards_files:
		rewards_csv = csv.reader(open(alg_path+'/'+file))
		next(rewards_csv) #skips the first line

		rewards_list = np.array([float(reward) for row in rewards_csv for reward in row])
		returns_list = np.zeros(len(rewards_list))

		returns_list[0] = rewards_list[0]

		for i in range(1,len(rewards_list)):
			returns_list[i] = returns_list[i-1] + rewards_list[i]

		if returns_list_files.size != 0:
			returns_list_files = np.vstack((returns_list_files, returns_list))
		else:
			returns_list_files = np.hstack((returns_list_files, returns_list))

	return returns_list_files
