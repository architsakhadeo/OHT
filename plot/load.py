import csv
import os
import numpy as np

def load_data(alg_path):
	dir_files = os.listdir(alg_path)
	rewards_files = np.array([i for i in dir_files if 'rewards' in i])
	numruns = len(rewards_files)

	returns_list_files = np.array([])

	for file in rewards_files:
		f = open(alg_path+'/'+file)
		rewards_list = np.loadtxt(f, skiprows=1).T # Import CSV and skip the header line

		returns_list = np.cumsum(rewards_list)

		if returns_list_files.size != 0:
			returns_list_files = np.vstack((returns_list_files, returns_list))
		else:
			returns_list_files = np.hstack((returns_list_files, returns_list))

	return returns_list_files
