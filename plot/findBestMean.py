import os
import subprocess

algorithms = ['olddata/esarsa/']
basePath = '../data/'

params_failures = {}
params_count = {}


for i in range(len(algorithms)):
	if os.path.isdir(basePath + algorithms[i]) == False:
		continue
	#failuresList = []
	parameterSettings = os.listdir(basePath + algorithms[i])
	#indexMinFailures = 0
	#minFailures = 0
	for j in range(len(parameterSettings)):
		if j % 10 == 0:
			print(algorithms[i], j)

		failures = 0
		runs = os.listdir(basePath + algorithms[i] + '/' + parameterSettings[j])
		count = 0
		for k in range(len(runs)):
			if 'episodes' in runs[k]:
				count += 1

				# subtract 1 for the first row
				failures += int(subprocess.check_output('cat ' + basePath + algorithms[i] + '/' + parameterSettings[j] + '/' + runs[k] + ' | wc -l', shell=True)) - 1
		failures /= count

		params_failures[parameterSettings[j]] = failures
		params_count[parameterSettings[j]] = count

		#if len(failuresList) != 0:
		#	if failures < minFailures:
		#		indexMinFailures = j
		#		minFailures = failures
		#else:
		#	minFailures = failures

		#failuresList.append(failures)
	
	#print('\nLeast number of failures for ' + algorithms[i] + ' is for the parameter setting: ' + parameterSettings[indexMinFailures])
	#print('Number of failures for that setting is: ', failuresList[indexMinFailures], '\n')

sortedlist = sorted(params_failures.items(), key=lambda x:x[1], reverse=True)
for i in sortedlist:
	print(i[0], i[1], params_count[i[0]])




