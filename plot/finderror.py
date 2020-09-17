import os
basepath = '/home/archit/scratch/cartpoles/data/hyperparam/cartpole/offline_learning/esarsa-adam/'
dirs = os.listdir(basepath)
string = ''''''
for dir in dirs:
	print(dir)
	subbasepath = basepath + dir + '/'
	subdirs = os.listdir(subbasepath)
	for subdir in subdirs:
		print(subdir)
		subsubbasepath = subbasepath + subdir + '/'
		subsubdirs = os.listdir(subsubbasepath)
		string += subsubbasepath + '\n'
		content = []
		for i in range(0,len(subsubdirs)-1):
			for j in range(i+1, len(subsubdirs)):
				a = os.system('diff ' + subsubbasepath + subsubdirs[i] + '/log_json.txt ' + subsubbasepath + subsubdirs[j] + '/log_json.txt')
				content.append([a, subsubdirs[i], subsubdirs[j]])
		filteredcontent = [i for i in content if i[0] == 0]
		for i in range(len(filteredcontent)):
			string += ' and '.join(filteredcontent[i][1:])
			if i != len(filteredcontent) - 1:
				string += ', '
		string += '\n\n'

f = open('offlinelearningerrors.txt','w')
f.write(string)
f.close()
