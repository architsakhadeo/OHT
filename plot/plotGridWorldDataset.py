import csv
import matplotlib.pyplot as plt
#optimallabel = 'optimal-policy'
#label = 'nearOptimalPolicyTracesWorse'

optimalPath = '../data/hyperparam/gridworld/Traces/optimal-policy/traces-0.csv' 
tracesPath = '../data/hyperparam/gridworld/Traces/optimal-policy-exploringstarts/traces-2.csv'

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

fig, ax = plt.subplots(figsize=[6,6])
ax.set_aspect("equal")
plt.plot([1 for i in range(5)], [i for i in range(5)], color="black", lw=0.2)
plt.plot([2 for i in range(5)], [i for i in range(5)], color="black", lw=0.2)
plt.plot([3 for i in range(5)], [i for i in range(5)], color="black", lw=0.2)
plt.plot([i for i in range(5)], [1 for i in range(5)], color="black", lw=0.2)
plt.plot([i for i in range(5)], [2 for i in range(5)], color="black", lw=0.2)
plt.plot([i for i in range(5)], [3 for i in range(5)], color="black", lw=0.2)

def plot(plt, Path, color, lw, linestyle, marker):
    filehandler = open(Path,'r')
    contents = csv.reader(filehandler)

    state0 = []
    state1 = []

    for row in contents:
        state0.append(row[0][1])
        state1.append(row[0][3])

    state0[0] = 0
    state1[0] = 0

    for i in range(len(state0)):
        state0[i] =  int(state0[i]) + 0.5
        state1[i] =  int(state1[i]) + 0.5

    plt.scatter(state0, state1, color='none', edgecolor=color, lw=lw, marker=marker)
    plt.plot(state0, state1, color=color, lw=lw, linestyle=linestyle)


plot(plt, tracesPath, colors[0] , 2.0, '-', '.')
plot(plt, optimalPath, 'black', 0.5, 'dashed', '.')

plt.xticks([0.5, 1.5, 2.5, 3.5], ['0', '1', '2', '3'])
plt.yticks([0.5, 1.5, 2.5, 3.5], ['0', '1', '2', '3'])

plt.text(0.4, 0.75, "Start")
plt.text(3.4, 3.75, "Goal")

plt.xlabel("x coordinates")
plt.ylabel("y coordinates")
plt.title("Random Policy data coverage")
plt.xlim([0, 4])
plt.ylim([0,4])
plt.savefig('random-policy.png')
plt.show()
'''
Scatter plot of these states in a box
'''