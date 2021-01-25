import matplotlib.pyplot as plt
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

labels = ['Lambda', 'Epsilon', 'Alpha-SGD']
iterationsScores = [-207.33333333333334, -38, -60.333333333333336, -35.666666666666664,-9.666666666666666, -16.333333333333332, -2.6666666666666665, -12, -2.6666666666666665, -0.6666666666666666, -5, -1.6666666666666667, -1.3333333333333333, -1, -5, -1.3333333333333333, -1.6666666666666667, -1, -1.6666666666666667, -2.6666666666666665]
tilings_tiles = [[8, 1], [16, 2], [16, 2], [8, 2], [8, 2], [8, 2], [32, 2], [32, 2], [16, 2], [16, 2], [32, 2], [16, 2], [16, 2], [32, 2], [16, 2], [32, 2], [16, 2], [32, 2], [32, 2], [16, 2]]

tilings = [tilings_tiles[i][0] for i in range(len(tilings_tiles))]
tiles = [tilings_tiles[i][1] for i in range(len(tilings_tiles))]

lambda_epsilon_sgd = [[0.6257928587087216, 0.03838832456798952, 0.012741350875693558], [0.39139579380003625, 0.03210049816412274, 0.019125022625848098], [0.9706933552429555, 0.05751093630136722, 0.031249676774067092], [0.9290973995507676, 0.013077139806528186, 0.005884724672330033], [0.8601456734933349, 0.006274720069017163, 0.004856787017492804], [0.7765624976385445, 0.02734749307719596, 0.008470479914014459], [0.6986426291543981, 0.06136479757377217, 0.00711650170267666], [0.9099464576498103, 0.07553253543258005, 0.003690029893858014], [0.9149325757318141, 0.06718467747217337, 0.003661829988894547], [0.9450992991807318, 0.0675835618600657, 0.005134768485025485], [0.9139361248027753, 0.050539621582500166, 0.009153490497106578], [0.9097569421130554, 0.06032641625526657, 0.011322149287099103], [0.9212051072593059, 0.07059219530529735, 0.009307958400713263], [0.789540114248514, 0.0818781659712849, 0.005655830641877416], [0.9956461938768759, 0.10858890418481654, 0.004758361682254401], [0.9471190286452326, 0.04183254435186376, 0.009252501061497087], [0.9308973831826439, 0.09736780203389296, 0.004436313526947109], [0.9013624242933974, 0.04086764212195326, 0.008359615842549059], [0.8902126412838645, 0.05574995469207301, 0.004742571710342393], [0.9421872712448776, 0.08103676419443369, 0.0067308481255901435]]
lambdav = [lambda_epsilon_sgd[i][0] for i in range(len(lambda_epsilon_sgd))]
epsilon = [lambda_epsilon_sgd[i][1] for i in range(len(lambda_epsilon_sgd))]
sgd = [lambda_epsilon_sgd[i][2] for i in range(len(lambda_epsilon_sgd))]


iterationNumbers = [i+1 for i in range(len(iterationsScores))]

iterationNumbersXticks = [str(i+1) for i in range(len(iterationNumbers))]
for i in range(0, len(iterationNumbers), 2):
    iterationNumbersXticks[i] = ""

ytickstilings = [str(tilings[i]) for i in range(len(iterationNumbers))]
ytickstiles = [str(tiles[i]) for i in range(len(iterationNumbers))]
plt.plot(iterationNumbers, [0 for i in range(len(iterationNumbers))], '--', color='black', linewidth=0.5)
plt.plot(iterationNumbers, iterationsScores, color=colors[0], marker='o', fillstyle='none')
#plt.plot(iterationNumbers, lambdav, color=colors[3], label=labels[0], marker='o', fillstyle='none')
#plt.plot(iterationNumbers, epsilon, color=colors[4], label=labels[1], marker='o', fillstyle='none')
#plt.plot(iterationNumbers, sgd, color=colors[5], label=labels[2], marker='o', fillstyle='none')
#plt.plot(iterationNumbers, tiles, color=colors[2], label=labels[1], marker='o', fillstyle='none')
#plt.yticks([1, 2, 4, 8, 16, 32], ['1', '2', '4', '8', '16', '32'], size=8)
plt.xticks(iterationNumbers, iterationNumbersXticks)
#plt.legend()
plt.title('Average return calculated over the last half of the run', pad=25, fontsize=10)
plt.xlabel('Iterations', labelpad=35)
plt.ylabel('Average return', rotation=0, labelpad=45)
plt.tight_layout()
#plt.show()
plt.savefig('../img/IterationsAverageReturn.png',dpi=500, bbox_inches='tight')