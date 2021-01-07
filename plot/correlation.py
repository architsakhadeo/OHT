import numpy as np
import matplotlib.pyplot as plt

realInfo = [('alpha=0.01_tiles=4_tilings=32', -196.58649841865102), ('alpha=0.01_tiles=2_tilings=32',
-281.09513959270595), ('alpha=0.01_tiles=4_tilings=4', -349.93440084839887), ('alpha=0.01_tiles=2_tilings=16', -355.8693052000867), ('alpha=0.1_tiles=1_tilings=32', -396.1450966702045), ('alpha=0.01_tiles=2_tilings=8', -421.35087569591764), ('alpha=0.01_tiles=4_tilings=16', -431.18388716950284), ('alpha=0.01_tiles=4_tilings=8', -484.2383930537226), ('alpha=0.1_tiles=4_tilings=32', -488.3103346218693), ('alpha=0.1_tiles=4_tilings=16', -493.1795960191387), ('alpha=0.01_tiles=2_tilings=4', -542.0149737382677), ('alpha=0.1_tiles=4_tilings=8', -575.9572159796596), ('alpha=0.1_tiles=2_tilings=32', -595.6034764043156), ('alpha=0.1_tiles=4_tilings=4', -607.6981469440556), ('alpha=0.1_tiles=1_tilings=16', -651.0499412252101), ('alpha=0.01_tiles=4_tilings=2', -696.5443107935486), ('alpha=0.1_tiles=2_tilings=16', -709.3592373266366), ('alpha=0.01_tiles=1_tilings=16', -777.6791933612324), ('alpha=0.01_tiles=1_tilings=32', -818.154773931034), ('alpha=0.01_tiles=1_tilings=8', -897.118189123462), ('alpha=0.1_tiles=2_tilings=8', -1026.1253952726406), ('alpha=0.1_tiles=2_tilings=4', -1318.742006919053), ('alpha=0.1_tiles=1_tilings=8', -1374.7521170268194), ('alpha=0.1_tiles=4_tilings=2', -1512.9166725702446), ('alpha=0.01_tiles=4_tilings=1', -1514.3145599037816), ('alpha=0.01_tiles=2_tilings=2', -1563.5457276644006), ('alpha=0.01_tiles=1_tilings=4', -1774.7962108054805), ('alpha=0.01_tiles=1_tilings=2', -2300.3190991785214), ('alpha=0.01_tiles=2_tilings=1', -2363.9160473559655), ('alpha=0.1_tiles=1_tilings=4', -2976.233599130296), ('alpha=0.1_tiles=2_tilings=2', -3256.6019536686167), ('alpha=0.1_tiles=4_tilings=1', -3358.841318929577), ('alpha=0.1_tiles=1_tilings=2', -4012.0496610682226), ('alpha=0.1_tiles=2_tilings=1', -4410.912443112588), ('alpha=0.1_tiles=1_tilings=1', -19523.0500461657), ('alpha=0.01_tiles=1_tilings=1', -21235.297669020307), ('alpha=1_tiles=1_tilings=1', -25077.534309712526), ('alpha=1_tiles=1_tilings=32', -25669.324509711772), ('alpha=1_tiles=1_tilings=16', -25696.381370119067), ('alpha=1_tiles=1_tilings=8', -25726.943122775225), ('alpha=1_tiles=2_tilings=32', -25775.11852847144), ('alpha=1_tiles=2_tilings=16', -25777.307080078714), ('alpha=1_tiles=1_tilings=4', -25792.131314790025), ('alpha=1_tiles=2_tilings=8', -25798.60221838609), ('alpha=1_tiles=2_tilings=4', -25818.41491358663), ('alpha=1_tiles=1_tilings=2', -25838.934919694784), ('alpha=1_tiles=4_tilings=2', -25843.279486580977), ('alpha=1_tiles=2_tilings=2', -25844.24749116683), ('alpha=1_tiles=2_tilings=1', -25848.46483688969), ('alpha=1_tiles=4_tilings=1', -25854.97074471035), ('alpha=1_tiles=4_tilings=4', -25855.811774519014),
('alpha=1_tiles=4_tilings=32', -25860.43416639995), ('alpha=1_tiles=4_tilings=16', -25862.163309049167), ('alpha=1_tiles=4_tilings=8', -25863.228550409374)]

modelInfo = [('alpha=0.1_tiles=4_tilings=8', -22.28533531395865), ('alpha=0.1_tiles=4_tilings=32', -24.081940767148886), ('alpha=0.1_tiles=4_tilings=16', -27.705738429303462), ('alpha=0.1_tiles=4_tilings=4', -32.12230474359338), ('alpha=0.01_tiles=4_tilings=8', -33.81498471163709), ('alpha=0.1_tiles=4_tilings=2', -36.912271408492614), ('alpha=0.01_tiles=4_tilings=4', -37.315628312679586), ('alpha=0.1_tiles=2_tilings=16', -41.13852644406956), ('alpha=0.01_tiles=4_tilings=2', -45.44588858779877), ('alpha=0.1_tiles=2_tilings=32', -46.50352978102515), ('alpha=0.01_tiles=4_tilings=16', -47.3356505247292), ('alpha=0.01_tiles=4_tilings=32',
-49.6255875822053), ('alpha=0.1_tiles=2_tilings=8', -50.972757897595), ('alpha=0.1_tiles=1_tilings=32', -67.84451566964398), ('alpha=0.01_tiles=4_tilings=1', -69.87306064392966), ('alpha=0.1_tiles=2_tilings=4', -71.23485703109786), ('alpha=0.1_tiles=1_tilings=16', -76.13113055528905), ('alpha=0.1_tiles=4_tilings=1', -77.30397643003484), ('alpha=0.1_tiles=1_tilings=8', -91.69561143282593), ('alpha=0.1_tiles=2_tilings=2', -107.15131243351546), ('alpha=0.01_tiles=2_tilings=16', -145.03484595756268), ('alpha=0.1_tiles=1_tilings=4', -155.90799535645357), ('alpha=0.01_tiles=2_tilings=32', -158.2323255430797), ('alpha=0.01_tiles=2_tilings=8', -195.44158905034968), ('alpha=0.01_tiles=2_tilings=4', -202.3677250429147),
('alpha=0.01_tiles=2_tilings=1', -238.0218419883737), ('alpha=0.01_tiles=1_tilings=16', -247.44692188555888), ('alpha=0.01_tiles=1_tilings=32', -256.3268211436725), ('alpha=0.1_tiles=2_tilings=1', -318.7885268222106), ('alpha=0.01_tiles=2_tilings=2', -322.1330636526119), ('alpha=0.01_tiles=1_tilings=8', -329.8470186584997), ('alpha=0.01_tiles=1_tilings=4', -332.67135612065806), ('alpha=0.1_tiles=1_tilings=2', -362.6327247351878), ('alpha=0.01_tiles=1_tilings=2', -413.37102368287236), ('alpha=0.01_tiles=1_tilings=1', -1491.1787334794092), ('alpha=0.1_tiles=1_tilings=1', -1691.3513799000127), ('alpha=1_tiles=1_tilings=32', -2903.354691304857), ('alpha=1_tiles=1_tilings=1', -2917.4947196317526), ('alpha=1_tiles=2_tilings=1', -3060.1327622160625), ('alpha=1_tiles=4_tilings=1', -3111.745866536779), ('alpha=1_tiles=2_tilings=8', -3117.759873151041), ('alpha=1_tiles=1_tilings=16', -3169.730481490806), ('alpha=1_tiles=1_tilings=4', -3177.98356504522), ('alpha=1_tiles=4_tilings=16', -3197.2791966501773), ('alpha=1_tiles=1_tilings=8', -3332.9638018085484), ('alpha=1_tiles=2_tilings=16', -3339.1873454091856), ('alpha=1_tiles=2_tilings=32', -3350.8958468829464), ('alpha=1_tiles=2_tilings=2', -3383.01897724332), ('alpha=1_tiles=1_tilings=2', -3390.1549745498282), ('alpha=1_tiles=2_tilings=4', -3418.9998305088957), ('alpha=1_tiles=4_tilings=8', -3424.1675747998415), ('alpha=1_tiles=4_tilings=4', -3443.0769224990254), ('alpha=1_tiles=4_tilings=2', -3453.4126696413937), ('alpha=1_tiles=4_tilings=32', -3461.9918424086395)]


realkeys = [key for (key, value) in realInfo]
realvalues = [value for (key, value) in realInfo]
modelkeys = [key for (key, value) in modelInfo]
modelvalues = [value for (key, value) in modelInfo]

rankingUnderReal = [i for i in range(len(realkeys))]
rankingUnderModel = [realkeys.index(modelkeys[i]) for i in range(len(modelkeys))]

correlation = np.corrcoef(rankingUnderReal, rankingUnderModel)
print(correlation)

def plotCorrelationPlot(rankingUnderReal, rankingUnderModel, plt):
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    plt.scatter(rankingUnderReal, rankingUnderModel)
    plt.plot([i+1 for i in range(len(rankingUnderReal))], [j+1 for j in range(len(rankingUnderReal))], '--', color='black', linewidth=0.75)
    plt.xlabel('Ranking in the real environment', labelpad=35)
    plt.ylabel('Ranking in the\noffline model', rotation=0, labelpad=55)
    plt.rcParams['figure.figsize'] = [10, 5]
    plt.text(-1, 50, 'Ideal correlation (dashed line) = 1.0', color='black', fontsize=8)
    plt.text(-1, 45, 'Correlation between the rankings = 0.86', color=colors[0], fontsize=8)
    plt.tight_layout()
    plt.savefig('../img/finalPlots/correlation.png',dpi=300, bbox_inches='tight')
    #plt.show()


def plotPerformances(realvalues, modelvalues, plt):
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    plt.scatter([i+1 for i in range(len(realvalues))], np.array(realvalues)/250000, label='Performance in\nthe real environment')
    modelvaluesRankedByRealRanking = [modelvalues[rankingUnderModel.index(rankingUnderReal[i])] for i in range(len(rankingUnderReal))]
    plt.scatter([i+1 for i in range(len(modelvalues))], np.array(modelvaluesRankedByRealRanking)/50000, label='Performance in\nthe offline model')
    plt.scatter([12], np.array(modelvaluesRankedByRealRanking[11]/50000), color=colors[2])
    plt.scatter([12], np.array(modelvaluesRankedByRealRanking[11]/50000), facecolors='none', edgecolors=colors[2], s=160)
    #plt.scatter([i for i in range(len(modelvalues))], np.array(modelvalues)/50000, label='Average reward in the offline model')
    plt.xlabel('Hyperparameter ranking in the real environment', labelpad=35)
    plt.ylabel('Average reward\nof each\nhyperparameter setting\n(AUC)', rotation=0, labelpad=55)
    #plt.rcParams['figure.figsize'] = [12, 8]
    plt.arrow(15, -0.04, -3, 0.035, color='black', width=0.00005, length_includes_head=True, head_length=0.002, head_width=0.002)
    plt.text(5, -0.05, 'Hyperparameters chosen\nby the offline model', fontsize=8)
    plt.legend(loc=(0.01, 0.1), prop={'size': 8})
    #plt.tight_layout()
    plt.savefig('../img/finalPlots/performances.png',dpi=300, bbox_inches='tight')
    #plt.show()

'''
plotCorrelationPlot(rankingUnderReal, rankingUnderModel, plt)
'''

plotPerformances(realvalues, modelvalues, plt)
