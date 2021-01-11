import numpy as np
import matplotlib.pyplot as plt

'''
# All AUC

realInfo = [('param_71', -1020.29464108163), ('param_62', -1075.1522590550128), ('param_70', -1082.3827016315643), ('param_61', -1120.5491096689525), ('param_80', -1147.4952825658631), ('param_31', -1155.8382406861404), ('param_43', -1159.429652872701), ('param_34', -1171.993548012208), ('param_60', -1207.4688935751203), ('param_79', -1218.428247712323), ('param_52', -1252.1491628976928), ('param_33', -1276.7073286305674), ('param_54', -1342.0098775130427), ('param_38', -1342.2048001602527), ('param_57', -1361.1218864865245), ('param_69', -1379.5443212491691), ('param_47', -1411.7457695217652), ('param_55', -1416.3471998354669), ('param_44', -1438.169712646495), ('param_53', -1441.358110603237), ('param_40', -1449.8322133408171), ('param_35', -1458.5377672960424), ('param_50', -1471.5957979083303), ('param_49', -1497.6583604160098), ('param_78', -1513.4167591102237), ('param_41', -1521.743989957769), ('param_56', -1548.1248957675746), ('param_42', -1630.1157178030844), ('param_32', -1787.2082519660808), ('param_51', -1800.4518280567547), ('param_65', -1984.5548105458045), ('param_64', -2017.182135526055), ('param_27', -2280.5242442080307), ('param_37', -2424.4241862102135), ('param_63', -2426.9072632382185), ('param_66', -2480.2797830719433), ('param_74', -2484.122930679087), ('param_73', -2584.680438531496), ('param_46', -2655.1511629985334), ('param_72', -2788.3093820654854), ('param_75', -2987.6968332501738), ('param_36', -3363.5162364328603), ('param_45', -3861.184727579764), ('param_30', -4989.902904128464), ('param_48', -6111.475941040648), ('param_58', -7569.381582381156), 
('param_76', -9349.674695064296), ('param_67', -9396.300085335106), ('param_28', -11413.40023021222), ('param_59', -12254.6580094332), ('param_18', -12327.506155131627), ('param_9', -12603.287409816905), ('param_26', -12726.03376694054), ('param_25', -12805.910343007497), ('param_68', -12808.869554453902), ('param_24', -13096.039925822886), ('param_21', -13954.355368797698), ('param_15', -14501.962102076712), ('param_22', -14630.939997109164), ('param_23', -14658.299731077288), ('param_12', -14849.633811412177), ('param_77', -15500.850295470413), ('param_39', -15539.9210287606), ('param_16', -15788.434044485912), ('param_17', -16049.960889899547), ('param_19', -16365.621153211749), ('param_20', -16704.612697607397), ('param_13', -16976.317312470153), ('param_14', -17266.587720840307), ('param_10', -18579.715351111026), ('param_11', -19327.294781595232), ('param_29', -19931.023602810965), ('param_0', -24085.762687882747), ('param_3', -26525.93929799493), ('param_6', -26570.801029245966), ('param_7', -26651.095028972748), ('param_8', -26651.25424990317), ('param_5', -26666.219657577694), ('param_4', -26676.999793035848), ('param_2', -26677.33709209738), ('param_1', -26685.359956561708)]

modelKNN1Info = [('param_7', -0.40523925646231856), ('param_62', -0.40523925646231856), ('param_61', -0.40523925646231856), ('param_60', -0.40523925646231856), ('param_6', -0.40523925646231856), ('param_59', -0.40523925646231856), ('param_58', -0.40523925646231856), ('param_57', -0.40523925646231856), ('param_56', -0.40523925646231856), ('param_55', -0.40523925646231856), ('param_54', -0.40523925646231856), ('param_5', -0.40523925646231856), ('param_4', -0.40523925646231856), 
('param_35', -0.40523925646231856), ('param_34', -0.40523925646231856), ('param_33', -0.40523925646231856), ('param_32', -0.40523925646231856), ('param_31', -0.40523925646231856), ('param_30', -0.40523925646231856), ('param_3', -0.40523925646231856), ('param_29', -0.40523925646231856), ('param_28', -0.40523925646231856), ('param_27', -0.40523925646231856), ('param_2', -0.40523925646231856), ('param_1', -0.40523925646231856), ('param_0', -0.40523925646231856), ('param_8', -0.528498717589788), ('param_67', -32.139939040871134), ('param_65', -32.822803970989725), ('param_68', -33.56729691981503), ('param_70', -33.89733230992214), ('param_71', -34.49659365570295), ('param_43', -35.025834521762306), ('param_44', -37.48831981861228), ('param_41', -47.330391373640914), ('param_64', -48.82543985197132), ('param_38', -49.1484535142368), ('param_69', -53.83383501475416), ('param_40', -59.36328192078482), ('param_79', -62.78666575976825), ('param_77', -63.04604223478786), ('param_76', -65.99160125454756), ('param_80', -66.29165479781274), ('param_37', -68.55861334110203), ('param_42', -70.53791739676568), ('param_74', -72.96392252045332), ('param_66', -83.96743984867602), ('param_52', -87.61868821428139), ('param_63', -89.82829078948397), ('param_53', -90.2803498959462), ('param_73', -98.06808192141746), ('param_47', -100.14928162999446), ('param_50', -100.33256953147087), ('param_39', -100.80152948420586), ('param_78', -103.25166120668362), ('param_36', -113.46890149300106), ('param_49', -119.64725939489885), ('param_51', -127.25826178432656), ('param_46', -136.06779679727578), ('param_75', -149.00728342887945), ('param_72', -150.495068586298), ('param_48', -198.22910952645807), ('param_45', -208.35486257250844), ('param_17', -309.0001678327685), ('param_16', -325.379913011318), ('param_14', -386.310578702892), ('param_13', -396.03963833535056), ('param_11', -428.31187513995474), ('param_26', -431.54700820526097), ('param_25', -435.1276038434875), ('param_23', -460.9948551142972), ('param_15', -472.5420473872733), ('param_22', -480.42777000139563), ('param_10', -512.0263730916158), ('param_20', -522.3943218768411), ('param_24', -550.272429472032), ('param_9', -559.6134474415659), ('param_19', -594.0537749137873), ('param_12', -615.3397103098811), ('param_21', -698.7998565320883), ('param_18', -710.0653655983511)]

modelKNN3Info = [('param_62', -0.5293864212943578), ('param_61', -0.5293864212943578), ('param_60', -0.5293864212943578), ('param_59', -0.5293864212943578), ('param_58', -0.5293864212943578), ('param_57', -0.5293864212943578), ('param_56', -0.5293864212943578), ('param_55', -0.5293864212943578), ('param_54', -0.5293864212943578), ('param_5', -0.5293864212943578), ('param_4', -0.5293864212943578), ('param_35', -0.5293864212943578), ('param_34', -0.5293864212943578), ('param_33', -0.5293864212943578), ('param_32', -0.5293864212943578), ('param_31', -0.5293864212943578), ('param_29', -0.5293864212943578), ('param_28', -0.5293864212943578), ('param_27', -0.5293864212943578), ('param_2', -0.5293864212943578), ('param_1', -0.5293864212943578), ('param_0', -0.596053089478182), ('param_3', -0.601747595125824), ('param_7', -0.703824781790022), ('param_8', -0.7878173624708502), ('param_30', -1.2936654787461261), ('param_6', -5.695721604039751), ('param_68', -48.54475674896058), ('param_71', -54.04350958968873), ('param_67', -70.07261712942989), ('param_77', -77.5696571549996), ('param_70', -77.78467427427663), ('param_76', -79.20160411078491), ('param_80', -108.7328160355292), ('param_69', -111.02841025611004), ('param_79', -112.46053547557351), 
('param_65', -117.34879338453985), ('param_41', -121.30516390812018), ('param_44', -122.16302749202), ('param_43', -125.78951964657897), ('param_66', -127.85894438767073), ('param_40', -128.94588673400918), ('param_49', -139.59027038350058), ('param_38', -144.03334166737795), ('param_47', -148.02789557394854), ('param_64', -148.3384653886698), ('param_52', -149.71050178775985), ('param_78', -156.75819586896174), ('param_53', -158.91239402698153), ('param_50', -164.28539226295774), ('param_42', -172.79629210626396), ('param_75', -173.5604073216637), ('param_74', -184.12933380092252), ('param_63', -185.55950188108372), ('param_36', -195.02743836493693), ('param_39', -210.69180974760732), ('param_46', -220.27010903984146), ('param_73', -220.6979653799519), ('param_37', -221.60189044090117), ('param_51', -221.890514562778), ('param_48', -270.81032412979255), ('param_45', -273.3294783228498), ('param_72', -282.9559286373669), ('param_25', -570.7281929186706), ('param_26', -586.9657151989873), ('param_14', -594.1766728881294), ('param_16', -620.0985042849078), ('param_17', -632.7984475370446), ('param_22', -649.0553816841473), ('param_23', -666.7038905348513), ('param_15', -718.3766006341708), ('param_13', -736.6898019313074), ('param_9', -742.1968979556151), ('param_24', -752.301352385767), ('param_20', -766.889525345473), ('param_11', -786.6824588711313), ('param_10', -796.5078893790342), ('param_19', -834.6761488332027), ('param_18', -835.4866308029624), ('param_21', -864.150147338398), ('param_12', -867.4541210289691)]
'''

# Real AUC, Models bottom 10th%

realInfo = [('param_71', -1020.29464108163), ('param_62', -1075.1522590550128), ('param_70', -1082.3827016315643), ('param_61', -1120.5491096689525), ('param_80', -1147.4952825658631), ('param_31', -1155.8382406861404), ('param_43', -1159.429652872701), ('param_34', -1171.993548012208), ('param_60', -1207.4688935751203), ('param_79', -1218.428247712323), ('param_52', -1252.1491628976928), ('param_33', -1276.7073286305674), ('param_54', -1342.0098775130427), ('param_38', -1342.2048001602527), ('param_57', -1361.1218864865245), ('param_69', -1379.5443212491691), ('param_47', -1411.7457695217652), ('param_55', -1416.3471998354669), ('param_44', -1438.169712646495), ('param_53', -1441.358110603237), ('param_40', -1449.8322133408171), ('param_35', -1458.5377672960424), ('param_50', -1471.5957979083303), ('param_49', -1497.6583604160098), ('param_78', -1513.4167591102237), ('param_41', -1521.743989957769), ('param_56', -1548.1248957675746), ('param_42', -1630.1157178030844), ('param_32', -1787.2082519660808), ('param_51', -1800.4518280567547), ('param_65', -1984.5548105458045), ('param_64', -2017.182135526055), ('param_27', -2280.5242442080307), ('param_37', -2424.4241862102135), ('param_63', -2426.9072632382185), ('param_66', -2480.2797830719433), ('param_74', -2484.122930679087), ('param_73', -2584.680438531496), ('param_46', -2655.1511629985334), ('param_72', -2788.3093820654854), ('param_75', -2987.6968332501738), ('param_36', -3363.5162364328603), ('param_45', -3861.184727579764), ('param_30', -4989.902904128464), ('param_48', -6111.475941040648), ('param_58', -7569.381582381156), 
('param_76', -9349.674695064296), ('param_67', -9396.300085335106), ('param_28', -11413.40023021222), ('param_59', -12254.6580094332), ('param_18', -12327.506155131627), ('param_9', -12603.287409816905), ('param_26', -12726.03376694054), ('param_25', -12805.910343007497), ('param_68', -12808.869554453902), ('param_24', -13096.039925822886), ('param_21', -13954.355368797698), ('param_15', -14501.962102076712), ('param_22', -14630.939997109164), ('param_23', -14658.299731077288), ('param_12', -14849.633811412177), ('param_77', -15500.850295470413), ('param_39', -15539.9210287606), ('param_16', -15788.434044485912), ('param_17', -16049.960889899547), ('param_19', -16365.621153211749), ('param_20', -16704.612697607397), ('param_13', -16976.317312470153), ('param_14', -17266.587720840307), ('param_10', -18579.715351111026), ('param_11', -19327.294781595232), ('param_29', -19931.023602810965), ('param_0', -24085.762687882747), ('param_3', -26525.93929799493), ('param_6', -26570.801029245966), ('param_7', -26651.095028972748), ('param_8', -26651.25424990317), ('param_5', -26666.219657577694), ('param_4', -26676.999793035848), ('param_2', -26677.33709209738), ('param_1', -26685.359956561708)]

modelKNN1Info =

modelKNN3Info = 

realkeys = [key for (key, value) in realInfo]
realvalues = [value for (key, value) in realInfo]
modelKNN1keys = [key for (key, value) in modelKNN1Info]
modelKNN1values = [value for (key, value) in modelKNN1Info]
modelKNN3keys = [key for (key, value) in modelKNN3Info]
modelKNN3values = [value for (key, value) in modelKNN3Info]

rankingUnderReal = [i for i in range(len(realkeys))]
rankingUnderModelKNN1 = [realkeys.index(modelKNN1keys[i]) for i in range(len(modelKNN1keys))]
rankingUnderModelKNN3 = [realkeys.index(modelKNN3keys[i]) for i in range(len(modelKNN3keys))]

correlation = np.corrcoef(rankingUnderReal, rankingUnderModelKNN3)
print(correlation)


def plotCorrelationPlot(rankingUnderReal, rankingUnderModelKNN1, plt):
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    plt.scatter(rankingUnderReal, rankingUnderModelKNN1, s = 5, color=colors[2])
    plt.plot([i+1 for i in range(len(rankingUnderReal))], [j+1 for j in range(len(rankingUnderReal))], '--', color='black', linewidth=0.75)
    plt.xlabel('Ranking in the real environment', labelpad=35)
    plt.ylabel('Ranking in the\noffline model KNN3', rotation=0, labelpad=55)
    plt.rcParams['figure.figsize'] = [10, 5]
    plt.text(4, 65, 'Ideal correlation (dashed line) = 1.0', color='black', fontsize=8)
    plt.text(4, 60, 'Correlation between the rankings = 0.32', color=colors[2], fontsize=8)
    plt.tight_layout()
    plt.savefig('../img/correlationKNN3.png',dpi=300, bbox_inches='tight')
    #plt.show()


def plotPerformances(realvalues, modelKNN1values, plt):
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    plt.scatter([i+1 for i in range(len(realvalues))], np.array(realvalues)/250000, label='Performance in\nthe real environment')
    modelKNN1valuesRankedByRealRanking = [modelKNN1values[rankingUnderModelKNN1.index(rankingUnderReal[i])] for i in range(len(rankingUnderReal))]
    plt.scatter([i+1 for i in range(len(modelKNN1values))], np.array(modelKNN1valuesRankedByRealRanking)/250000, label='Performance in\nthe offline modelKNN1')
    #plt.scatter([12], np.array(modelKNN1valuesRankedByRealRanking[11]/250000), color=colors[2])
    #lt.scatter([12], np.array(modelKNN1valuesRankedByRealRanking[11]/250000), facecolors='none', edgecolors=colors[2], s=160)
    #plt.scatter([i for i in range(len(modelKNN1values))], np.array(modelKNN1values)/50000, label='Average reward in the offline modelKNN1')
    plt.xlabel('Hyperparameter ranking in the real environment', labelpad=35)
    plt.ylabel('Average reward\nof each\nhyperparameter setting\n(AUC)', rotation=0, labelpad=55)
    #plt.rcParams['figure.figsize'] = [12, 8]
    plt.arrow(15, -0.04, -3, 0.035, color='black', width=0.00005, length_includes_head=True, head_length=0.002, head_width=0.002)
    plt.text(5, -0.05, 'Hyperparameters chosen\nby the offline modelKNN1', fontsize=8)
    plt.legend(loc=(0.01, 0.1), prop={'size': 8})
    #plt.tight_layout()
    plt.savefig('../img/performances.png',dpi=300, bbox_inches='tight')
    #plt.show()


def plotPerformancesMultiple(realvalues, modelKNN1values, modelKNN3values, plt):
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    #plt.scatter([i+1 for i in range(len(realvalues))], np.array(realvalues)/250000, label='Performance in\nthe real environment', s = 5)
    modelKNN1valuesRankedByRealRanking = [modelKNN1values[rankingUnderModelKNN1.index(rankingUnderReal[i])] for i in range(len(rankingUnderReal))]
    plt.scatter([i+1 for i in range(len(modelKNN1values))], np.array(modelKNN1valuesRankedByRealRanking)/250000, label='Performance in\nthe offline model KNN1', s = 5, color=colors[1])
    #plt.scatter([realkeys.index(modelKNN1keys[0])+1], np.array(modelKNN1valuesRankedByRealRanking[realkeys.index(modelKNN1keys[0])]/250000), color=colors[1])
    plt.scatter([realkeys.index(modelKNN1keys[0])+1], np.array(modelKNN1valuesRankedByRealRanking[realkeys.index(modelKNN1keys[0])]/250000), facecolors='none', edgecolors=colors[1], s=160)

    modelKNN3valuesRankedByRealRanking = [modelKNN3values[rankingUnderModelKNN3.index(rankingUnderReal[i])] for i in range(len(rankingUnderReal))]
    plt.scatter([i+1 for i in range(len(modelKNN3values))], np.array(modelKNN3valuesRankedByRealRanking)/250000, label='Performance in\nthe offline model KNN3', s = 5, color=colors[2])
    #plt.scatter([12], np.array(modelKNN3valuesRankedByRealRanking[11]/250000), color=colors[2])
    plt.scatter([realkeys.index(modelKNN3keys[0])+1], np.array(modelKNN3valuesRankedByRealRanking[realkeys.index(modelKNN1keys[0])]/250000), facecolors='none', edgecolors=colors[2], s=160)

    #plt.plot([i+1 for i in range(len(realvalues))], [-0.005 for i in range(len(realvalues))], '--', color='black', linewidth=0.75)

    #plt.scatter([i for i in range(len(modelKNN1values))], np.array(modelKNN1values)/50000, label='Average reward in the offline modelKNN1')
    plt.xlabel('Hyperparameter ranking in the real environment', labelpad=35)
    plt.ylabel('Average reward\nof each\nhyperparameter setting\n(AUC)', rotation=0, labelpad=55)
    plt.rcParams['figure.figsize'] = [16, 8]
    #plt.arrow(15, -0.04, -3, 0.035, color='black', width=0.00005, length_includes_head=True, head_length=0.002, head_width=0.002)
    #plt.text(5, -0.05, 'Hyperparameters chosen\nby the offline modelKNN1', fontsize=8)
    plt.legend(loc=(0.01, 0.1), prop={'size': 8})
    #plt.tight_layout()
    plt.savefig('../img/performancesKNN1,3.png',dpi=300, bbox_inches='tight')
    #plt.show()


#plotCorrelationPlot(rankingUnderReal, rankingUnderModelKNN3, plt)


plotPerformancesMultiple(realvalues, modelKNN1values, modelKNN3values, plt)
