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

# Real online env, AUC
#realInfo = [('param_25', -285.8124695485127), ('param_16', -308.22540630357304), ('param_26', -313.6040946356895), ('param_13', -314.9741063953518), ('param_24', -331.5340815005979), ('param_23', -347.6194235783255), ('param_17', -360.106786245317), ('param_22', -374.7081285437368), ('param_14', -394.5975930386011), ('param_15', -395.34422109788943), ('param_11', -402.47112043780675), ('param_21', -498.91428957157666), ('param_10', -537.0473539391037), ('param_12', -621.8647194155847), ('param_18', -662.8414555071835), ('param_9', -788.6560894099841), ('param_19', -943.5648699237646), ('param_0', -2256.3771957815), ('param_3', -2551.6604838237713), ('param_6', -2570.1085502376236), ('param_20', -2774.2912733175585), ('param_7', -2909.53175425217), ('param_8', -2933.799559750072), ('param_4', -2997.4951209820333), ('param_5', -3107.8013434923473), ('param_1', -3303.218339723092), ('param_2', -3379.934509131741)]

# Random, 5K
#modelKNN1Info = [('param_26', -12.764177522868314), ('param_25', -13.117898414084209), ('param_17', -14.269629629925085), ('param_23', -16.233607700933497), ('param_16', -16.38793850765555), ('param_14', -18.200305734683216), ('param_22', -18.282720883089066), ('param_13', -21.766668211028758), ('param_20', -24.542928746398132), ('param_15', -31.64927558262611), ('param_11', -32.84418935795326), ('param_24', -34.78774530792752), ('param_10', -46.67702929906538), ('param_19', -56.687794740292766), ('param_21', -59.94960442847311), ('param_12', -64.00543361373003), ('param_18', -76.95831262637489), ('param_9', -93.89674044505414), ('param_7', -340.722291338904), ('param_8',  -343.8097408409277), ('param_2', -368.8279121847484), ('param_5', -384.8351164609141), ('param_4', -442.5031281366773), ('param_6', -507.192349892928), ('param_1', -516.2735385635699), ('param_3', -598.4641806584592), ('param_0', -682.433239703506)]

# Random, 50K
#modelKNN1Info = [('param_23', -18.968495739682968), ('param_26', -21.271527965748035), ('param_25', -22.461427903585044), ('param_17', -23.2033353618412), ('param_22', -23.986963301668496), ('param_16', -27.793213610214462), ('param_14', -30.960707489981072), ('param_20', -31.896956552198365), ('param_24', -31.952849658242094), ('param_13', -34.95601614333101), ('param_11', -50.400959226385304), ('param_15', -51.67816413679977), ('param_10', -61.873314891715765), ('param_19', -69.20302738588991), ('param_21', -73.32073921932529), ('param_12', -75.8434592425906), ('param_18', -83.92403559843467), ('param_9', -100.21888443661415), ('param_5', -262.48402192341644), ('param_2', -283.47774464356417), ('param_8', -322.4723974864214), ('param_7', -324.72013462458347), ('param_4', -334.7878793614427), ('param_1', -399.0746596328589), ('param_6', -409.903778683294), ('param_0', -423.0688498929607), ('param_3', -478.602708803116)]

# Suboptimalfixed, eps = 0.01, 5K
#modelKNN1Info = [('param_9', 0.0), ('param_8', 0.0), ('param_7', 0.0), ('param_6', 0.0), ('param_5', 0.0), ('param_4', 0.0), ('param_3', 0.0), ('param_26', 0.0), ('param_25', 0.0), ('param_24', 0.0), ('param_23', 0.0), ('param_22', 0.0), ('param_21', 0.0), ('param_20', 0.0), ('param_2', 0.0), ('param_19', 0.0), ('param_18', 0.0), ('param_17', 0.0), ('param_16', 0.0), ('param_15', 0.0), ('param_14', 0.0), ('param_13', 0.0), ('param_12', 0.0), ('param_11', 0.0), ('param_10', 0.0), ('param_1', 0.0), ('param_0', 0.0)]

# Suboptimalfixed, eps = 0.01, 50K
#modelKNN1Info = [('param_9', 0.0), ('param_8', 0.0), ('param_7', 0.0), ('param_6', 0.0), ('param_5', 0.0), ('param_4', 0.0), ('param_3', 0.0), ('param_26', 0.0), ('param_25', 0.0), ('param_24', 0.0), ('param_23', 0.0), ('param_22', 0.0), ('param_21', 0.0), ('param_20', 0.0), ('param_2', 0.0), ('param_19', 0.0), ('param_18', 0.0), ('param_17', 0.0), ('param_16', 0.0), ('param_15', 0.0), ('param_14', 0.0), ('param_13', 0.0), ('param_12', 0.0), ('param_11', 0.0), ('param_10', 0.0), ('param_1', 0.0), ('param_0', 0.0)]

# Suboptimalfixed, eps = 0.3, 5K
#modelKNN1Info = [('param_23', -2.3492100095172663), ('param_25', -3.123335288614397), ('param_26', -3.3159908276197294), ('param_17', -4.708035486031476), ('param_16', -6.634427120117443), ('param_24', -10.166876780498267), ('param_13', -10.607478364571703), ('param_22', -11.121678680348465), ('param_14', -12.297748884471558), ('param_15', -12.68465786667758), ('param_21', -14.577093169993747), ('param_11', -16.106842484344675), ('param_18', -19.478913474498015), ('param_20', -19.979676098252515), ('param_19', -20.362946695281618), ('param_0', -20.3960145831672), ('param_9', -20.51441492505736), ('param_10', -20.79499723171724), ('param_12', -20.809906696376306), ('param_7', -24.46765834000807), ('param_2', -35.33803303562148), ('param_3', -39.203614972896034), ('param_6', -46.88218127676781), ('param_5', -48.23349425089031), ('param_4', -49.9598040354527), ('param_1', -52.1337930678797), ('param_8', -60.73316686515016)]

# Suboptimalfixed, eps = 0.3, 50K
#modelKNN1Info = [('param_9', 0.0), ('param_8', 0.0), ('param_7', 0.0), ('param_6', 0.0), ('param_5', 0.0), ('param_4', 0.0), ('param_3', 0.0), ('param_26', 0.0), ('param_25', 0.0), ('param_24', 0.0), ('param_23', 0.0), ('param_22', 0.0), ('param_21', 0.0), ('param_20', 0.0), ('param_2', 0.0), ('param_19', 0.0), ('param_18', 0.0), ('param_17', 0.0), ('param_16', 0.0), ('param_15', 0.0), ('param_14', 0.0), ('param_13', 0.0), ('param_12', 0.0), ('param_11', 0.0), ('param_10', 0.0), ('param_1', 0.0), ('param_0', 0.0)]

# Learning, eps = 0.081, 5K
#modelKNN1Info = [('param_17', -18.983078479749263), ('param_11', -20.43511456453593), ('param_13', -21.249861728193697), ('param_20', -21.385199305350863), ('param_14', -21.692799468056375), ('param_26', -22.2808636428551), ('param_16', -22.30617006185688), ('param_25', -22.39367115183548), ('param_23', -23.619806815583534), ('param_24', -23.85010516781004), ('param_10', -24.187055365533475), ('param_15', -24.559149263728305), ('param_22', -25.21514590817922), ('param_19', -26.3636657305936), ('param_21', -29.075142344188876), ('param_18', -29.68864911464537), ('param_12', -32.762321789575125), ('param_9', -33.70118877458078), ('param_0', -58.64537396540637), ('param_1', -77.7575006193548), ('param_2', -110.91127511593564), ('param_4', -121.0263824788873), ('param_3', -122.10003213042798), ('param_6', -128.7886275767614), ('param_5', -130.51199172877503), ('param_8', -152.37901192640794), ('param_7', -178.46467752363048)]

# Learning, eps = 0.081, 50K
#modelKNN1Info = [('param_26', -7.030951353873652), ('param_25', -8.121648903277269), ('param_23', -8.368872551725815), ('param_17', -9.23745065056816), ('param_24', -10.89167731632033), ('param_16', -16.00636043798336), ('param_14', -19.1399383737337), ('param_11', -23.938998981582728), ('param_20', -24.898146994531512), ('param_18', -25.947563691699973), ('param_15', -26.042222349958433), ('param_22', -27.115777708733642), ('param_13', -27.465498341193328), ('param_19', -29.40258658137639), ('param_10', -32.8483820720988), ('param_21', -39.79466237011869), ('param_9', -40.49385833155584), ('param_12', -44.194488287700565), ('param_0', -74.62492244294893), ('param_4', -143.7105800015034), ('param_1', -151.46451694606358), ('param_2', -156.07909901002483), ('param_6', -178.5757713321343), ('param_3', -194.77448622497158), ('param_5', -196.20764486293265), ('param_8', -205.8234049802723), ('param_7', -206.30650028514486)]

# Learning, eps = 0.3, 5K
#modelKNN1Info = [('param_25', -32.23615308913506), ('param_26', -35.89667910275274), ('param_17', -36.67286012192482), ('param_16', -43.74959390384386), ('param_14', -45.936522664751436), ('param_23', -49.56386601579181), ('param_13', -50.4825284896432), ('param_22', -53.415273478491535), ('param_24', -60.01024486863375), ('param_11', -61.45737369980077), ('param_15', -64.65688581604803), ('param_21', -71.44338070678359), ('param_10', -78.49594909182903), ('param_20', -80.47506083516234), ('param_9', -83.23908009748678), ('param_12', -83.44771786406537), ('param_18', -88.17945126527704), ('param_19', -89.0470097619032), ('param_0', -241.49803315680464), ('param_2', -293.9067675645362), ('param_3', -294.81378669727053), ('param_5', -354.3732474938825), ('param_1', -356.9820271404869), ('param_4', -368.380539881307), ('param_7', -372.2856720756736), ('param_8', -387.96524542981905), ('param_6', -402.83772183602053)]

# Learning, eps = 0.3, 50K
#modelKNN1Info = [('param_23', -7.175310791525495), ('param_25', -7.8320042334248985), ('param_26', -9.392312030772196), ('param_17', -11.996750877722002), ('param_18', -13.19087570649321), ('param_21', -13.621739045961856), ('param_24', -13.84682340572606), ('param_14', -14.486031076799646), ('param_16', -15.204716813878713), ('param_13', -16.52759334711725), ('param_11', -17.229811321295035), ('param_15', -19.143009584048407), ('param_20', -23.642208317430015), ('param_9', -24.048035529212328), ('param_22', -24.111134391832262), ('param_12', -26.513751902257304), ('param_19', -28.007337014733224), ('param_10', -35.306458650245084), ('param_0', -121.33473241969152), ('param_3', -138.6853920562479), ('param_7', -148.2370908538695), ('param_2', -148.65543149334607), ('param_5', -150.9463592675911), ('param_6', -174.86137148307444), ('param_8', -177.13002139383292), ('param_1', -177.28674112545954), ('param_4', -184.57772193519702)]

dataset = 0
realInfo = {'param_19': -1311.1666666666667, 'param_28': -1353.5333333333333, 'param_20': -1370.8666666666666, 'param_29': -1751.6666666666667, 'param_18': -1806.9, 'param_9': -1834.4333333333334, 'param_0': -1836.5333333333333, 'param_27': -1839.5666666666666, 'param_40': -2162.866666666667, 'param_41': -2170.9333333333334, 'param_39': -2179.0666666666666, 'param_30': -2206.1, 'param_12': -2207.8, 'param_21': -2211.4333333333334, 'param_3': -2211.6666666666665, 'param_50': -2211.7, 'param_31': -2222.2, 'param_48': -2222.5, 'param_49': -2224.266666666667, 'param_22': -2230.6, 'param_32': -2236.7, 'param_23': -2238.233333333333, 'param_24': -2240.6, 'param_43': -2240.766666666667, 'param_34': -2241.0333333333333, 'param_6': -2242.733333333333, 'param_53': -2243.5666666666666, 'param_42': -2243.8333333333335, 'param_52': -2244.133333333333, 'param_15': -2246.9666666666667, 'param_44': -2247.233333333333, 'param_25': -2249.3, 'param_33': -2251.766666666667, 'param_35': -2253.233333333333, 'param_26': -2253.5, 'param_51': -2257.133333333333, 'param_7': -2281.5333333333333, 'param_16': -2282.5666666666666, 'param_8': -2374.9, 'param_17': -2376.733333333333, 'param_4': -2524.8333333333335, 'param_13': -2525.266666666667, 'param_10': -2797.4666666666667, 'param_1': -2803.733333333333, 'param_5': -3195.0, 'param_14': -3196.8333333333335, 'param_38': -3255.1666666666665, 'param_2': -4122.766666666666, 'param_11': -4124.5, 'param_37': -4919.1, 'param_47': -4952.566666666667, 'param_36': -5000.333333333333, 'param_45': -5068.433333333333, 'param_46': -5163.5}
KNNoptimalInfo = {'param_45': -9.8, 'param_46': -15.6, 'param_37': -206.7, 'param_47': -267.8, 'param_36': -394.2, 'param_20': -440.1, 'param_29': -536.3, 'param_38': -590.1, 'param_27': -735.1, 'param_9': -759.2, 'param_0': -765.1, 'param_18': -786.6, 'param_28': -835.2, 'param_30': -859.2, 'param_3': -871.4, 'param_19': -871.9, 'param_34': -872.0, 'param_33': -875.1, 'param_22': -875.7, 'param_6': -876.9, 'param_12': -879.0, 'param_15': -879.5, 'param_24': -880.6, 'param_48': -882.6, 'param_21': -883.6, 'param_43': -883.7, 'param_52': -885.6, 'param_35': -886.6, 'param_53': -886.6, 'param_17': -887.9, 'param_25': -887.9, 'param_44': -888.2, 'param_49': -888.3, 'param_41': -889.0, 'param_8': -889.3, 'param_16': -889.6, 'param_50': -889.7, 'param_42': -890.0, 'param_51': -890.2, 'param_7': -891.1, 'param_31': -894.7, 'param_26': -896.0, 'param_40': -901.3, 'param_39': -907.6, 'param_23': -912.2, 'param_32': -914.6, 'param_13': -924.1, 'param_4': -937.0, 'param_14': -954.3, 'param_5': -964.5, 'param_10': -1122.4, 'param_1': -1135.1, 'param_11': -1223.8, 'param_2': -1234.9}
KNNrandomInfo = {'param_27': -3161.4, 'param_18': -3405.2, 'param_0': -3485.8, 'param_9': -3506.1, 'param_28': -3794.9, 'param_19': -4162.3, 'param_39': -5273.2, 'param_49': -5330.2, 'param_50': -5333.0, 'param_41': -5353.3, 'param_21': -5355.9, 'param_3': -5388.4, 'param_30': -5392.5, 'param_36': -5398.5, 'param_12': -5422.8, 'param_48': -5436.4, 'param_40': -5443.2, 'param_22': -5538.4, 'param_31': -5543.3, 'param_51': -5641.5, 'param_42': -5665.2, 'param_52': -5671.5, 'param_44': -5680.7, 'param_24': -5682.1, 'param_15': -5686.7, 'param_53': -5696.9, 'param_6': -5701.9, 'param_43': -5703.4, 'param_25': -5715.1, 'param_33': -5717.7, 'param_32': -5728.2, 'param_34': -5741.5, 'param_35': -5754.5, 'param_26': -5766.7, 'param_23': -5802.3, 'param_16': -5817.6, 'param_7': -5837.9, 'param_8': -6079.9, 'param_17': -6081.4, 'param_13': -6445.9, 'param_4': -6480.1, 'param_29': -6507.8, 'param_45': -6557.5, 'param_37': -6788.3, 'param_20': -6976.6, 'param_5': -7788.7, 'param_14': -7802.4, 'param_46': -8049.0, 'param_38': -8141.5, 'param_10': -8308.1, 'param_1': -8469.5, 'param_47': -9820.4, 'param_2': -10430.9, 'param_11': -10598.0}

#realkeys = [key for (key, value) in realInfo]
#realvalues = [value for (key, value) in realInfo]
#modelKNN1keys = [key for (key, value) in modelKNN1Info]
#modelKNN1values = [value for (key, value) in modelKNN1Info]
#modelKNN3keys = [key for (key, value) in modelKNN3Info]
#modelKNN3values = [value for (key, value) in modelKNN3Info]

realkeys = list(realInfo.keys())
realvalues = list(realInfo.values())
KNNoptimalInfokeys = list(KNNoptimalInfo.keys())
KNNoptimalInfovalues = list(KNNoptimalInfo.values())
KNNrandomInfokeys = list(KNNrandomInfo.keys())
KNNrandomInfovalues = list(KNNrandomInfo.values())

for i in range(len(realvalues)):
    realvalues[i] *= -1
for i in range(len(KNNoptimalInfovalues)):
    KNNoptimalInfovalues[i] *= -1
for i in range(len(KNNrandomInfovalues)):
    KNNrandomInfovalues[i] *= -1

rankingUnderReal = [i for i in range(len(realkeys))]
rankingUnderModelKNN1 = [realkeys.index(KNNoptimalInfokeys[i]) for i in range(len(KNNoptimalInfokeys))]
rankingUnderModelKNN3 = [realkeys.index(KNNrandomInfokeys[i]) for i in range(len(KNNrandomInfokeys))]

#correlation = np.corrcoef(rankingUnderReal, rankingUnderModelKNN1)
#print(correlation)


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
    plt.scatter([i+1 for i in range(len(realvalues))], np.array(realvalues)/50000, label='Real environment', s=5)
    modelKNN1valuesRankedByRealRanking = [modelKNN1values[rankingUnderModelKNN1.index(rankingUnderReal[i])] for i in range(len(rankingUnderReal))]
    plt.scatter([i+1 for i in range(len(modelKNN1values))], np.array(modelKNN1valuesRankedByRealRanking)/50000, label='Offline model', s=5, color=colors[1])
    #plt.scatter([12], np.array(modelKNN1valuesRankedByRealRanking[11]/250000), color=colors[2])
    plt.scatter([realkeys.index(modelKNN1keys[0])+1], np.array(modelKNN1valuesRankedByRealRanking[realkeys.index(modelKNN1keys[0])]/50000), facecolors='none', edgecolors=colors[1], s=160)
    #plt.scatter([i for i in range(len(modelKNN1values))], np.array(modelKNN1values)/50000, label='Average reward in the offline modelKNN1')
    plt.title("Learning policy, epsilon = 0.3,\n 50K timesteps of collected data, Correlation= 0.86", size=10)
    plt.xlabel('Hyperparameter ranking in the real environment', labelpad=35)
    plt.ylabel('Average reward\nof each\nhyperparameter setting\n(AUC)', rotation=0, labelpad=65)
    plt.rcParams['figure.figsize'] = [16, 8]
    #plt.arrow(15, -0.04, -3, 0.035, color='black', width=0.00005, length_includes_head=True, head_length=0.002, head_width=0.002)
    #plt.text(5, -0.05, 'Hyperparameters chosen\nby the offline modelKNN1', fontsize=8)
    plt.legend(loc=(0.01, 0.1), prop={'size': 8})
    plt.tight_layout()
    plt.savefig('../img/Final/Learning_Eps30_50K.png',dpi=300, bbox_inches='tight')
    #plt.show()


def plotPerformancesMultiple(realvalues, modelKNN1values, modelKNN3values, plt):
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    plt.scatter([i+1 for i in range(len(realvalues))], np.array(realvalues), label='Performance in\nthe real environment', s = 5)
    modelKNN1valuesRankedByRealRanking = [modelKNN1values[rankingUnderModelKNN1.index(rankingUnderReal[i])] for i in range(len(rankingUnderReal))]
    plt.scatter([i+1 for i in range(len(modelKNN1values))], np.array(modelKNN1valuesRankedByRealRanking), label='Performance in\nthe offline model (optimal)', s = 5, color=colors[1])
    #plt.scatter([realkeys.index(modelKNN1keys[0])+1], np.array(modelKNN1valuesRankedByRealRanking[realkeys.index(modelKNN1keys[0])]/250000), color=colors[1])
    #plt.scatter([realkeys.index(modelKNN1keys[0])+1], np.array(modelKNN1valuesRankedByRealRanking[realkeys.index(modelKNN1keys[0])]/250000), facecolors='none', edgecolors=colors[1], s=160)

    modelKNN3valuesRankedByRealRanking = [modelKNN3values[rankingUnderModelKNN3.index(rankingUnderReal[i])] for i in range(len(rankingUnderReal))]
    plt.scatter([i+1 for i in range(len(modelKNN3values))], np.array(modelKNN3valuesRankedByRealRanking), label='Performance in\nthe offline model (random)', s = 5, color=colors[2])
    #plt.scatter([12], np.array(modelKNN3valuesRankedByRealRanking[11]/250000), color=colors[2])
    #plt.scatter([realkeys.index(modelKNN3keys[0])+1], np.array(modelKNN3valuesRankedByRealRanking[realkeys.index(modelKNN1keys[0])]/250000), facecolors='none', edgecolors=colors[2], s=160)

    #plt.plot([i+1 for i in range(len(realvalues))], [-0.005 for i in range(len(realvalues))], '--', color='black', linewidth=0.75)

    #plt.scatter([i for i in range(len(modelKNN1values))], np.array(modelKNN1values)/50000, label='Average reward in the offline modelKNN1')
    plt.xlabel('Hyperparameter ranking in the real environment', labelpad=35)
    plt.ylabel('# Failures\nof each\nhyperparameter\nsetting\n(AUC)', rotation=0, labelpad=55)
    plt.rcParams['figure.figsize'] = [16, 8]
    #plt.arrow(15, -0.04, -3, 0.035, color='black', width=0.00005, length_includes_head=True, head_length=0.002, head_width=0.002)
    #plt.text(5, -0.05, 'Hyperparameters chosen\nby the offline modelKNN1', fontsize=8)
    #plt.legend(loc=(0.01, 0.1), prop={'size': 8})
    plt.legend()
    plt.tight_layout()
    plt.savefig('../img/modelHyperparamComparison.png',dpi=300, bbox_inches='tight')
    #plt.show()


#plotCorrelationPlot(rankingUnderReal, rankingUnderModelKNN1, plt)

#plotPerformances(realvalues, modelKNN1values, plt)

plotPerformancesMultiple(realvalues, KNNoptimalInfovalues, KNNrandomInfovalues, plt)
