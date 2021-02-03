ac_true_env = ["../../../../../../Downloads/data_timeoutsAndtransitions_acrobot/acrobot/online_learning/esarsa/step50k/gridsearch_realenv/"]

# ac_cm = [
#     "../../data/hyperparam_randomStart/acrobot/offline_learning/knn-ensemble/k3/esarsa/step10k/drop0.2/ensembleseed1/optimalfixed_eps0/",
#     "../../data/hyperparam_randomStart/acrobot/offline_learning/knn-ensemble/k3/esarsa/step10k/drop0.2/ensembleseed2/optimalfixed_eps0/",
#     "../../data/hyperparam_randomStart/acrobot/offline_learning/knn-ensemble/k3/esarsa/step10k/drop0.2/ensembleseed3/optimalfixed_eps0/",
#     "../../data/hyperparam_randomStart/acrobot/offline_learning/knn-ensemble/k3/esarsa/step10k/drop0.2/ensembleseed4/optimalfixed_eps0/",
#     "../../data/hyperparam_randomStart/acrobot/offline_learning/knn-ensemble/k3/esarsa/step10k/drop0.2/ensembleseed5/optimalfixed_eps0/",
# ]

ac_cm = [
    "../../data/hyperparam_randomStart/acrobot/offline_learning/knn-ensemble/k3_50k_timeout200_randinit/esarsa/step10k/drop0.2/ensembleseed1/optimalfixed_eps0/",
    "../../data/hyperparam_randomStart/acrobot/offline_learning/knn-ensemble/k3_50k_timeout200_randinit/esarsa/step10k/drop0.2/ensembleseed2/optimalfixed_eps0/",
    "../../data/hyperparam_randomStart/acrobot/offline_learning/knn-ensemble/k3_50k_timeout200_randinit/esarsa/step10k/drop0.2/ensembleseed3/optimalfixed_eps0/",
    "../../data/hyperparam_randomStart/acrobot/offline_learning/knn-ensemble/k3_50k_timeout200_randinit/esarsa/step10k/drop0.2/ensembleseed4/optimalfixed_eps0/",
    "../../data/hyperparam_randomStart/acrobot/offline_learning/knn-ensemble/k3_50k_timeout200_randinit/esarsa/step10k/drop0.2/ensembleseed5/optimalfixed_eps0/",
]

ac_fqi = [
    "../../../../../../Downloads/data_timeoutsAndtransitions_acrobot/acrobot/offline_learning/fqi/fqi-adam/alpha_hidden_epsilon/step10k_env/data_eps0/lockat_baseline_online/"
]

ac_rnd = [2, 0, 29, 10, 28, 30, 0, 29, 14, 24, 9, 8, 29, 11, 28, 9, 7, 5, 6, 8, 9, 7, 9, 3, 21, 2, 8, 21, 6, 20, 2, 16]

"""
cartpole action noise 1%
"""
cpn1_true_env = ["../../data/hyperparam_v1/cartpole-noisy-action/noise_1perc/online_learning/esarsa-adam/step50k/sweep/"]

# Without random start
distStart_farTrans_time200 = [
    "../../data/hyperparam_v1/cartpole-noisy-action/noise_1perc/offline_learning/knn-ens/farStart/farTrans/k3/timeout200/esarsa/step10k_env/data_eps0/drop0.2/ensembleseed1",
    "../../data/hyperparam_v1/cartpole-noisy-action/noise_1perc/offline_learning/knn-ens/farStart/farTrans/k3/timeout200/esarsa/step10k_env/data_eps0/drop0.2/ensembleseed2",
    "../../data/hyperparam_v1/cartpole-noisy-action/noise_1perc/offline_learning/knn-ens/farStart/farTrans/k3/timeout200/esarsa/step10k_env/data_eps0/drop0.2/ensembleseed3",
    "../../data/hyperparam_v1/cartpole-noisy-action/noise_1perc/offline_learning/knn-ens/farStart/farTrans/k3/timeout200/esarsa/step10k_env/data_eps0/drop0.2/ensembleseed4",
    "../../data/hyperparam_v1/cartpole-noisy-action/noise_1perc/offline_learning/knn-ens/farStart/farTrans/k3/timeout200/esarsa/step10k_env/data_eps0/drop0.2/ensembleseed5",
]
distStart_closeTrans_time200 = [
    "../../data/hyperparam_v1/cartpole-noisy-action/noise_1perc/offline_learning/knn-ens/farStart/closeTrans/k3/timeout200/esarsa/step10k_env/data_eps0/drop0.2/ensembleseed1",
    "../../data/hyperparam_v1/cartpole-noisy-action/noise_1perc/offline_learning/knn-ens/farStart/closeTrans/k3/timeout200/esarsa/step10k_env/data_eps0/drop0.2/ensembleseed2",
    "../../data/hyperparam_v1/cartpole-noisy-action/noise_1perc/offline_learning/knn-ens/farStart/closeTrans/k3/timeout200/esarsa/step10k_env/data_eps0/drop0.2/ensembleseed3",
    "../../data/hyperparam_v1/cartpole-noisy-action/noise_1perc/offline_learning/knn-ens/farStart/closeTrans/k3/timeout200/esarsa/step10k_env/data_eps0/drop0.2/ensembleseed4",
    "../../data/hyperparam_v1/cartpole-noisy-action/noise_1perc/offline_learning/knn-ens/farStart/closeTrans/k3/timeout200/esarsa/step10k_env/data_eps0/drop0.2/ensembleseed5",
]
trueStart_farTrans_time1000 = [
     "../../data/hyperparam_v1/cartpole-noisy-action/noise_1perc/offline_learning/knn-ens/randomInit/farTrans/k3/timeout1000/esarsa/step10k_env/data_eps0/drop0.2/ensembleseed1",
     "../../data/hyperparam_v1/cartpole-noisy-action/noise_1perc/offline_learning/knn-ens/randomInit/farTrans/k3/timeout1000/esarsa/step10k_env/data_eps0/drop0.2/ensembleseed2",
     "../../data/hyperparam_v1/cartpole-noisy-action/noise_1perc/offline_learning/knn-ens/randomInit/farTrans/k3/timeout1000/esarsa/step10k_env/data_eps0/drop0.2/ensembleseed3",
     "../../data/hyperparam_v1/cartpole-noisy-action/noise_1perc/offline_learning/knn-ens/randomInit/farTrans/k3/timeout1000/esarsa/step10k_env/data_eps0/drop0.2/ensembleseed4",
     "../../data/hyperparam_v1/cartpole-noisy-action/noise_1perc/offline_learning/knn-ens/randomInit/farTrans/k3/timeout1000/esarsa/step10k_env/data_eps0/drop0.2/ensembleseed5",
]
cpn1_fqi = [
    "../../data/hyperparam_v1/cartpole-noisy-action/noise_1perc/offline_learning/fqi/fqi-adam/alpha_hidden_epsilon/step10k_env/data_eps0/lockat_baseline_online/"
]

# With random start
RS_distStart_closeTrans_time200 = [
    "../../data/hyperparam_rs_v1/cartpole-noisy-action/noise_1perc/offline_learning/knn-ens/farStart/closeTrans/k3/timeout200/esarsa/step10k_env/data_eps0/drop0.2/ensembleseed1",
    "../../data/hyperparam_rs_v1/cartpole-noisy-action/noise_1perc/offline_learning/knn-ens/farStart/closeTrans/k3/timeout200/esarsa/step10k_env/data_eps0/drop0.2/ensembleseed2",
    "../../data/hyperparam_rs_v1/cartpole-noisy-action/noise_1perc/offline_learning/knn-ens/farStart/closeTrans/k3/timeout200/esarsa/step10k_env/data_eps0/drop0.2/ensembleseed3",
    "../../data/hyperparam_rs_v1/cartpole-noisy-action/noise_1perc/offline_learning/knn-ens/farStart/closeTrans/k3/timeout200/esarsa/step10k_env/data_eps0/drop0.2/ensembleseed4",
    "../../data/hyperparam_rs_v1/cartpole-noisy-action/noise_1perc/offline_learning/knn-ens/farStart/closeTrans/k3/timeout200/esarsa/step10k_env/data_eps0/drop0.2/ensembleseed5",
]
RS_trueStart_farTrans_time1000 = [
    "../../data/hyperparam_rs_v1/cartpole-noisy-action/noise_1perc/offline_learning/knn-ens/randomInit/farTrans/k3/timeout1000/esarsa/step10k_env/data_eps0/drop0.2/ensembleseed1",
    "../../data/hyperparam_rs_v1/cartpole-noisy-action/noise_1perc/offline_learning/knn-ens/randomInit/farTrans/k3/timeout1000/esarsa/step10k_env/data_eps0/drop0.2/ensembleseed2",
    "../../data/hyperparam_rs_v1/cartpole-noisy-action/noise_1perc/offline_learning/knn-ens/randomInit/farTrans/k3/timeout1000/esarsa/step10k_env/data_eps0/drop0.2/ensembleseed3",
    "../../data/hyperparam_rs_v1/cartpole-noisy-action/noise_1perc/offline_learning/knn-ens/randomInit/farTrans/k3/timeout1000/esarsa/step10k_env/data_eps0/drop0.2/ensembleseed4",
    "../../data/hyperparam_rs_v1/cartpole-noisy-action/noise_1perc/offline_learning/knn-ens/randomInit/farTrans/k3/timeout1000/esarsa/step10k_env/data_eps0/drop0.2/ensembleseed5",
]
RS_cpn1_fqi = [
    "../../data/hyperparam_randomStart/cartpole-noisy-action/noise_1perc/offline_learning/fqi/fqi-adam/alpha_hidden_epsilon/step10k_env/data_eps0/lockat_baseline_online/"
]

cpn1_rnd = [2, 0, 29, 10, 28, 30, 0, 29, 14, 24, 9, 8, 29, 11, 28, 9, 7, 5, 6, 8, 9, 7, 9, 3, 21, 2, 8, 21, 6, 20, 2, 16]




AcrobotdistantStart_regularTrans_timeout200 = [
    "../../../../../../Downloads/data_timeoutsAndtransitions_acrobot/acrobot/randomrestarts/offline_learning/knn-ensemble/k3_50k_timeout200_distantStart_regularTrans/esarsa/step10k/drop0.2/ensembleseed1/optimalfixed_eps0/",
    "../../../../../../Downloads/data_timeoutsAndtransitions_acrobot/acrobot/randomrestarts/offline_learning/knn-ensemble/k3_50k_timeout200_distantStart_regularTrans/esarsa/step10k/drop0.2/ensembleseed2/optimalfixed_eps0/",
    "../../../../../../Downloads/data_timeoutsAndtransitions_acrobot/acrobot/randomrestarts/offline_learning/knn-ensemble/k3_50k_timeout200_distantStart_regularTrans/esarsa/step10k/drop0.2/ensembleseed3/optimalfixed_eps0/",
    "../../../../../../Downloads/data_timeoutsAndtransitions_acrobot/acrobot/randomrestarts/offline_learning/knn-ensemble/k3_50k_timeout200_distantStart_regularTrans/esarsa/step10k/drop0.2/ensembleseed4/optimalfixed_eps0/",
    "../../../../../../Downloads/data_timeoutsAndtransitions_acrobot/acrobot/randomrestarts/offline_learning/knn-ensemble/k3_50k_timeout200_distantStart_regularTrans/esarsa/step10k/drop0.2/ensembleseed5/optimalfixed_eps0/"
]

AcrobottrueStart_adversarialTrans_timeout1000 = [
    "../../../../../../Downloads/data_timeoutsAndtransitions_acrobot/acrobot/randomrestarts/offline_learning/knn-ensemble/k3_50k_timeout1000_trueStart_adversarialTrans/esarsa/step10k/drop0.2/ensembleseed1/optimalfixed_eps0/",
    "../../../../../../Downloads/data_timeoutsAndtransitions_acrobot/acrobot/randomrestarts/offline_learning/knn-ensemble/k3_50k_timeout1000_trueStart_adversarialTrans/esarsa/step10k/drop0.2/ensembleseed2/optimalfixed_eps0/",
    "../../../../../../Downloads/data_timeoutsAndtransitions_acrobot/acrobot/randomrestarts/offline_learning/knn-ensemble/k3_50k_timeout1000_trueStart_adversarialTrans/esarsa/step10k/drop0.2/ensembleseed3/optimalfixed_eps0/",
    "../../../../../../Downloads/data_timeoutsAndtransitions_acrobot/acrobot/randomrestarts/offline_learning/knn-ensemble/k3_50k_timeout1000_trueStart_adversarialTrans/esarsa/step10k/drop0.2/ensembleseed4/optimalfixed_eps0/",
    "../../../../../../Downloads/data_timeoutsAndtransitions_acrobot/acrobot/randomrestarts/offline_learning/knn-ensemble/k3_50k_timeout1000_trueStart_adversarialTrans/esarsa/step10k/drop0.2/ensembleseed5/optimalfixed_eps0/",
]


k1_notimeout = ["../../../../../../Downloads/data_timeoutsAndtransitions_acrobot/acrobot/offline_learning/knn-ensemble/k1_notimeout/esarsa/step10k/optimalfixed_eps0/"]
k1_timeout1000 = ["../../../../../../Downloads/data_timeoutsAndtransitions_acrobot/acrobot/offline_learning/knn-ensemble/k1_timeout1000/esarsa/step10k/optimalfixed_eps0/"]
k3ensemble_notimeout = [
    "../../../../../../Downloads/data_timeoutsAndtransitions_acrobot/acrobot/offline_learning/knn-ensemble/k3ensemble_notimeout/esarsa/step10k/drop0.2/ensembleseed1/optimalfixed_eps0/",
    "../../../../../../Downloads/data_timeoutsAndtransitions_acrobot/acrobot/offline_learning/knn-ensemble/k3ensemble_notimeout/esarsa/step10k/drop0.2/ensembleseed2/optimalfixed_eps0/",
    "../../../../../../Downloads/data_timeoutsAndtransitions_acrobot/acrobot/offline_learning/knn-ensemble/k3ensemble_notimeout/esarsa/step10k/drop0.2/ensembleseed3/optimalfixed_eps0/",
    "../../../../../../Downloads/data_timeoutsAndtransitions_acrobot/acrobot/offline_learning/knn-ensemble/k3ensemble_notimeout/esarsa/step10k/drop0.2/ensembleseed4/optimalfixed_eps0/",
    "../../../../../../Downloads/data_timeoutsAndtransitions_acrobot/acrobot/offline_learning/knn-ensemble/k3ensemble_notimeout/esarsa/step10k/drop0.2/ensembleseed5/optimalfixed_eps0/"
]
k3ensemble_timeout1000 = [
    "../../../../../../Downloads/data_timeoutsAndtransitions_acrobot/acrobot/offline_learning/knn-ensemble/k3ensemble_timeout1000/esarsa/step10k/drop0.2/ensembleseed1/optimalfixed_eps0/",
    "../../../../../../Downloads/data_timeoutsAndtransitions_acrobot/acrobot/offline_learning/knn-ensemble/k3ensemble_timeout1000/esarsa/step10k/drop0.2/ensembleseed2/optimalfixed_eps0/",
    "../../../../../../Downloads/data_timeoutsAndtransitions_acrobot/acrobot/offline_learning/knn-ensemble/k3ensemble_timeout1000/esarsa/step10k/drop0.2/ensembleseed3/optimalfixed_eps0/",
    "../../../../../../Downloads/data_timeoutsAndtransitions_acrobot/acrobot/offline_learning/knn-ensemble/k3ensemble_timeout1000/esarsa/step10k/drop0.2/ensembleseed4/optimalfixed_eps0/",
    "../../../../../../Downloads/data_timeoutsAndtransitions_acrobot/acrobot/offline_learning/knn-ensemble/k3ensemble_timeout1000/esarsa/step10k/drop0.2/ensembleseed5/optimalfixed_eps0/"
    ]
k3ensemble_adversarial_notimeout = [
    "../../../../../../Downloads/data_timeoutsAndtransitions_acrobot/acrobot/offline_learning/knn-ensemble/k3ensemble_adversarial_notimeout/esarsa/step10k/drop0.2/ensembleseed1/optimalfixed_eps0/",
    "../../../../../../Downloads/data_timeoutsAndtransitions_acrobot/acrobot/offline_learning/knn-ensemble/k3ensemble_adversarial_notimeout/esarsa/step10k/drop0.2/ensembleseed2/optimalfixed_eps0/",
    "../../../../../../Downloads/data_timeoutsAndtransitions_acrobot/acrobot/offline_learning/knn-ensemble/k3ensemble_adversarial_notimeout/esarsa/step10k/drop0.2/ensembleseed3/optimalfixed_eps0/",
    "../../../../../../Downloads/data_timeoutsAndtransitions_acrobot/acrobot/offline_learning/knn-ensemble/k3ensemble_adversarial_notimeout/esarsa/step10k/drop0.2/ensembleseed4/optimalfixed_eps0/",
    "../../../../../../Downloads/data_timeoutsAndtransitions_acrobot/acrobot/offline_learning/knn-ensemble/k3ensemble_adversarial_notimeout/esarsa/step10k/drop0.2/ensembleseed5/optimalfixed_eps0/"
]
k3ensemble_adverarial_timeout1000 = [
    "../../../../../../Downloads/data_timeoutsAndtransitions_acrobot/acrobot/offline_learning/knn-ensemble/k3_50k_timeout1000_trueStart_adversarialTrans/esarsa/step10k/drop0.2/ensembleseed1/optimalfixed_eps0/",
    "../../../../../../Downloads/data_timeoutsAndtransitions_acrobot/acrobot/offline_learning/knn-ensemble/k3_50k_timeout1000_trueStart_adversarialTrans/esarsa/step10k/drop0.2/ensembleseed2/optimalfixed_eps0/",
    "../../../../../../Downloads/data_timeoutsAndtransitions_acrobot/acrobot/offline_learning/knn-ensemble/k3_50k_timeout1000_trueStart_adversarialTrans/esarsa/step10k/drop0.2/ensembleseed3/optimalfixed_eps0/",
    "../../../../../../Downloads/data_timeoutsAndtransitions_acrobot/acrobot/offline_learning/knn-ensemble/k3_50k_timeout1000_trueStart_adversarialTrans/esarsa/step10k/drop0.2/ensembleseed4/optimalfixed_eps0/",
    "../../../../../../Downloads/data_timeoutsAndtransitions_acrobot/acrobot/offline_learning/knn-ensemble/k3_50k_timeout1000_trueStart_adversarialTrans/esarsa/step10k/drop0.2/ensembleseed5/optimalfixed_eps0/"
]
k3_adversarial_timeout1000_subruns = [
    "../../../../../../Downloads/data_timeoutsAndtransitions_acrobot/acrobot/offline_learning/knn-ensemble/k3_adversarial_timeout1000_subruns/esarsa/step10k/optimalfixed_eps0/"
]
ac_CEM = [
    "../../../../../../Downloads/data_timeoutsAndtransitions_acrobot/list/acrobot/online_learning/esarsa/step50k/list_realenv/"
]



"""
Ablation study - Cartpole
"""
trueStart_farTrans_time0 = [
    "../../data/hyperparam_v1/cartpole-noisy-action/noise_1perc/offline_learning/knn-ens/randomInit/farTrans/k3/timeout0/esarsa/step10k_env/data_eps0/drop0.2/ensembleseed1",
    "../../data/hyperparam_v1/cartpole-noisy-action/noise_1perc/offline_learning/knn-ens/randomInit/farTrans/k3/timeout0/esarsa/step10k_env/data_eps0/drop0.2/ensembleseed2",
    "../../data/hyperparam_v1/cartpole-noisy-action/noise_1perc/offline_learning/knn-ens/randomInit/farTrans/k3/timeout0/esarsa/step10k_env/data_eps0/drop0.2/ensembleseed3",
    "../../data/hyperparam_v1/cartpole-noisy-action/noise_1perc/offline_learning/knn-ens/randomInit/farTrans/k3/timeout0/esarsa/step10k_env/data_eps0/drop0.2/ensembleseed4",
    "../../data/hyperparam_v1/cartpole-noisy-action/noise_1perc/offline_learning/knn-ens/randomInit/farTrans/k3/timeout0/esarsa/step10k_env/data_eps0/drop0.2/ensembleseed5",
]
trueStart_closeTrans_time1000 = [
    "../../data/hyperparam_v1/cartpole-noisy-action/noise_1perc/offline_learning/knn-ens/randomInit/closeTrans/k3/timeout1000/esarsa/step10k_env/data_eps0/drop0.2/ensembleseed1",
    "../../data/hyperparam_v1/cartpole-noisy-action/noise_1perc/offline_learning/knn-ens/randomInit/closeTrans/k3/timeout1000/esarsa/step10k_env/data_eps0/drop0.2/ensembleseed2",
    "../../data/hyperparam_v1/cartpole-noisy-action/noise_1perc/offline_learning/knn-ens/randomInit/closeTrans/k3/timeout1000/esarsa/step10k_env/data_eps0/drop0.2/ensembleseed3",
    "../../data/hyperparam_v1/cartpole-noisy-action/noise_1perc/offline_learning/knn-ens/randomInit/closeTrans/k3/timeout1000/esarsa/step10k_env/data_eps0/drop0.2/ensembleseed4",
    "../../data/hyperparam_v1/cartpole-noisy-action/noise_1perc/offline_learning/knn-ens/randomInit/closeTrans/k3/timeout1000/esarsa/step10k_env/data_eps0/drop0.2/ensembleseed5",
]
trueStart_closeTrans_time0 = [
    "../../data/hyperparam_v1/cartpole-noisy-action/noise_1perc/offline_learning/knn-ens/randomInit/closeTrans/k3/timeout0/esarsa/step10k_env/data_eps0/drop0.2/ensembleseed1",
    "../../data/hyperparam_v1/cartpole-noisy-action/noise_1perc/offline_learning/knn-ens/randomInit/closeTrans/k3/timeout0/esarsa/step10k_env/data_eps0/drop0.2/ensembleseed2",
    "../../data/hyperparam_v1/cartpole-noisy-action/noise_1perc/offline_learning/knn-ens/randomInit/closeTrans/k3/timeout0/esarsa/step10k_env/data_eps0/drop0.2/ensembleseed3",
    "../../data/hyperparam_v1/cartpole-noisy-action/noise_1perc/offline_learning/knn-ens/randomInit/closeTrans/k3/timeout0/esarsa/step10k_env/data_eps0/drop0.2/ensembleseed4",
    "../../data/hyperparam_v1/cartpole-noisy-action/noise_1perc/offline_learning/knn-ens/randomInit/closeTrans/k3/timeout0/esarsa/step10k_env/data_eps0/drop0.2/ensembleseed5",
]
trueStart_noEnsemble_time1000 = [
    "../../data/hyperparam_v1/cartpole-noisy-action/noise_1perc/offline_learning/knn-ens/randomInit/closeTrans/k1/timeout1000/esarsa/step10k_env/data_eps0/drop0/ensembleseed0",
]
trueStart_noEnsemble_time0 = [
    "../../data/hyperparam_v1/cartpole-noisy-action/noise_1perc/offline_learning/knn-ens/randomInit/closeTrans/k1/timeout0/esarsa/step10k_env/data_eps0/drop0/ensembleseed0",
]

"""
Dataset size study - Cartpole
"""
trueStart_farTrans_time1000_5k = [
    "../../data/hyperparam_v1/cartpole-noisy-action/noise_1perc/offline_learning/knn-ens/randomInit/farTrans/k3/timeout1000/esarsa/step5k_env/data_eps0/drop0.2/ensembleseed1",
    "../../data/hyperparam_v1/cartpole-noisy-action/noise_1perc/offline_learning/knn-ens/randomInit/farTrans/k3/timeout1000/esarsa/step5k_env/data_eps0/drop0.2/ensembleseed2",
    "../../data/hyperparam_v1/cartpole-noisy-action/noise_1perc/offline_learning/knn-ens/randomInit/farTrans/k3/timeout1000/esarsa/step5k_env/data_eps0/drop0.2/ensembleseed3",
    "../../data/hyperparam_v1/cartpole-noisy-action/noise_1perc/offline_learning/knn-ens/randomInit/farTrans/k3/timeout1000/esarsa/step5k_env/data_eps0/drop0.2/ensembleseed4",
    "../../data/hyperparam_v1/cartpole-noisy-action/noise_1perc/offline_learning/knn-ens/randomInit/farTrans/k3/timeout1000/esarsa/step5k_env/data_eps0/drop0.2/ensembleseed5",
]
trueStart_farTrans_time1000_2k = [
    "../../data/hyperparam_v1/cartpole-noisy-action/noise_1perc/offline_learning/knn-ens/randomInit/farTrans/k3/timeout1000/esarsa/step2k_env/data_eps0/drop0.2/ensembleseed1",
    "../../data/hyperparam_v1/cartpole-noisy-action/noise_1perc/offline_learning/knn-ens/randomInit/farTrans/k3/timeout1000/esarsa/step2k_env/data_eps0/drop0.2/ensembleseed2",
    "../../data/hyperparam_v1/cartpole-noisy-action/noise_1perc/offline_learning/knn-ens/randomInit/farTrans/k3/timeout1000/esarsa/step2k_env/data_eps0/drop0.2/ensembleseed3",
    "../../data/hyperparam_v1/cartpole-noisy-action/noise_1perc/offline_learning/knn-ens/randomInit/farTrans/k3/timeout1000/esarsa/step2k_env/data_eps0/drop0.2/ensembleseed4",
    "../../data/hyperparam_v1/cartpole-noisy-action/noise_1perc/offline_learning/knn-ens/randomInit/farTrans/k3/timeout1000/esarsa/step2k_env/data_eps0/drop0.2/ensembleseed5",
]
trueStart_farTrans_time1000_1k = [
    "../../data/hyperparam_v1/cartpole-noisy-action/noise_1perc/offline_learning/knn-ens/randomInit/farTrans/k3/timeout1000/esarsa/step1k_env/data_eps0/drop0.2/ensembleseed1",
    "../../data/hyperparam_v1/cartpole-noisy-action/noise_1perc/offline_learning/knn-ens/randomInit/farTrans/k3/timeout1000/esarsa/step1k_env/data_eps0/drop0.2/ensembleseed2",
    "../../data/hyperparam_v1/cartpole-noisy-action/noise_1perc/offline_learning/knn-ens/randomInit/farTrans/k3/timeout1000/esarsa/step1k_env/data_eps0/drop0.2/ensembleseed3",
    "../../data/hyperparam_v1/cartpole-noisy-action/noise_1perc/offline_learning/knn-ens/randomInit/farTrans/k3/timeout1000/esarsa/step1k_env/data_eps0/drop0.2/ensembleseed4",
    "../../data/hyperparam_v1/cartpole-noisy-action/noise_1perc/offline_learning/knn-ens/randomInit/farTrans/k3/timeout1000/esarsa/step1k_env/data_eps0/drop0.2/ensembleseed5",
]

"""
hyperparam v2: average calibration model performance with runs in inner loop
"""
v2_cpn1_true_env = ["../../data/hyperparam_v2/cartpole-noisy-action/noise_1perc/online_learning/esarsa-adam/step50k/sweep/"]

v2_trueStart_farTrans_time1000 = [
    "../../data/hyperparam_v2/cartpole-noisy-action/noise_1perc/offline_learning/knn-ens/randomInit/farTrans/k3/timeout1000/esarsa/step10k_env/data_eps0/drop0/ensembleseed0",
]
v2_trueStart_closeTrans_time1000 = [
    "../../data/hyperparam_v2/cartpole-noisy-action/noise_1perc/offline_learning/knn-ens/randomInit/closeTrans/k3/timeout1000/esarsa/step10k_env/data_eps0/drop0/ensembleseed0",
]