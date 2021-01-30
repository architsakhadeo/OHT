ac_true_env = ["../../data/hyperparam_randomStart/acrobot/online_learning/esarsa/step50k/gridsearch_realenv/"]

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
    "../../data/hyperparam_randomStart/acrobot/offline_learning/fqi/fqi-adam/alpha_hidden_epsilon/step10k_env/data_eps0/lockat_baseline_online/"
]

ac_rnd = [2, 0, 29, 10, 28, 30, 0, 29, 14, 24, 9, 8, 29, 11, 28, 9, 7, 5, 6, 8, 9, 7, 9, 3, 21, 2, 8, 21, 6, 20, 2, 16]

"""
cartpole action noise 1%
"""
cpn1_true_env = ["../../data/hyperparam_v1/cartpole-noisy-action/noise_1perc/online_learning/esarsa-adam/step50k/sweep/"]

# compare
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

cpn1_rnd = [2, 0, 29, 10, 28, 30, 0, 29, 14, 24, 9, 8, 29, 11, 28, 9, 7, 5, 6, 8, 9, 7, 9, 3, 21, 2, 8, 21, 6, 20, 2, 16]