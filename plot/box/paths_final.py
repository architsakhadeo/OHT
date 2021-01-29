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
    "../../data/hyperparam_randomStart/cartpole-noisy-action/noise_1perc/offline_learning/fqi/fqi-adam/alpha_hidden_epsilon/step10k_env/data_eps0/lockat_baseline_online/"
]

ac_rnd = [19, 4, 3, 4, 31, 30, 8, 3, 27, 21, 13, 10, 16, 0, 15, 33, 30, 22, 5, 19, 25, 19, 32, 9, 1, 5, 5, 32, 15, 16, 24, 32, 21, 17, 18, 16]


"""
cartpole action noise 1%
"""
cpn1_true_env = ["../../data/hyperparam_randomStart/cartpole-noisy-action/noise_1perc/online_learning/esarsa-adam/step50k/sweep/"]

# cpn1_cm = [
#     "../../data/hyperparam_randomStart/cartpole-noisy-action/noise_1perc/offline_learning/knn-ens/farStart/farTrans/k3/timeout200/esarsa/step10k_env/data_eps0/drop0.2/ensembleseed1/",
#     "../../data/hyperparam_randomStart/cartpole-noisy-action/noise_1perc/offline_learning/knn-ens/farStart/farTrans/k3/timeout200/esarsa/step10k_env/data_eps0/drop0.2/ensembleseed2/",
#     "../../data/hyperparam_randomStart/cartpole-noisy-action/noise_1perc/offline_learning/knn-ens/farStart/farTrans/k3/timeout200/esarsa/step10k_env/data_eps0/drop0.2/ensembleseed3/",
#     "../../data/hyperparam_randomStart/cartpole-noisy-action/noise_1perc/offline_learning/knn-ens/farStart/farTrans/k3/timeout200/esarsa/step10k_env/data_eps0/drop0.2/ensembleseed4/",
#     "../../data/hyperparam_randomStart/cartpole-noisy-action/noise_1perc/offline_learning/knn-ens/farStart/farTrans/k3/timeout200/esarsa/step10k_env/data_eps0/drop0.2/ensembleseed5/",
# ]
# Best
ns1_timeout200_farStart = [
    "../../data/hyperparam_randomStart/cartpole-noisy-action/noise_1perc/offline_learning/knn-ens/farStart/farTrans/k3/timeout200/esarsa/step10k_env/data_eps0/drop0.2/ensembleseed1/",
    "../../data/hyperparam_randomStart/cartpole-noisy-action/noise_1perc/offline_learning/knn-ens/farStart/farTrans/k3/timeout200/esarsa/step10k_env/data_eps0/drop0.2/ensembleseed2/",
    "../../data/hyperparam_randomStart/cartpole-noisy-action/noise_1perc/offline_learning/knn-ens/farStart/farTrans/k3/timeout200/esarsa/step10k_env/data_eps0/drop0.2/ensembleseed3/",
    "../../data/hyperparam_randomStart/cartpole-noisy-action/noise_1perc/offline_learning/knn-ens/farStart/farTrans/k3/timeout200/esarsa/step10k_env/data_eps0/drop0.2/ensembleseed4/",
    "../../data/hyperparam_randomStart/cartpole-noisy-action/noise_1perc/offline_learning/knn-ens/farStart/farTrans/k3/timeout200/esarsa/step10k_env/data_eps0/drop0.2/ensembleseed5/",
]

# compare
ns1_timeout200 = [
    "../..//data/hyperparam_randomStart/cartpole-noisy-action/noise_1perc/offline_learning/knn-ens/randomInit/farTrans/k3/timeout200/esarsa/step10k_env/data_eps0/drop0.2/ensembleseed1/",
    "../..//data/hyperparam_randomStart/cartpole-noisy-action/noise_1perc/offline_learning/knn-ens/randomInit/farTrans/k3/timeout200/esarsa/step10k_env/data_eps0/drop0.2/ensembleseed2/",
    "../..//data/hyperparam_randomStart/cartpole-noisy-action/noise_1perc/offline_learning/knn-ens/randomInit/farTrans/k3/timeout200/esarsa/step10k_env/data_eps0/drop0.2/ensembleseed3/",
    # "../..//data/hyperparam_randomStart/cartpole-noisy-action/noise_1perc/offline_learning/knn-ens/randomInit/farTrans/k3/timeout200/esarsa/step10k_env/data_eps0/drop0.2/ensembleseed4/",
    "../..//data/hyperparam_randomStart/cartpole-noisy-action/noise_1perc/offline_learning/knn-ens/randomInit/farTrans/k3/timeout200/esarsa/step10k_env/data_eps0/drop0.2/ensembleseed5/",
]

ns1_timeout1000_farStart = [
    "../../data/hyperparam_randomStart/cartpole-noisy-action/noise_1perc/offline_learning/knn-ens/farStart/farTrans/k3/timeout1000/esarsa/step10k_env/data_eps0/drop0.2/ensembleseed1/",
    "../../data/hyperparam_randomStart/cartpole-noisy-action/noise_1perc/offline_learning/knn-ens/farStart/farTrans/k3/timeout1000/esarsa/step10k_env/data_eps0/drop0.2/ensembleseed2/",
    "../../data/hyperparam_randomStart/cartpole-noisy-action/noise_1perc/offline_learning/knn-ens/farStart/farTrans/k3/timeout1000/esarsa/step10k_env/data_eps0/drop0.2/ensembleseed3/",
    "../../data/hyperparam_randomStart/cartpole-noisy-action/noise_1perc/offline_learning/knn-ens/farStart/farTrans/k3/timeout1000/esarsa/step10k_env/data_eps0/drop0.2/ensembleseed4/",
    "../../data/hyperparam_randomStart/cartpole-noisy-action/noise_1perc/offline_learning/knn-ens/farStart/farTrans/k3/timeout1000/esarsa/step10k_env/data_eps0/drop0.2/ensembleseed5/",
]

cpn1_fqi = [
    "../../data/hyperparam_randomStart/cartpole-noisy-action/noise_1perc/offline_learning/fqi/fqi-adam/alpha_hidden_epsilon/step10k_env/data_eps0/lockat_baseline_online/"
]

cpn1_rnd = [18, 12, 31, 20, 3, 27, 20, 26, 22, 29, 16, 30, 35, 33, 23, 21, 14, 5, 0, 19, 21, 23, 32, 5, 17, 29, 21, 2, 15, 28, 23, 35, 25, 20, 14, 20]
