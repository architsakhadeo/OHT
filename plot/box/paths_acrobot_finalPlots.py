ac_rnd = [34, 4, 43, 30, 24, 32, 40, 11, 20, 30, 3, 16, 53, 45, 0, 21, 43, 23, 44, 50, 9, 41, 37, 37, 11, 2, 26, 33, 18, 20]
basepath = "../../data/finalPlots/data/hyperparam_v5/"
basepath_new = "../../data/finalPlots/data/hyperparam_v5_newData/"
ac_true = [basepath + "acrobot/online_learning/esarsa/step15k/sweep/"]
ac_fqi_tc = [basepath + "acrobot/offline_learning/random_restarts/fqi-linear/fqi-adam/alpha_hidden_epsilon/step5k_env/optimalfixed_eps0/lambda1e-3/lockat_baseline_online/"]
ac_fqi_nn = [basepath + "acrobot/offline_learning/random_restarts/fqi/fqi-adam/alpha_hidden_epsilon/step5k_env/optimalfixed_eps0/earlystop/lambda1e-3/lockat_baseline_online/"]


# PLOT 1
ac_knnlaplace_optim_5k = [basepath + "acrobot/offline_learning/knn/learning/k3_laplace/timeout500/esarsa/step5k_env/data_optimal/drop0/sweep_rep1/"]
ac_knnraw_optim_5k = [basepath + "acrobot/offline_learning/knn/learning/k3/timeout500/esarsa/step5k_env/data_optimal/drop0/sweep/"]
ac_networkscaledlaplace_optim_5k = [basepath + "acrobot/offline_learning/network/learning/clip_scale_laplace_separated/timeout500/esarsa/step5k_env/data_optimal/sweep/"]
ac_networkscaledraw_optim_5k = [basepath + "acrobot/offline_learning/network/learning/clip_scale_separated/timeout500/esarsa/step5k_env/data_optimal/sweep/"]

# PLOT 2
ac_knnlaplace_avg_5k_new = [basepath_new + "acrobot/offline_learning/knn/learning/k3_laplace/timeout500/esarsa/step5k_env/data_suboptimal/drop0/sweep_rep1/"]
ac_knnlaplace_avg_2500_new = [basepath_new + "acrobot/offline_learning/knn/learning/k3_laplace/timeout500/esarsa/step2.5k_env/data_suboptimal/drop0/sweep_rep1/"]
ac_knnlaplace_avg_1k_new = [basepath_new + "acrobot/offline_learning/knn/learning/k3_laplace/timeout500/esarsa/step1k_env/data_suboptimal/drop0/sweep_rep1/"]
ac_knnlaplace_avg_500_new = [basepath_new + "acrobot/offline_learning/knn/learning/k3_laplace/timeout500/esarsa/step500_env/data_suboptimal/drop0/sweep_rep1/"]

ac_knnlaplace_optim_5k_new = [basepath_new + "acrobot/offline_learning/knn/learning/k3_laplace/timeout500/esarsa/step5k_env/data_optimal/drop0/sweep_rep1/"]
ac_knnlaplace_bad_5k_new = [basepath_new + "acrobot/offline_learning/knn/learning/k3_laplace/timeout500/esarsa/step5k_env/data_subsuboptimal/drop0/sweep_rep1/"]

# PLOT 3
acshift_true = [basepath + "acrobot/online_learning/esarsa/step15k/sweep/"]
acshift_knnlaplace_optim_5k_50kstep = [basepath + "acrobot_shift/online_learning/shift/esarsa/step50k/best/"]
acshift_knnlaplace_optim_5k = [basepath + "acrobot_shift/online_learning/shift/esarsa/step15k/best/"]
acshift_esarsa_true_trans = [basepath + "acrobot_shift/policy_transfer/shift/load_default/esarsa/best/"]
acshift_esarsa_calibration_trans = [basepath + "acrobot_shift/policy_transfer/shift/load_calibration_default/esarsa/fixed/"]
acshift_fqi_tc_optim_5k = [basepath + "acrobot_shift/policy_transfer/shift/load_default/fqi-linear/lambda1e-3/"]
acshift_fqi_nn_optim_5k = [basepath + "acrobot_shift/policy_transfer/shift/load_default/fqi/lambda1e-3/"]

# PLOT temp
ac_dqn = [basepath + "acrobot/online_learning/dqn/step600k/sweep/"]
ac_actorcritic = [basepath + "acrobot/online_learning/ac/step30k/sweep/"]
ac_dqn_knnlaplace_optim = [basepath + "acrobot/offline_learning/knn/learning/k3_laplace/timeout2k/dqn/step15k_env/data_optimal/drop0/sweep_rep1/"]
ac_actorcritic_knnlaplace_optim = [basepath + "acrobot/offline_learning/knn/learning/k3_laplace/timeout500/ac/step30k_env/data_optimal/drop0/sweep_rep1/"]
