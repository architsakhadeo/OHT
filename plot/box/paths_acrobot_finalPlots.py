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
ac_knnlaplace_optim_5k_new = [basepath_new + "acrobot/offline_learning/knn/learning/k3_laplace/timeout500/esarsa/step5k_env/data_optimal/drop0/sweep_rep1/"]
ac_knnlaplace_optim_2500_new = [basepath_new + "acrobot/offline_learning/knn/learning/k3_laplace/timeout500/esarsa/step2.5k_env/data_optimal/drop0/sweep_rep1/"]
ac_knnlaplace_optim_1k_new = [basepath_new + "acrobot/offline_learning/knn/learning/k3_laplace/timeout500/esarsa/step1k_env/data_optimal/drop0/sweep_rep1/"]
ac_knnlaplace_optim_500_new = [basepath_new + "acrobot/offline_learning/knn/learning/k3_laplace/timeout500/esarsa/step500_env/data_optimal/drop0/sweep_rep1/"]

ac_knnlaplace_avg_5k_new = [basepath_new + "acrobot/offline_learning/knn/learning/k3_laplace/timeout500/esarsa/step5k_env/data_suboptimal/drop0/sweep_rep1/"]
ac_knnlaplace_bad_5k_new = [basepath_new + "acrobot/offline_learning/knn/learning/k3_laplace/timeout500/esarsa/step5k_env/data_subsuboptimal/drop0/sweep_rep1/"]

# PLOT 3
# ac_knnlaplace_optim_5k = [basepath + "acrobot/offline_learning/knn/learning/k3_laplace/timeout500/esarsa/step5k_env/data_optimal/drop0/sweep_rep1/"]
ac_knnlaplace_suboptim_5k = [basepath + "acrobot/offline_learning/knn/learning/k3_laplace/timeout500/esarsa/step5k_env/data_suboptimal/drop0/sweep_rep1/"]
ac_knnlaplace_subsuboptim_5k = [basepath + "acrobot/offline_learning/knn/learning/k3_laplace/timeout500/esarsa/step5k_env/data_subsuboptimal/drop0/sweep_rep1/"]
