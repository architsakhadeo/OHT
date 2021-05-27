pr_rnd = [34, 4, 43, 30, 24, 32, 40, 11, 20, 30, 3, 16, 53, 45, 0, 21, 43, 23, 44, 50, 9, 41, 37, 37, 11, 2, 26, 33, 18, 20]
basepath = "../../data/finalPlots/data/hyperparam_v5/"
basepath_new = "../../data/finalPlots/data/hyperparam_v5_newData/"

pr_true = [basepath + "puddlerand/online_learning/esarsa/step30k/sweep/"]
pr_fqi_tc = [basepath + "puddlerand/offline_learning/random_restarts/fqi-linear/fqi-adam/alpha_hidden_epsilon/step5k_env/optimalfixed_eps0/lambda1e-5/lockat_baseline_online/"]
pr_fqi_nn = [basepath + "puddlerand/offline_learning/random_restarts/fqi/fqi-adam/alpha_hidden_epsilon/step5k_env/optimalfixed_eps0/earlystop/lambda1e-3/lockat_baseline_online/"]

# PLOT 1
pr_knnlaplace_optim_5k = [basepath + "puddlerand/offline_learning/knn/learning/k3_laplace/timeout1000/esarsa/step5k_env/data_optimal/drop0/sweep_rep1/"]
pr_knnraw_optim_5k = [basepath + "puddlerand/offline_learning/knn/learning/k3/timeout1000/esarsa/step5k_env/data_optimal/drop0/sweep/"]
pr_networkscaledlaplace_optim_5k = [basepath + "puddlerand/offline_learning/network/learning/clip_scale_laplace_separated/timeout1000/esarsa/step5k_env/data_optimal/sweep/"]
pr_networkscaledraw_optim_5k = [basepath + "puddlerand/offline_learning/network/learning/clip_scale_separated/timeout1000/esarsa/step5k_env/data_optimal/sweep/"]

# PLOT 2
pr_knnlaplace_avg_5k_new = [basepath_new + "puddlerand/offline_learning/knn/learning/k3_laplace/timeout1000/esarsa/step5k_env/data_suboptimal/drop0/sweep_rep1/"]
pr_knnlaplace_avg_2500_new = [basepath_new + "puddlerand/offline_learning/knn/learning/k3_laplace/timeout1000/esarsa/step2.5k_env/data_suboptimal/drop0/sweep_rep1/"]
pr_knnlaplace_avg_1k_new = [basepath_new + "puddlerand/offline_learning/knn/learning/k3_laplace/timeout1000/esarsa/step1k_env/data_suboptimal/drop0/sweep_rep1/"]
pr_knnlaplace_avg_500_new = [basepath_new + "puddlerand/offline_learning/knn/learning/k3_laplace/timeout1000/esarsa/step500_env/data_suboptimal/drop0/sweep_rep1/"]

pr_knnlaplace_optim_5k_new = [basepath_new + "puddlerand/offline_learning/knn/learning/k3_laplace/timeout1000/esarsa/step5k_env/data_optimal/drop0/sweep_rep1/"]
pr_knnlaplace_bad_5k_new = [basepath_new + "puddlerand/offline_learning/knn/learning/k3_laplace/timeout1000/esarsa/step5k_env/data_subsuboptimal/drop0/sweep_rep1/"]

# PLOT 4

# pr_knnlaplace_optim_5k_plot3 = [basepath + "puddlerand/offline_learning/knn/learning/k3_laplace/timeout1000/esarsa/step5k_env/data_optimal/drop0/sweep_rep1/"]
# pr_knnlaplace_suboptim_5k_plot3 = [basepath + "puddlerand/offline_learning/knn/learning/k3_laplace/timeout1000/esarsa/step5k_env/data_suboptimal/drop0/sweep_rep1/"]
# pr_knnlaplace_subsuboptim_5k_plot3 = [basepath + "puddlerand/offline_learning/knn/learning/k3_laplace/timeout1000/esarsa/step5k_env/data_subsuboptimal/drop0/sweep_rep1/"]

