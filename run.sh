# puddlerand - online learning
# parallel ./main -config "config/hyperparam_v5/puddlerand/online_learning/ac/step30k/sweep.json" -run {1} -sweep {2} ::: {1..60} ::: {0..35}
parallel ./main -config "config/hyperparam_v5/puddlerand/online_learning/dqn/step150k/sweep.json" -run {1} -sweep {2} ::: {1..60} ::: {0..23}

# puddlerand - offline learning
# parallel ./main -config "config/hyperparam_v5/puddlerand/offline_learning/knn/learning/k3/timeout500/ac/step15k_env/data_optimal/drop0/sweep.json" -run {1} -sweep {2} ::: {1..60} ::: {0..35}
# parallel ./main -config "config/hyperparam_v5/puddlerand/offline_learning/knn/learning/k3_laplace/timeout500/ac/step15k_env/data_optimal/drop0/sweep_rep0.json" -run {1} -sweep {2} ::: {1..60} ::: {0..35}
parallel ./main -config "config/hyperparam_v5/puddlerand/offline_learning/knn/learning/k3/timeout500/dqn/step15k_env/data_optimal/drop0/sweep.json" -run {1} -sweep {2} ::: {1..60} ::: {0..23}
parallel ./main -config "config/hyperparam_v5/puddlerand/offline_learning/knn/learning/k3_laplace/timeout500/dqn/step15k_env/data_optimal/drop0/sweep_rep0.json" -run {1} -sweep {2} ::: {1..60} ::: {0..23}


# # acrobot - online learning
# parallel ./main -config "config/hyperparam_v5/acrobot/online_learning/ac/step30k/sweep.json" -run {1} -sweep {2} ::: {1..60} ::: {0..35}
parallel ./main -config "config/hyperparam_v5/acrobot/online_learning/dqn/step150k/sweep.json" -run {1} -sweep {2} ::: {1..60} ::: {0..35}

# # acrobot - offline learning
# parallel ./main -config "config/hyperparam_v5/acrobot/offline_learning/knn/learning/k3/timeout500/ac/step15k_env/data_optimal/drop0/sweep.json" -run {1} -sweep {2} ::: {1..60} ::: {0..35}
# parallel ./main -config "config/hyperparam_v5/acrobot/offline_learning/knn/learning/k3_laplace/timeout500/ac/step15k_env/data_optimal/drop0/sweep_rep1.json" -run {1} -sweep {2} ::: {1..60} ::: {0..35}
parallel ./main -config "config/hyperparam_v5/acrobot/offline_learning/knn/learning/k3/timeout500/dqn/step15k_env/data_optimal/drop0/sweep.json" -run {1} -sweep {2} ::: {1..60} ::: {0..23}
parallel ./main -config "config/hyperparam_v5/acrobot/offline_learning/knn/learning/k3_laplace/timeout500/dqn/step15k_env/data_optimal/drop0/sweep.json" -run {1} -sweep {2} ::: {1..60} ::: {0..23}