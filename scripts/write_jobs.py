def write_job(sweep, run, config_list, prev=0, line_per_file=1):
    f = open("tasks_{}.sh".format(prev), "w")
    count = 0
    for config in config_list:
        for s in range(sweep):
            for r in range(run):
                f.write("./main -config {} -run {} -sweep {}\n".format(config, r, s))
                count += 1
                if count % line_per_file == 0:
                    f.close()
                    prev += 1
                    f = open("tasks_{}.sh".format(prev), "w")

write_job(27, 30, [
            "config/hyperparam/cartpole/offline_learning/knn-ensemble/k1/esarsa/step7.5k_env/drop0.1/ensembleseed0/lockat_baseline.json",
            "config/hyperparam/cartpole/offline_learning/knn-ensemble/k1/esarsa/step7.5k_env/drop0.1/ensembleseed1/lockat_baseline.json",
            "config/hyperparam/cartpole/offline_learning/knn-ensemble/k1/esarsa/step7.5k_env/drop0.1/ensembleseed2/lockat_baseline.json",
            "config/hyperparam/cartpole/offline_learning/knn-ensemble/k1/esarsa/step7.5k_env/drop0.1/ensembleseed3/lockat_baseline.json",
            "config/hyperparam/cartpole/offline_learning/knn-ensemble/k1/esarsa/step7.5k_env/drop0.1/ensembleseed4/lockat_baseline.json",
            "config/hyperparam/cartpole/offline_learning/knn-ensemble/k1/esarsa/step7.5k_env/drop0.1/ensembleseed5/lockat_baseline.json",
            "config/hyperparam/cartpole/offline_learning/knn-ensemble/k1/esarsa/step7.5k_env/drop0.2/ensembleseed1/lockat_baseline.json",
            "config/hyperparam/cartpole/offline_learning/knn-ensemble/k1/esarsa/step7.5k_env/drop0.2/ensembleseed2/lockat_baseline.json",
            "config/hyperparam/cartpole/offline_learning/knn-ensemble/k1/esarsa/step7.5k_env/drop0.2/ensembleseed3/lockat_baseline.json",
            "config/hyperparam/cartpole/offline_learning/knn-ensemble/k1/esarsa/step7.5k_env/drop0.2/ensembleseed4/lockat_baseline.json",
            "config/hyperparam/cartpole/offline_learning/knn-ensemble/k1/esarsa/step7.5k_env/drop0.2/ensembleseed5/lockat_baseline.json",
            "config/hyperparam/cartpole/offline_learning/knn-ensemble/k1/esarsa/step7.5k_env/drop0.3/ensembleseed1/lockat_baseline.json",
            "config/hyperparam/cartpole/offline_learning/knn-ensemble/k1/esarsa/step7.5k_env/drop0.3/ensembleseed2/lockat_baseline.json",
            "config/hyperparam/cartpole/offline_learning/knn-ensemble/k1/esarsa/step7.5k_env/drop0.3/ensembleseed3/lockat_baseline.json",
            "config/hyperparam/cartpole/offline_learning/knn-ensemble/k1/esarsa/step7.5k_env/drop0.3/ensembleseed4/lockat_baseline.json",
            "config/hyperparam/cartpole/offline_learning/knn-ensemble/k1/esarsa/step7.5k_env/drop0.3/ensembleseed5/lockat_baseline.json",

            "config/hyperparam/cartpole/offline_learning/knn-ensemble/k3/esarsa/step7.5k_env/drop0.1/ensembleseed0/lockat_baseline.json",
            "config/hyperparam/cartpole/offline_learning/knn-ensemble/k3/esarsa/step7.5k_env/drop0.1/ensembleseed1/lockat_baseline.json",
            "config/hyperparam/cartpole/offline_learning/knn-ensemble/k3/esarsa/step7.5k_env/drop0.1/ensembleseed2/lockat_baseline.json",
            "config/hyperparam/cartpole/offline_learning/knn-ensemble/k3/esarsa/step7.5k_env/drop0.1/ensembleseed3/lockat_baseline.json",
            "config/hyperparam/cartpole/offline_learning/knn-ensemble/k3/esarsa/step7.5k_env/drop0.1/ensembleseed4/lockat_baseline.json",
            "config/hyperparam/cartpole/offline_learning/knn-ensemble/k3/esarsa/step7.5k_env/drop0.1/ensembleseed5/lockat_baseline.json",
            "config/hyperparam/cartpole/offline_learning/knn-ensemble/k3/esarsa/step7.5k_env/drop0.2/ensembleseed1/lockat_baseline.json",
            "config/hyperparam/cartpole/offline_learning/knn-ensemble/k3/esarsa/step7.5k_env/drop0.2/ensembleseed2/lockat_baseline.json",
            "config/hyperparam/cartpole/offline_learning/knn-ensemble/k3/esarsa/step7.5k_env/drop0.2/ensembleseed3/lockat_baseline.json",
            "config/hyperparam/cartpole/offline_learning/knn-ensemble/k3/esarsa/step7.5k_env/drop0.2/ensembleseed4/lockat_baseline.json",
            "config/hyperparam/cartpole/offline_learning/knn-ensemble/k3/esarsa/step7.5k_env/drop0.2/ensembleseed5/lockat_baseline.json",
            "config/hyperparam/cartpole/offline_learning/knn-ensemble/k3/esarsa/step7.5k_env/drop0.3/ensembleseed1/lockat_baseline.json",
            "config/hyperparam/cartpole/offline_learning/knn-ensemble/k3/esarsa/step7.5k_env/drop0.3/ensembleseed2/lockat_baseline.json",
            "config/hyperparam/cartpole/offline_learning/knn-ensemble/k3/esarsa/step7.5k_env/drop0.3/ensembleseed3/lockat_baseline.json",
            "config/hyperparam/cartpole/offline_learning/knn-ensemble/k3/esarsa/step7.5k_env/drop0.3/ensembleseed4/lockat_baseline.json",
            "config/hyperparam/cartpole/offline_learning/knn-ensemble/k3/esarsa/step7.5k_env/drop0.3/ensembleseed5/lockat_baseline.json",

            "config/hyperparam/cartpole/offline_learning/knn-ensemble/k5/esarsa/step7.5k_env/drop0.1/ensembleseed0/lockat_baseline.json",
            "config/hyperparam/cartpole/offline_learning/knn-ensemble/k5/esarsa/step7.5k_env/drop0.1/ensembleseed1/lockat_baseline.json",
            "config/hyperparam/cartpole/offline_learning/knn-ensemble/k5/esarsa/step7.5k_env/drop0.1/ensembleseed2/lockat_baseline.json",
            "config/hyperparam/cartpole/offline_learning/knn-ensemble/k5/esarsa/step7.5k_env/drop0.1/ensembleseed3/lockat_baseline.json",
            "config/hyperparam/cartpole/offline_learning/knn-ensemble/k5/esarsa/step7.5k_env/drop0.1/ensembleseed4/lockat_baseline.json",
            "config/hyperparam/cartpole/offline_learning/knn-ensemble/k5/esarsa/step7.5k_env/drop0.1/ensembleseed5/lockat_baseline.json",
            "config/hyperparam/cartpole/offline_learning/knn-ensemble/k5/esarsa/step7.5k_env/drop0.2/ensembleseed1/lockat_baseline.json",
            "config/hyperparam/cartpole/offline_learning/knn-ensemble/k5/esarsa/step7.5k_env/drop0.2/ensembleseed2/lockat_baseline.json",
            "config/hyperparam/cartpole/offline_learning/knn-ensemble/k5/esarsa/step7.5k_env/drop0.2/ensembleseed3/lockat_baseline.json",
            "config/hyperparam/cartpole/offline_learning/knn-ensemble/k5/esarsa/step7.5k_env/drop0.2/ensembleseed4/lockat_baseline.json",
            "config/hyperparam/cartpole/offline_learning/knn-ensemble/k5/esarsa/step7.5k_env/drop0.2/ensembleseed5/lockat_baseline.json",
            "config/hyperparam/cartpole/offline_learning/knn-ensemble/k5/esarsa/step7.5k_env/drop0.3/ensembleseed1/lockat_baseline.json",
            "config/hyperparam/cartpole/offline_learning/knn-ensemble/k5/esarsa/step7.5k_env/drop0.3/ensembleseed2/lockat_baseline.json",
            "config/hyperparam/cartpole/offline_learning/knn-ensemble/k5/esarsa/step7.5k_env/drop0.3/ensembleseed3/lockat_baseline.json",
            "config/hyperparam/cartpole/offline_learning/knn-ensemble/k5/esarsa/step7.5k_env/drop0.3/ensembleseed4/lockat_baseline.json",
            "config/hyperparam/cartpole/offline_learning/knn-ensemble/k5/esarsa/step7.5k_env/drop0.3/ensembleseed5/lockat_baseline.json",

            "config/hyperparam/cartpole/online_learning/esarsa-adam/step50k/sweep.json"
            ]
          , prev=0, line_per_file=20) #1h
