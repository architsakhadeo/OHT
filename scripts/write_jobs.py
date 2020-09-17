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


#write_job(1, 30, ["config/hyperparam/cartpole/offline_data/dqn/step10k_env/lockat_-0.1.json",
#                   "config/hyperparam/cartpole/offline_data/dqn/step10k_env/lockat_baseline.json",
#                   "config/hyperparam/cartpole/offline_data/dqn/step10k_env/lockat_halfbaseline.json",
#                   "config/hyperparam/cartpole/offline_data/dqn/step10k_env/lockat_quarterbaseline.json",
#                   "config/hyperparam/cartpole/offline_data/dqn/step10k_env/lockat_random.json",
#                   "config/hyperparam/cartpole/offline_data/dqn/step1k_env/lockat_-0.1.json",
#                   "config/hyperparam/cartpole/offline_data/dqn/step1k_env/lockat_baseline.json",
#                   "config/hyperparam/cartpole/offline_data/dqn/step1k_env/lockat_halfbaseline.json",
#                   "config/hyperparam/cartpole/offline_data/dqn/step1k_env/lockat_quarterbaseline.json",
#                   "config/hyperparam/cartpole/offline_data/dqn/step1k_env/lockat_random.json",
#                   "config/hyperparam/cartpole/offline_data/dqn/step20k_env/lockat_-0.1.json",
#                   "config/hyperparam/cartpole/offline_data/dqn/step20k_env/lockat_baseline.json",
#                   "config/hyperparam/cartpole/offline_data/dqn/step20k_env/lockat_halfbaseline.json",
#                   "config/hyperparam/cartpole/offline_data/dqn/step20k_env/lockat_quarterbaseline.json",
#                   "config/hyperparam/cartpole/offline_data/dqn/step20k_env/lockat_random.json"]
#           , prev=0, line_per_file=30)

# write_job(5, 30, ["config/hyperparam/cartpole/offline_learning/dqn-adam/step10k_env/lockat_-0.1.json",
#                   "config/hyperparam/cartpole/offline_learning/dqn-adam/step10k_env/lockat_baseline.json",
#                   "config/hyperparam/cartpole/offline_learning/dqn-adam/step10k_env/lockat_halfbaseline.json",
#                   "config/hyperparam/cartpole/offline_learning/dqn-adam/step10k_env/lockat_quarterbaseline.json",
#                   "config/hyperparam/cartpole/offline_learning/dqn-adam/step10k_env/lockat_random.json",
#                   "config/hyperparam/cartpole/offline_learning/dqn-adam/step1k_env/lockat_-0.1.json",
#                   "config/hyperparam/cartpole/offline_learning/dqn-adam/step1k_env/lockat_baseline.json",
#                   "config/hyperparam/cartpole/offline_learning/dqn-adam/step1k_env/lockat_halfbaseline.json",
#                   "config/hyperparam/cartpole/offline_learning/dqn-adam/step1k_env/lockat_quarterbaseline.json",
#                   "config/hyperparam/cartpole/offline_learning/dqn-adam/step1k_env/lockat_random.json",
#                   ]
#           , prev=0, line_per_file=40)

# write_job(5, 30, ["config/hyperparam/cartpole/online_learning/dqn-sgd/step50k/sweep_lr.json"]
#           , prev=75, line_per_file=20)

write_job(27, 30, ["config/hyperparam/cartpole/offline_learning/esarsa-adam/step10k_env/lockat_-0.1.json",
                   "config/hyperparam/cartpole/offline_learning/esarsa-adam/step10k_env/lockat_baseline.json",
                   "config/hyperparam/cartpole/offline_learning/esarsa-adam/step10k_env/lockat_halfbaseline.json",
                   "config/hyperparam/cartpole/offline_learning/esarsa-adam/step10k_env/lockat_quarterbaseline.json",
                   "config/hyperparam/cartpole/offline_learning/esarsa-adam/step10k_env/lockat_random.json",
		   "config/hyperparam/cartpole/offline_learning/esarsa-adam/step1k_env/lockat_-0.1.json",
                   "config/hyperparam/cartpole/offline_learning/esarsa-adam/step1k_env/lockat_baseline.json",
                   "config/hyperparam/cartpole/offline_learning/esarsa-adam/step1k_env/lockat_halfbaseline.json",
                   "config/hyperparam/cartpole/offline_learning/esarsa-adam/step1k_env/lockat_quarterbaseline.json",
                   "config/hyperparam/cartpole/offline_learning/esarsa-adam/step1k_env/lockat_random.json",
		   "config/hyperparam/cartpole/offline_learning/esarsa-adam/step20k_env/lockat_-0.1.json",
                   "config/hyperparam/cartpole/offline_learning/esarsa-adam/step20k_env/lockat_baseline.json",
                   "config/hyperparam/cartpole/offline_learning/esarsa-adam/step20k_env/lockat_halfbaseline.json",
                   "config/hyperparam/cartpole/offline_learning/esarsa-adam/step20k_env/lockat_quarterbaseline.json",
                   "config/hyperparam/cartpole/offline_learning/esarsa-adam/step20k_env/lockat_random.json",
		   "config/hyperparam/cartpole/online_learning/esarsa-adam/step50k/sweep_lr.json"]
           , prev=0, line_per_file=30)

#write_job(5, 30, ["config/hyperparam/cartpole/online_learning/esarsa-adam/step50k/sweep_lr.json"]
#          , prev=150, line_per_file=5)
