import include_parent_folder
import numpy as np
import os

import utils.log_and_plot_func as pf
from Experiments.control import saved_file_name
from utils.collect_config import Sweeper, ParameterConfig

def accumulate_reward(reward_step, num_steps):
    accum = np.zeros(num_steps)
    sum = 0
    for i in range(num_steps):
        sum += reward_step[i]
        accum[i] = sum
    return accum
def accum_reward_all_runs(folder, run_num, num_steps):
    one_setting = []
    run_count = 0
    for run_idx in range(run_num):
        name = "{}/run{}_rewardPerStep.npy".format(folder, str(run_idx))
        if os.path.isfile(name):
            reward_step = np.load(name)
            accum_r = accumulate_reward(reward_step, num_steps)
            one_setting.append(accum_r)
            run_count += 1
    if run_count > 0:
        one_setting = np.array(one_setting)
        np.save("{}/{}runs_accum_reward".format(folder, run_count), one_setting)
    else:
        print(folder, "doesn't exist")
    if len(one_setting) > 0:
        return one_setting
    else:
        return None

def return_per_ep(reward_step, step_ep):
    checked = 0
    returns = np.zeros(len(step_ep))
    for idx in range(len(step_ep)):
        num = step_ep[idx]
        returns[idx] = np.sum(reward_step[checked: checked + num])
        checked += num
    return np.array(returns)
def return_per_ep_all_runs(folder, run_num, scale_r=1):
    one_setting = []
    print(folder)
    run_count = 0
    for run_idx in range(run_num):
        reward_name = "{}/run{}_rewardPerStep.npy".format(folder, str(run_idx))
        step_name = "{}/run{}_stepPerEp.npy".format(folder, str(run_idx))
        if os.path.isfile(reward_name):
            reward_step = np.load(reward_name)
            step_ep = np.load(step_name)
            accum_r = return_per_ep(reward_step, step_ep)
            one_setting.append(accum_r)
            run_count += 1
    if run_count > 0:
        one_setting = np.array(one_setting)
        np.save("{}/{}runs_return_per_ep".format(folder, run_count), one_setting)
    else:
        print(folder, "doesn't exist")
    if len(one_setting) > 0:
        return one_setting * scale_r
    else:
        return None

def num_steps_per_ep_all_runs(folder, run_num, num_ep):
    one_setting = []
    print(folder)
    run_count = 0
    for run_idx in range(run_num):
        step_name = "{}/run{}_stepPerEp.npy".format(folder, str(run_idx))
        if os.path.isfile(step_name):
            step_ep = np.load(step_name)[:num_ep]
            one_setting.append(step_ep)
            run_count += 1
    if run_count > 0:
        one_setting = np.array(one_setting)
        np.save("{}/{}runs_step_per_ep".format(folder, run_count), one_setting)
    else:
        print(folder, "doesn't exist")
    if len(one_setting) > 0:
        return one_setting
    else:
        return None

# def control_exp(env_list, result_path, handcode=None):
#     for env in env_list:
#         control_exp_single_env(env, result_path, handcode=handcode)
def control_exp_single_env(env_name, result_path, handcode=None, eval=False):
    sweeper = Sweeper('../Parameters/{}.json'.format(env_name.lower()), "control_param")
    total_comb = sweeper.get_total_combinations()
    run_num = 30
    all_data = {}
    label = {}
    handcode_data = None
    for idx in range(total_comb):
        sweeper = Sweeper('../Parameters/{}.json'.format(env_name.lower()), "control_param")
        config = sweeper.parse(idx)
        agent_params = config.agent_params
        env_params = config.env_params
        num_steps = config.exp_params.num_steps
        agent_params_temp = config.agent_params
        setattr(agent_params_temp, "exp_result_path", result_path)
        config_temp = config
        config_temp.agent_params = agent_params_temp

        folder, _ = saved_file_name(config_temp, 0, eval=eval)
        if env_name.lower() in ["cp"]:
            ignore_zero = False
            exp_smooth = None
            best = "largeAUC"
            one_setting = return_per_ep_all_runs(folder, run_num, num_steps)
            lim_y = [0, 200]
            lim_x = [1, 10000]
            if handcode:
                handcode_data = return_per_ep_all_runs(handcode, run_num, num_steps)
        # # The following block plots steps per episode
        # elif env_name.lower() in ["ccp"]:
        #     ignore_zero = False
        #     exp_smooth = None
        #     num_ep = 150
        #     best = "largeAUC"
        #     one_setting = num_steps_per_ep_all_runs(folder, run_num, num_ep)
        #     lim_y = [0, 500]
        #     lim_x = [1, num_ep]
        #     if handcode:
        #         handcode_data_temp = num_steps_per_ep_all_runs(handcode, run_num, num_ep)
        #         fixed_length = num_ep
        #         handcode_data = []
        #         for i in handcode_data_temp:
        #             if len(i)>=fixed_length: handcode_data.append(i)
        #         handcode_data = np.array(handcode_data)
        #     else:
        #         handcode_data = None

        # The following block plots the number of failure cases
        elif env_name.lower() in ["ccp", "ccp-sarsa", "ccp-sarsa-model", "ccp-dqn", "ccp-dqn-model"]:
            ignore_zero = False
            exp_smooth = None
            best = "smallAUC"
            one_setting = accum_reward_all_runs(folder, run_num, num_steps)
            if one_setting is None:
                continue
            one_setting *= -1 # number of failures = -1 * accumulate reward
            lim_y = [0, 500]
            lim_x = [1, 50000]
            if handcode:
                handcode_data = accum_reward_all_runs(handcode, run_num, num_steps)
                handcode_data *= -1 # number of failures = -1 * accumulate reward
            else:
                handcode_data = None
        else:
            raise NotImplemented

        if one_setting is not None:
            key = ''
            if config.agent == "DQN":
                key = "B{}_sync{}_batch{}".format(
                    folder.split("_B")[1].split("_")[0],
                    folder.split("_sync")[1].split("_")[0],
                    folder.split("_batch")[1].split("_")[0]
                )
            elif config.agent == "ExpectedSarsaTileCodingContinuing":
                key = "tiling{}_tile{}_lmbda{}_epsilon{}".format(
                    folder.split("_TC")[1].split("x")[0],
                    folder.split("_TC")[1].split("_")[0].split("x")[1],
                    folder.split("_lmbda")[1].split("_")[0],
                    folder.split("_epsilon")[1].split("_")[0],
                )
            if env_params.drift_prob > 0:
                key += "drift_scale{}_life{}_prob{}".format(
                    folder.split("_scale")[1].split("_")[0],
                    folder.split("_life")[1].split("_")[0],
                    folder.split("_prob")[1].split("_")[0])
            elif env_params.drift_prob < 0:
                key += "drift_scale{}".format(
                    folder.split("_scale")[1].split("_")[0])
            else:
                key += "drift_none"
            if key in all_data.keys():
                all_data[key].append(one_setting)
                label[key].append(agent_params.alpha)
            else:
                all_data[key] = [one_setting]
                label[key] = [agent_params.alpha]
    save_path = "../data/plots/{}".format(env_name)
    pf.plot_control_exp_curve(all_data, label, lim_x, lim_y, ignore_zero=ignore_zero, exp_smooth=exp_smooth, save_path=None, handcode=handcode_data, best=best)

def multiple_agent(path_list, run_num, num_steps):
    all_data = {}
    all_label = {}
    for path, label in path_list:
        one_setting = accum_reward_all_runs(path, run_num, num_steps)
        one_setting *= -1
        if label in all_data.keys():
            all_data[label].append(one_setting)
            all_label[label].append(label)
        else:
            all_data[label] = [one_setting]
            all_label[label] = [label]
    pf.plot_control_exp_curve(all_data, all_label, [0, num_steps], [-10, 90])

def quick_test(path):
    reward = np.load(path)
    accum = np.zeros(len(reward))
    sum = 0
    for i in range(len(reward)):
        sum += reward[i]
        accum[i] = sum
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(accum * -1)
    plt.show()


if __name__ == '__main__':
    control_exp_single_env("ccp-sarsa-model", "../data/exp_result/simulatorTraining/handcode_pd8",
                           handcode="../data/offline_env/test/HandCoded/",
                           eval=True)
    control_exp_single_env("ccp-dqn-model", "../data/exp_result/simulatorTraining/handcode_pd8",
                           handcode="../data/offline_env/test/HandCoded/",
                           eval=True)
    # control_exp_single_env("ccp-sarsa", "../data/exp_result/",
    #                        handcode="../data/offline_env/test/HandCoded/",
    #                        eval=True)
    # control_exp_single_env("ccp-dqn", "../data/exp_result/",
    #                        handcode="../data/offline_env/test/HandCoded/",
    #                        eval=True)

    # multiple_agent(
    #     [["../data/exp_result/offlineTraining_simulatorEval/handcode_freq2/offline_eval/ContinuingCartpoleEnvironment_DQN_B500_sync25_NN[128, 128]_batch16_alpha0.001_inputObs_training10000", "DQN"],
    #      ["../data/exp_result/offlineTraining_simulatorEval/handcode_freq2/offline_eval/ContinuingCartpoleEnvironment_ExpectedSarsaTileCodingContinuing_lmbda0.6_epsilon0.2_alpha0.1_inputSep_pair_TC32x2_training10000", "Sarsa"],
    #      ["../data/offline_env/test/HandCoded/", "HandCoded"]],
    #     10,
    #     10000
    # )

    # quick_test("../data/offline_env/test/HandCoded_reward.npy")