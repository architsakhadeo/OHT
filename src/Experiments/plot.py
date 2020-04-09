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
    print(folder)
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

def num_steps_per_ep_all_runs(folder, run_num):
    one_setting = []
    print(folder)
    if os.path.isfile("{}/{}runs_step_per_ep.npy".format(folder, run_num)):
        one_setting = np.load("{}/{}runs_step_per_ep.npy".format(folder, run_num), allow_pickle=True)
    else:
        run_count = 0
        for run_idx in range(run_num):
            step_name = "{}/run{}_stepPerEp.npy".format(folder, str(run_idx))
            if os.path.isfile(step_name):
                step_ep = np.load(step_name)
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

def control_exp(env_list, result_path):
    for env in env_list:
        control_exp_single_env(env, result_path)
def control_exp_single_env(env_name, result_path):
    sweeper = Sweeper('../Parameters/{}.json'.format(env_name.lower()), "control_param")
    total_comb = sweeper.get_total_combinations()
    run_num = 30
    all_data = {}
    label = {}
    for idx in range(total_comb):
        sweeper = Sweeper('../Parameters/{}.json'.format(env_name.lower()), "control_param")
        config = sweeper.parse(idx)
        agent_params = config.agent_params
        num_steps = config.exp_params.num_steps
        agent_params_temp = config.agent_params
        setattr(agent_params_temp, "exp_result_path", result_path)
        config_temp = config
        config_temp.agent_params = agent_params_temp

        folder, _ = saved_file_name(config_temp, 0)
        if env_name.lower() in ["cp"]:
            ignore_zero = False
            exp_smooth = None
            one_setting = return_per_ep_all_runs(folder, run_num, num_steps)
            lim_y = [0, 200]
            lim_x = [1, 10000]
        elif env_name.lower() in ["ccp"]:
            ignore_zero = False
            exp_smooth = None
            # one_setting = num_steps_per_ep_all_runs(folder, run_num)
            one_setting = accum_reward_all_runs(folder, run_num, num_steps)
            lim_y = [-500, 0]
            # lim_y = [0, 500]
            lim_x = [1, 100000]
        else:
            raise NotImplemented

        if one_setting is not None:
            key = "B{}_sync{}".format(
                folder.split("_B")[1].split("_")[0],
                folder.split("_sync")[1].split("_")[0]
            )
            if key in all_data.keys():
                all_data[key].append(one_setting)
                label[key].append(agent_params.alpha)
            else:
                all_data[key] = [one_setting]
                label[key] = [agent_params.alpha]
    save_path = "../data/plots/{}".format(env_name)
    pf.plot_control_exp_curve(all_data, label, lim_x, lim_y, ignore_zero=ignore_zero, exp_smooth=exp_smooth, save_path=None)

if __name__ == '__main__':
    control_exp(["CCP"], "../data/exp_result")
