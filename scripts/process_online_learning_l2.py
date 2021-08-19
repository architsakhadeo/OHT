import os
import os.path
import numpy as np
from itertools import product

def compute_mean_totals(data_param_path, dataset_num=30):
    results_files = os.listdir(data_param_path)
    totals_files = [f for f in results_files if 'totals' in f]
    assert len(totals_files) == dataset_num
    total_rewards = []
    total_episodes = []
    for totals_f in totals_files:
        with open(os.path.join(data_param_path, totals_f)) as f:
            totals = f.readlines()
        totals = totals[-1].split(',')
        total_rewards.append(float(totals[0]))
        total_episodes.append(int(totals[1]))
    return np.mean(total_rewards), np.mean(total_episodes)

def compute_mean_episode_lens(data_param_path, dataset_num=30):
    results_files = os.listdir(data_param_path)
    episodes_files = [f for f in results_files if 'episodes' in f]
    assert len(episodes_files) == dataset_num
    total_episode_lens = []
    for episodes_f in episodes_files:
        with open(os.path.join(data_param_path, episodes_f)) as f:
            episode_lens = f.readlines()
        episode_lens = episode_lens[1:]
        episode_lens = [int(e) for e in episode_lens]
        total_episode_lens.append(np.mean(episode_lens))
    return np.mean(total_episode_lens)

def process_online_l2(data_path, dataset_num=30):
    reward_means = []
    episode_means = []
    for param_dir in os.listdir(os.path.abspath(data_path)):
        param_id = int(param_dir.split('_')[-1])
        data_param_path = os.path.join(data_path, param_dir)
        r_mean, e_mean = compute_mean_totals(data_param_path)
        reward_means.append(r_mean)
        episode_means.append(e_mean)
    # print(f'Index for min reward: {reward_means.index(min(reward_means))}, for max e_len: {episode_means.index(max(episode_means))}')
    print(f'median epi nums: {np.mean(episode_means)}, rewards: {np.mean(reward_means)}')
    if 'acrobot' in data_path:
        print(f'num steps to succ: {15000.0 / np.mean(episode_means)}')
    else:
        print(f'return per episode: {np.mean(np.array(reward_means)/np.array(episode_means))}')

def process_online_eplen(data_path, dataset_num=30):
    episode_len_means = []
    for param_dir in os.listdir(os.path.abspath(data_path)):
        data_param_path = os.path.join(data_path, param_dir)
        e_len_mean = compute_mean_episode_lens(data_param_path)
        episode_len_means.append(e_len_mean)
    print(f'median epi lens: {np.mean(episode_len_means)}')

if __name__ == "__main__":
    # Linear TC
    envs = ['acrobot', 'puddlerand']
    algs = ['fqi-linear']   #['fqi', 'fqi-linear']
    num_steps = [30]#, 15]
    l2reg_scales = [1, 3, 5]

    # for env, alg, num_step, l2reg_scale in product(envs, algs, num_steps, l2reg_scales):
    #     print(f'env: {env}, alg: {alg}, num_step: {num_step}k, l2reg_scale: 1e-{l2reg_scale}')
    #     data_path = (f'data/hyperparam_v7/{env}/offline_learning/random_restarts/'+
    #             f'{alg}/fqi-adam/alpha_hidden_epsilon/step{num_step}k_env/'+
    #             f'optimalfixed_eps0/lambda1e-{l2reg_scale}/lockat_baseline_online/')
    #     if not os.path.isdir(os.path.abspath(data_path)):
    #         print('Path not found.')
    #         continue
    #     process_online_l2(data_path=data_path, dataset_num=30)

    print()
    # NN, early stopping
    algs = ['fqi']

    for env, alg, num_step, l2reg_scale in product(envs, algs, num_steps, l2reg_scales):
        print(f'\nenv: {env}, alg: {alg}, num_step: {num_step}k, l2reg_scale: 1e-{l2reg_scale}')
        data_path = (f'data/hyperparam_v7/{env}/offline_learning/random_restarts/'+
                f'{alg}/fqi-adam/alpha_hidden_epsilon/step{num_step}k_env/'+
                f'mixed_eps0/earlystop/lambda1e-{l2reg_scale}/lockat_baseline_online/')
        print('Checking:', data_path)
        if not os.path.isdir(os.path.abspath(data_path)):
            print('Path not found.')
            continue
        process_online_l2(data_path=data_path, dataset_num=30)


    # Transfer
    # envs = ['acrobot']
    # algs = ['fqi', 'fqi-linear']
    # num_steps = [5]
    # l2reg_scales = [3]
    # for env, alg, num_step, l2reg_scale in product(envs, algs, num_steps, l2reg_scales):
    #     print(f'env: {env}, alg: {alg}, num_step: {num_step}k, l2reg_scale: 1e-{l2reg_scale}')
    #     data_path = (f'data/hyperparam_v5/{env}_shift/policy_transfer/shift/load_default/{alg}/lambda1e-{l2reg_scale}/')
    #     if not os.path.isdir(os.path.abspath(data_path)):
    #         print('Path not found.')
    #         continue
    #     process_online_l2(data_path=data_path, dataset_num=30)
    # data_path = 'data/hyperparam/acrobot/offline_learning/fqi/fqi-adam/alpha_hidden_epsilon/step10k_env/data_eps0/lockat_baseline_online/'
    # if not os.path.isdir(os.path.abspath(data_path)):
    #     print('Path not found.')
    #     exit(0)
    # process_online_eplen(data_path=data_path, dataset_num=30)