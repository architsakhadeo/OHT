import os
import os.path
import numpy as np
import matplotlib.pyplot as plt
from itertools import product

def get_log_seed(log_path):
    with open(log_path) as f:
        log = f.readlines()

    seed = 0
    for line in log:
        if line.startswith('seed'):
            seed = int(line.split('=')[-1])
            return seed
    print(f'Unable to find seed for {log_path}')
    return seed

def get_final_prog(prog_path):
    with open(prog_path) as f:
        prog = f.readlines()

    final_prog = np.float64(prog[-1].split(',')[0])
    return final_prog

def plot_train_valid_prog(prog_path):
    with open(prog_path) as f:
        prog = f.readlines()[1:]

    t = np.arange(len(prog), dtype=np.float64)
    prog = np.array([list(map(float, p.split(','))) for p in prog])
    train_err, valid_err = prog[:, 0], prog[:, 1]
    plt.figure(figsize=(18, 12))
    plt.plot(t, train_err, 'b', label='train')
    plt.plot(t, valid_err, 'g', label='valid')
    # plt.title(f'Param {os.path.basename(os.path.dirname(prog_path))}')
    plt.title(f'Param {prog_path}')
    plt.xlabel('Iteration (x1000)')
    plt.ylabel('MSTDE')
    plt.legend(loc='upper right')
    plt.show()
    # print()


def get_hyperparam(log_path):
    with open(log_path) as f:
        log = f.readlines()

    alpha = 0
    nn_hidden = '32,32'
    for line in log:
        if line.startswith('alpha'):
            alpha = line.split('=')[-1][:-1]
        elif 'hidden' in line:
            nn_hidden = line.split('=')[-1][:-1]
            return alpha, nn_hidden
    print(f'Unable to find alpha for {log_path}')
    return alpha, nn_hidden

def process_results(data_path, weight_path, dataset_num=30, plot_prog=False):
    rankings = [dict() for i in range(dataset_num)]

    data_params = os.listdir(data_path)
    for param_dir in data_params:
        param_id = int(param_dir.split('_')[-1])
        data_param_path = os.path.join(data_path, param_dir)
        data_id = prog_val = -1
        files = os.listdir(data_param_path)
        if 'progs-0.csv' not in files:
            continue
        if plot_prog:
            plot_train_valid_prog(os.path.join(data_param_path, 'progs-0.csv'))
            continue
            # exit(0)
        for filename in files:
            if filename == 'log_json.txt':
                data_id = get_log_seed(os.path.join(data_param_path, filename))
            elif filename == 'progs-0.csv':
                prog_val = get_final_prog(os.path.join(data_param_path, filename))
        if data_id == -1 or prog_val == -1:
            print(f'Error in {param_dir}')
            continue
        rankings[data_id][param_id] = prog_val

    if plot_prog:
        return

    top_params = []
    for param_dict in rankings:
        best_param = min(param_dict, key=param_dict.get)
        best_weight = os.path.join(weight_path, f'param_{best_param}')
        print(f'"{best_weight}/",')
        alpha, nn_hidden = get_hyperparam(os.path.join(data_path, f'param_{best_param}/log_json.txt'))
        print(f'alpha={alpha}, nn_hidden={nn_hidden}')
        top_params.append(best_param)

    # top_weights = [os.path.join(weight_path, f'param_{i}') for i in top_params]
    # for best_weight in top_weights:
    #     print(f'"{best_weight}",')


if __name__ == "__main__":
    # cartpole_weight = "weight/hyperparam/cartpole/fqi-linear/step10k_env/fixed_eps0/"
    # cartpole_data = "data/hyperparam/cartpole-noisy-action/noise_1perc/offline_learning/fqi-linear/fqi-adam/step10k_env/data_eps0/lockat_baseline/"
    # print('best params for cartpole (eps 0):')
    # process_results(cartpole_data, cartpole_weight, dataset_num=10)

    # acrobot_weight = "weight/hyperparam/acrobot/fqi-linear/step10k_env/fixed_eps0/"
    # acrobot_data = "data/hyperparam/acrobot/offline_learning/fqi-linear/fqi-adam/step10k_env/data_eps0/lockat_baseline/"
    # print('best params for acrobot (eps 0):')
    # process_results(acrobot_data, acrobot_weight, dataset_num=10)

    # acrobot_weight_025 = "weight/hyperparam/acrobot/fqi-linear/step10k_env/fixed_eps0.25/"
    # acrobot_data_025 = "data/hyperparam/acrobot/offline_learning/fqi-linear/fqi-adam/step10k_env/data_eps0.25/lockat_baseline/"
    # print('best params for acrobot (eps 0.25):')
    # process_results(acrobot_data_025, acrobot_weight_025, dataset_num=10)

    # acrobot_weight_1 = "weight/hyperparam/acrobot/fqi-linear/step10k_env/fixed_eps1/"
    # acrobot_data_1 = "data/hyperparam/acrobot/offline_learning/fqi-linear/fqi-adam/step10k_env/data_eps1/lockat_baseline/"
    # print('best params for acrobot (eps 1):')
    # process_results(acrobot_data_1, acrobot_weight_1, dataset_num=10)

    # offline training with TC
    # Adam
    # acrobot_linear_weight = 'weight/hyperparam_ap/acrobot/random_restarts/fqi-linear/step15k_env/fixed_eps0/lambda0/'
    # acrobot_linear_data = 'data/hyperparam_ap/acrobot/offline_learning/random_restarts/fqi-linear/fqi-adam/alpha_hidden_epsilon/step15k_env/data_eps0/lambda0/lockat_baseline/'
    # print('best params for acrobot linear:')
    # process_results(acrobot_linear_data, acrobot_linear_weight, dataset_num=30)

    # puddle_linear_weight = 'weight/hyperparam_ap/puddleworld/random_restarts/fqi-linear/step15k_env/fixed_eps0/lambda0/'
    # puddle_linear_data = 'data/hyperparam_ap/puddleworld/offline_learning/random_restarts/fqi-linear/fqi-adam/alpha_hidden_epsilon/step15k_env/data_eps0/lambda0/lockat_baseline/'
    # print('best params for puddle linear:')
    # process_results(puddle_linear_data, puddle_linear_weight, dataset_num=30)

    # SGD
    # acrobot_linear_sgd_weight = 'weight/hyperparam_ap/acrobot/random_restarts/fqi-linear/fqi-sgd/step15k_env/fixed_eps0/lambda0/'
    # acrobot_linear_sgd_data = 'data/hyperparam_ap/acrobot/offline_learning/random_restarts/fqi-linear/fqi-sgd/alpha_hidden_epsilon/step15k_env/data_eps0/lambda0/lockat_baseline/'
    # print('best params for acrobot linear SGD:')
    # process_results(acrobot_linear_sgd_data, acrobot_linear_sgd_weight, dataset_num=30)

    # puddle_linear_sgd_weight = 'weight/hyperparam_ap/puddleworld/random_restarts/fqi-linear/fqi-sgd/step15k_env/fixed_eps0/lambda0/'
    # puddle_linear_sgd_data = 'data/hyperparam_ap/puddleworld/offline_learning/random_restarts/fqi-linear/fqi-sgd/alpha_hidden_epsilon/step15k_env/data_eps0/lambda0/lockat_baseline/'
    # print('best params for puddle linear SGD:')
    # process_results(puddle_linear_sgd_data, puddle_linear_sgd_weight, dataset_num=30)

    # Offline training with NN
    # Adam
    # acrobot_weight = 'weight/hyperparam_ap/acrobot/random_restarts/fqi/step15k_env/fixed_eps0/lambda0/'
    # acrobot_data = 'data/hyperparam_ap/acrobot/offline_learning/random_restarts/fqi/fqi-adam/alpha_hidden_epsilon/step15k_env/data_eps0/lambda0/lockat_baseline/'
    # print('best params for acrobot:')
    # process_results(acrobot_data, acrobot_weight, dataset_num=30)

    # puddle_weight = 'weight/hyperparam_ap/puddleworld/random_restarts/fqi/step15k_env/fixed_eps0/lambda0/'
    # puddle_data = 'data/hyperparam_ap/puddleworld/offline_learning/random_restarts/fqi/fqi-adam/alpha_hidden_epsilon/step15k_env/data_eps0/lambda0/lockat_baseline/'
    # print('best params for puddle:')
    # process_results(puddle_data, puddle_weight, dataset_num=30)

    # Adam, early stop
    envs = ['puddlerand']#['acrobot', 'puddlerand']
    algs = ['fqi']#, 'fqi-linear']
    num_steps = [5]#, 15]
    l2reg_scales = [5]#[1, 3, 5]
    for env, alg, num_step, l2reg_scale in product(envs, algs, num_steps, l2reg_scales):
        earlystop_weight = f'weight/hyperparam_v5/{env}/random_restarts/{alg}/step{num_step}k_env/optimalfixed_eps0/earlystop/lambda1e-{l2reg_scale}/'
        earlystop_data = f'data/hyperparam_v5/{env}/offline_learning/random_restarts/{alg}/fqi-adam/alpha_hidden_epsilon/step{num_step}k_env/optimalfixed_eps0/earlystop/lambda1e-{l2reg_scale}/lockat_baseline/'
        print('best params for earlystop:')
        process_results(earlystop_data, earlystop_weight, dataset_num=30, plot_prog=True)



    # SGD
    # acrobot_sgd_weight = 'weight/hyperparam_ap/acrobot/random_restarts/fqi/fqi-sgd/step15k_env/fixed_eps0/lambda0/'
    # acrobot_sgd_data = 'data/hyperparam_ap/acrobot/offline_learning/random_restarts/fqi/fqi-sgd/alpha_hidden_epsilon/step15k_env/data_eps0/lambda0/lockat_baseline/'
    # print('best params for acrobot SGD:')
    # process_results(acrobot_sgd_data, acrobot_sgd_weight, dataset_num=30)

    # puddle_sgd_weight = 'weight/hyperparam_ap/puddleworld/random_restarts/fqi/fqi-sgd/step15k_env/fixed_eps0/lambda0/'
    # puddle_sgd_data = 'data/hyperparam_ap/puddleworld/offline_learning/random_restarts/fqi/fqi-sgd/alpha_hidden_epsilon/step15k_env/data_eps0/lambda0/lockat_baseline/'
    # print('best params for puddle SGD:')
    # process_results(puddle_sgd_data, puddle_sgd_weight, dataset_num=30)
