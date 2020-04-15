import matplotlib.pyplot as plt
import os
import numpy as np
from utils.collect_config import ParameterConfig
import pickle as pkl

def get_color_by_lable(label, index):
    new_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                  '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
                  '#bcbd22', '#17becf']

    color_dict = {
    }
    if label in color_dict.keys():
        return color_dict[label]
    else:
        return new_colors[index%len(new_colors)]

def write_param_log(agent_params, env_params, env, file_path, exp_params=None, save_pkl=False, print_log=False):
    if os.path.isfile(file_path + "/record.pkl"):
        print("Log exist")

    if save_pkl:
        obj = ParameterConfig()
        setattr(obj, "agent_params", agent_params)
        setattr(obj, "env_params", env_params)
        setattr(obj, "environment", env)
        with open(file_path+"/record.pkl", "wb") as param_obj:
            pkl.dump(obj, param_obj)
        print("param saved in", file_path)

    with open(file_path + "/param.txt", "w") as param_record:
        param_record.write("------ Agent parameters ------\n\n")
        est_len = 20
        for pair in agent_params.__dict__:
            space = " " * (est_len - len(str(pair))) + ": "
            if print_log: print(str(pair), space, str(agent_params.__dict__[pair]))
            info = str(pair) + space + str(agent_params.__dict__[pair]) + "\n"
            param_record.write(info)
        param_record.write("\n\n------ Environment parameters ------\n\n")
        param_record.write("Env: " + str(env) + "\n\n")
        for pair in env_params.__dict__:
            space = " " * (est_len - len(str(pair))) + ": "
            if print_log: print(str(pair), space, str(env_params.__dict__[pair]))
            info = str(pair) + space + str(env_params.__dict__[pair]) + "\n"
            param_record.write(info)
        if exp_params is not None:
            param_record.write("\n\n------ Control exp parameters ------\n\n")
            for pair in exp_params.__dict__:
                space = " " * (est_len - len(str(pair))) + ": "
                if print_log: print(str(pair), space, str(exp_params.__dict__[pair]))
                info = str(pair) + space + str(exp_params.__dict__[pair]) + "\n"
                param_record.write(info)

    print("log saved in", file_path)

def plot_control_exp_curve(all_data, label, lim_x, lim_y, ignore_zero=False, exp_smooth=None, save_path=None, num_color=None, handcode=None):
    best_lrs = {}
    for k in all_data.keys():

        plt.figure()
        plt.xlim(lim_x[0], lim_x[1])
        plt.ylim(lim_y[0], lim_y[1])
        print("\ntitle", k)
        mean, upper, lower, lr = plot_control_exp_curve_single_key(plt, all_data[k], label[k], lim_x, lim_y, ignore_zero=ignore_zero, exp_smooth=exp_smooth, total_number=num_color)
        best_lrs[k] = [mean, upper, lower, lr]

        plt.title(k)
        plt.legend()
        if save_path:
            plt.savefig(save_path + k + ".png")
            plt.close()
            plt.clf()
    if not save_path:
        plt.show()

    plt.figure()
    plt.xlim(lim_x[0], lim_x[1])
    plt.ylim(lim_y[0], lim_y[1])

    for bk in best_lrs.keys():
        mean, upper, lower, lr = best_lrs[bk]
        lb = str(bk)

        x = np.linspace(1, len(mean), len(mean))
        plt.plot(x, mean, label=lb)
        plt.fill_between(x, upper, lower, alpha=0.3)
        plt.xticks(lim_x, lim_x)
        curve = np.clip(mean, lim_y[0], lim_y[1])
        # auc = np.sum(curve - lim_y[0])
    if handcode is not None:
        mean, upper, lower = calculate_avg_default(handcode, exp_smooth=exp_smooth)
        x = np.linspace(1, len(mean), len(mean))
        plt.plot(x, mean, "--", label="hand_code")
    plt.title("best settings")
    plt.legend()
    plt.show()

def plot_control_exp_curve_single_key(canvas, all_data, labels, range_x, range_y, ignore_zero=False, exp_smooth=None, total_number=None):
    auc = np.zeros(len(all_data))
    for i in range(len(all_data)):
        c = get_color_by_lable(labels[i], i)
        if ignore_zero:
            mean, upper, lower = calculate_avg_ignoring_zero(all_data[i], exp_smooth=exp_smooth)
        else:
            mean, upper, lower = calculate_avg_default(all_data[i], exp_smooth=exp_smooth)

        x = np.linspace(1, len(mean), len(mean))
        canvas.plot(x, mean, label=labels[i], color=c)
        canvas.fill_between(x, upper, lower, facecolor=c, alpha=0.3)
        curve = mean[range_x[0]-1: range_x[1]]
        auc[i] = np.sum(curve[len(curve)//2:] - range_y[0])

    print("All auc =", auc)
    best_i = np.argmax(auc)
    mean, upper, lower = calculate_avg_default(all_data[best_i], exp_smooth=exp_smooth)
    print("Best setting =", np.max(auc), labels[best_i])
    return mean, upper, lower, labels[best_i]

def calculate_avg_ignoring_zero(data, exp_smooth=None):
    if exp_smooth is not None:
        data_temp = []
        for i in data:
            zero_idx = np.where(i == 0)[0]
            if len(zero_idx) == 0:
                data_temp.append(i)
            else:
                data_temp.append(i[:zero_idx[0]])
        data_temp = np.array(data_temp)
        data = exponential_smooth(data_temp, beta=exp_smooth)
    max_len = data.shape[1]
    num_run = data.shape[0]
    done = False
    i = 0
    mean = []
    ste = []
    while i < max_len and not done:
        bits = data[:, i]
        count_nonzero = np.where(bits!=0)[0]
        if len(count_nonzero) < (num_run * 0.5):
            done = True
        else:
            mean.append(np.sum(bits[count_nonzero]) / len(count_nonzero))
            ste.append(np.abs(np.std(bits[count_nonzero])) / np.sqrt(len(count_nonzero)))
            i += 1

    mean = np.array(mean)
    ste = np.array(ste)
    upper = mean + ste
    lower = mean - ste
    return mean, upper, lower


def calculate_avg_default(data, exp_smooth=None):
    if exp_smooth is not None:
        data = exponential_smooth(data, beta=exp_smooth)
    mean = data.mean(axis=0)
    ste = np.abs(np.std(data, axis=0)) / np.sqrt(len(data))
    upper = mean + ste
    lower = mean - ste
    return mean, upper, lower


def exponential_smooth(all_data, beta):
    max_len = np.max(np.array([len(i) for i in all_data]))
    all_row = np.zeros((len(all_data), max_len))
    for i in range(len(all_data)):
        data = all_data[i]
        J = 0
        new_data = np.zeros(len(data))
        for idx in range(len(data)):
            J *= (1-beta)
            J += beta
            rate = beta / J
            if idx == 0:
                new_data[idx] = data[idx] * rate
            else:
                new_data[idx] = data[idx] * rate + new_data[idx - 1] * (1 - rate)
        all_row[i, :len(new_data)] = new_data
    return all_row
