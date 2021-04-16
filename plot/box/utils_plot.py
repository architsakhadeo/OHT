import os
import sys
cwd = os.getcwd()
sys.path.insert(0, cwd+'/../..')
from plot.box.utils_data import *
import matplotlib
import matplotlib.pyplot as plt
# c_default = matplotlib.cm.get_cmap('cool')
# c_default = matplotlib.cm.get_cmap('hsv')

# colorblind friendly color cycle https://gist.github.com/thriveth/8560036
c_default =  ['#377eb8', '#ff7f00', '#4daf4a',
              '#f781bf', '#a65628', '#984ea3',
              '#999999', '#e41a1c', '#dede00']
c_dict = {
    "calibration": '#377eb8',
    "calibration (grid search)": '#377eb8',
    "random": '#ff7f00',
    "fqi": '#4daf4a',
    "cem": '#f781bf',
    "calibration (cem)": '#a65628',

    "no Adversarial": '#984ea3',
    "no Ensemble": '#999999',

    "10k": "red",
    "5k": "tab:blue",
    "2k": "mediumseagreen",
    "1k": "orange",

    "eps 0": "tab:blue",
    "eps 0.25": "lightgreen",
    "eps 1": "orange",
    "training": "grey",

    "reward -0.02": '#377eb8',
    "reward -0.01": '#f781bf',
    "reward -0.004": '#4daf4a',
    "reward -0.002": '#dede00',

    "return -320": '#4daf4a',
    "return -40": '#dede00',

    "return -360": '#4daf4a',
    "return -45": '#dede00',

    "random data": '#e41a1c'
}
m_default = [".", "^", "+", "*", "s", "D", "h", "H", "."]
m_dict = {
    "eps 0": ".",
    "eps 0.25": "^",
    "eps 1": "+",
}

def cmap(key, idx):
    if key in c_dict.keys():
        return c_dict[key]
    else:
        # return c_default(idx)
        return c_default[int(len(c_default)*idx)]#[int(idx%len(c_default))]
def mmap(key, idx):
    if key in m_dict.keys():
        return m_dict[key]
    else:
        return m_default[idx%len(m_default)]

def plot_each_run(te, cms, source, title, ylim=None, outer=None, sparse_reward=None, max_len=np.inf):
    te_data = loading_average(te, source, outer=outer, sparse_reward=sparse_reward, max_len=max_len)
    te_rank = ranking_allruns(te_data)
    te_rank = te_rank["true"]

    cms_data = loading_average(cms, source, outer=outer, sparse_reward=sparse_reward, max_len=max_len)

    m_lst = list(cms_data.keys())
    all_models = []
    for model in m_lst:
        one_model = []
        for rk in te_rank.keys():
            one_run = []
            for e in te_rank[rk]:
                pk = e[0]
                one_run.append(cms_data[model][rk][pk])
            one_model.append(one_run)
        all_models.append(one_model)
    plot_scatters(all_models, m_lst, title)

    # for rk in te_rank.keys():
    #     one_run = []
    #     m_lst = list(cms_data.keys())
    #     for model in m_lst:
    #         model_rank = []
    #         for e in te_rank[rk]:
    #             pk = e[0]
    #             model_rank.append(cms_data[model][rk][pk])
    #         one_run.append(model_rank)
    #     plot_scatters_one_run(one_run, m_lst, "{}_{}".format(title, rk))

def plot_scatters(all_data, label, title):
    plt.figure()

    for idx in range(len(all_data)):
        model_data = np.array(all_data[idx])
        lb = model_data.min(axis=0)
        hb = model_data.max(axis=0)
        xs = [i+1 for i in range(len(lb))]
        # for i in range(len(xs)):
        #     plt.fill_between([xs[i]], [lb[i]], [hb[i]], facecolor=cmap(label[idx], idx/len(all_data)), alpha=0.2)
        plt.fill_between(xs, lb, hb, facecolor=cmap(label[idx], idx/len(all_data)), alpha=0.1)
        for run in model_data:
            plt.scatter(xs, run,
                        s=2, color=cmap(label[idx], idx/len(all_data)), alpha=0.2)

        avg = model_data.mean(axis=0)
        plt.scatter(xs, avg, marker='^', facecolors='none', s=35,
                    label=label[idx], edgecolors=cmap(label[idx], idx/len(all_data)), alpha=1)

    # for idx in range(len(all_data)):
    #     model_data = all_data[idx]
    #     # print(np.array(model_data).shape, idx)
    #     for run in model_data:
    #         plt.scatter([i+1 for i in range(len(run))], run,
    #                     s=2, color=cmap(label[idx], idx/len(all_data)), alpha=0.3)
    #
    #     model_data = np.array(model_data)
    #     # print(model_data.shape)
    #     avg = model_data.mean(axis=0)
    #     # print(label[idx], avg)
    #     plt.scatter([i+1 for i in range(len(avg))], avg, marker='^', facecolors='none', s=35,
    #                 label=label[idx], edgecolors=cmap(label[idx], idx/len(all_data)), alpha=1)

    plt.legend()
    plt.tight_layout()
    plt.savefig("{}.png".format(title))
    plt.close()
    plt.clf()

def plot_scatters_one_run(data, label, title):
    plt.figure()
    for idx in range(len(data)):
        d = data[idx]
        plt.scatter([i+1 for i in range(len(d))], d,
                    label=label[idx], s=5, color=cmap(label[idx], idx/len(data)))
        max_idx = np.array(d).argmax()
        plt.scatter([max_idx+1], np.array(d[max_idx]), facecolors='none', edgecolors=cmap(label[idx], idx/len(data)), s=160)

    plt.legend()
    plt.tight_layout()
    plt.savefig("{}.png".format(title))
    plt.close()
    plt.clf()

def plot_generation(te, cms, ranges, source, title, ylim=None, yscale="linear", res_scale=1,
                    outer=None, sparse_reward=None, max_len=np.inf):

    te_data = loading_average(te, source, outer=outer, sparse_reward=sparse_reward, max_len=max_len)
    te_data = average_run(te_data["true"])

    te_thrd = []
    for perc in ranges:
        te_thrd.append(percentile_avgeraged_run(te_data, perc))

    cms_data = loading_average(cms, source, outer=outer, sparse_reward=sparse_reward, max_len=max_len)
    filtered = {}
    models_rank = ranking_allruns(cms_data)
    for model in cms_data.keys():
        ranks = models_rank[model]

        filtered[model] = []
        for perc in ranges:
            target = percentile_worst(ranks, perc, te_data)
            # data = [te_data[item[1]] for item in target]
            data = [item[2] for item in target]
            filtered[model].append(data)
    plot_boxs(filtered, te_thrd, ranges, title, ylim=ylim, yscale=yscale, res_scale=res_scale)
    # plot_violins(filtered, te_thrd, ranges, title, ylim=ylim, yscale=yscale, res_scale=res_scale)

def plot_compare_top(te, cms, fqi, rand_lst, source, title,
                     cem=None,
                     ylim=None, yscale="linear", res_scale=1, outer=None, sparse_reward=None, max_len=np.inf):
    ranges = [0]
    # true env data dictionary
    te_data = loading_average(te, source, outer=outer, sparse_reward=sparse_reward, max_len=max_len)
    te_data = average_run(te_data["true"])
    # print(te_data)

    # fqi data
    # # all performance
    if fqi is not None:
        fqi_data_all = loading_average(fqi, source, outer=outer, sparse_reward=sparse_reward, max_len=max_len)["fqi"] # 30 runs in total, but different parameters
        fqi_data = []
        for rk in fqi_data_all.keys():
            for pk in fqi_data_all[rk].keys():
                fqi_data.append(fqi_data_all[rk][pk])


    if cem is not None:
        # cem_data_all = loading_average(cem, source, outer=outer, sparse_reward=sparse_reward, max_len=max_len)["cem"] # 30 runs in total, but different parameters
        cem_data_all = loading_average(cem, source, outer=outer, sparse_reward=sparse_reward, max_len=max_len)["calibration (cem)"] # 30 runs in total, but different parameters
        # cem_rank = ranking_allruns(cem_data_all)["cem"]
        cem_data = []
        for rk in cem_data_all.keys():
            for pk in cem_data_all[rk].keys():
                cem_data.append(cem_data_all[rk][pk])

    # random data list
    rand_data = performance_by_param(rand_lst, te_data)

    # top true env data performance
    te_thrd = []
    for perc in ranges:
        te_thrd.append(percentile_avgeraged_run(te_data, perc))

    if cem is not None and fqi is not None:
        filtered = {"random": [rand_data], "fqi": [fqi_data], "calibration (cem)": [cem_data]}
    elif fqi is not None:
        filtered = {"random": [rand_data], "fqi": [fqi_data]}
    elif cem is not None:
        filtered = {"random": [rand_data], "calibration (cem)": [cem_data]}
    else:
        filtered = {"random": [rand_data]}

    #filtered = {"random": [rand_data]}
    cms_data = loading_average(cms, source, outer=outer, sparse_reward=sparse_reward, max_len=max_len)
    models_rank = ranking_allruns(cms_data)
    for model in cms_data.keys():
        ranks = models_rank[model]

        filtered[model] = []
        for perc in ranges:
            target = percentile_worst(ranks, perc, te_data)
            # data = [te_data[item[1]] for item in target]
            data = [item[2] for item in target]
            filtered[model].append(data)
    # print(filtered)
    plot_violins(filtered, te_thrd, ranges, title, ylim=ylim, yscale=yscale, res_scale=res_scale)
    #plot_boxs(filtered, te_thrd, ranges, title, ylim=ylim, yscale=yscale, res_scale=res_scale)
# def plot_compare_top(te, cms, fqi, rand_lst, source, title,
#                      ylim=None, yscale="linear", res_scale=1, outer=None, sparse_reward=None, max_len=np.inf):
#     ranges = [0]
#     # true env data dictionary
#     te_data = loading_average(te, source, outer=outer, sparse_reward=sparse_reward, max_len=max_len)
#     te_data = average_run(te_data["true"])
#
#     # fqi data
#     # all performance
#     fqi_data_all = loading_average(fqi, source, outer=outer, sparse_reward=sparse_reward, max_len=max_len)["fqi"] # 30 runs in total, but different parameters
#     # fqi_rank = ranking_allruns(fqi_data_all)["fqi"]
#     fqi_data = []
#     for rk in fqi_data_all.keys():
#         for pk in fqi_data_all[rk].keys():
#             fqi_data.append(fqi_data_all[rk][pk])
#
#     # random data list
#     rand_data = performance_by_param(rand_lst, te_data)
#
#     # top true env data performance
#     te_thrd = []
#     for perc in ranges:
#         te_thrd.append(percentile_avgeraged_run(te_data, perc))
#
#     filtered = {"random": [rand_data], "fqi": [fqi_data]}
#     # filtered = {"random": [rand_data]}
#     cms_data = loading_average(cms, source, outer=outer, sparse_reward=sparse_reward, max_len=max_len)
#     models_rank = ranking_allruns(cms_data)
#     for model in cms_data.keys():
#         ranks = models_rank[model]
#         # print(model)
#         filtered[model] = []
#         for perc in ranges:
#             target = percentile_worst(ranks, perc, te_data)
#             # data = [te_data[item[1]] for item in target]
#             data = [item[2] for item in target]
#             filtered[model].append(data)
#     # print(filtered)
#     plot_violins(filtered, te_thrd, ranges, title, ylim=ylim, yscale=yscale, res_scale=res_scale)
#     #plot_boxs(filtered, te_thrd, ranges, title, ylim=ylim, yscale=yscale, res_scale=res_scale)


def performance_by_param(rand_lst, data):
    perf = []
    for i in rand_lst:
        pk = "param_{}".format(i)
        perf.append(data[pk])
    return perf

"""
input:
    filtered: {
        modelname: [10 percentile data list, 20 percentile data list, 30 percentile data list],
        ...
        }
    thrd: [10 percentile threshold, 20 percentile threshold, 30 percentile threshold]
"""
def plot_boxs(filtered, thrd, xlabel, title, ylim=None, yscale='linear', res_scale=1):

    all_models = list(filtered.keys())
    xlocations = range(len(filtered[all_models[0]]))
    width = 0.8 / len(all_models) if len(xlocations) > 2 else 0.2

    fig, ax = plt.subplots(figsize=(6.4*max(1, len(all_models)/5), 4.8))

    for i in range(len(thrd)):
        ax.plot([-(width+0.01)*len(all_models), xlocations[-1]+width], [thrd[i] * res_scale]*2, "--", color="black", linewidth=0.75)

    for idx in range(len(all_models)):
        perct = filtered[all_models[idx]]
        perct = [np.array(x) * res_scale for x in perct]
        positions_group = [x-(width+0.01)*idx for x in xlocations]

        # bp = ax.boxplot(perct, positions=positions_group, widths=width, patch_artist=True)
        bp = ax.boxplot(perct, positions=positions_group, widths=width)
        set_box_color(bp, cmap(all_models[idx], idx/len(all_models)))

        plt.plot([], c=cmap(all_models[idx], idx/len(all_models)), label=all_models[idx])

    xtcs = []
    for loc in xlocations:
        xtcs.append(loc - 0.35)
    ax.set_xticks(xtcs)
    ax.set_xticklabels(xlabel)
    
    # Acrobot plotting (Please do not delete)
    #loc, labels = plt.yticks()
    #labels = [str(-1.0 *loc[i]) for i in range(len(loc))]
    #plt.yticks(loc[1:-1], labels[1:-1])
    #plt.title('Acrobot: ' + title.split('/')[-1])
    #plt.xlabel('Top percentile', labelpad=35)
    #plt.ylabel('Steps to\nsuccess (AUC)', rotation=0, labelpad=55)
    ax.set_xlim([-(width+0.01)*len(all_models)-width, xlocations[-1]+width*len(all_models)])
    #if ylim is not None:
    #    ax.set_ylim(ylim)

    if ylim is not None and yscale != "log":
        ax.set_ylim(ylim)

    plt.yscale(yscale)

    plt.legend()
    plt.tight_layout()
    # plt.show()
    plt.savefig("{}.png".format(title))
    plt.close()
    plt.clf()
    return

def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)
    plt.setp(bp["fliers"], markeredgecolor=color)

    # for patch in bp['boxes']:
    #     patch.set_facecolor(color)

def plot_violins(filtered, thrd, xlabel, title, ylim=None, yscale="linear", res_scale=1):
    all_models = list(filtered.keys())
    xlocations = range(len(filtered[all_models[0]]))
    width = 0.8 / len(all_models) if len(xlocations) > 2 else 0.2

    fig, ax = plt.subplots(figsize=(6.4*max(1, len(all_models)/5), 4.8))

    for idx in range(len(all_models)):
        perct = filtered[all_models[idx]]
        perct = [np.array(x) * res_scale for x in perct]
        positions_group = [x-(width+0.01)*idx for x in xlocations]
        # print(perct,all_models[idx], "---")
        vp = ax.violinplot(perct, positions=positions_group, widths=width)
        set_voilin_color(vp, cmap(all_models[idx], idx/len(all_models)))

        plt.plot([], c=cmap(all_models[idx], idx/len(all_models)), label=all_models[idx])

    for i in range(len(thrd)):
        ax.plot([-(width+0.01)*len(all_models), xlocations[-1]+width], [thrd[i] * res_scale]*2, "--", color="black", linewidth=0.75)

    xtcs = []
    for loc in xlocations:
        xtcs.append(loc - 0.35)
    ax.set_xticks(xtcs)
    ax.set_xticklabels(xlabel)

    # Acrobot plotting (Please do not delete)
    #loc, labels = plt.yticks()
    #labels = [str(-1.0 *loc[i]) for i in range(len(loc))]
    #plt.yticks(loc[1:-1], labels[1:-1])
    #plt.title('Acrobot: ' + title.split('/')[-1])
    #plt.xlabel('Top percentile', labelpad=35)
    #plt.ylabel('Steps to\nsuccess (AUC)', rotation=0, labelpad=55)
    ax.set_xlim([-(width+0.01)*len(all_models)-width, xlocations[-1]+width*len(all_models)])

    if ylim is not None and yscale != "log":
        ax.set_ylim(ylim)

    plt.yscale(yscale)

    plt.legend(loc=1)
    plt.tight_layout()
    # plt.show()
    plt.savefig("{}.png".format(title))
    plt.close()
    plt.clf()
    return

def set_voilin_color(violin_parts, color):
    for pc in violin_parts['bodies']:
        pc.set_facecolor(color)
        pc.set_edgecolor(color)
    for partname in ('cbars','cmins','cmaxes'):
        vp = violin_parts[partname]
        vp.set_edgecolor(color)
        vp.set_linewidth(1)


def list2array(lst):
    temp = []
    for line in lst:
        line = line.strip("\n").strip("[").strip("]")
        num = line.split(" ")
        for i in range(len(num)):
            num[i] = float(num[i])
        temp.append(num)
    return np.array(temp)


def get_angle(cossin):
    cos, sin = cossin
    ang = []
    for i in range(len(cos)):
        c, s = cos[i], sin[i]
        acos = math.acos(c)
        asin = math.asin(s)
        if (c >= 0 and s >= 0): # acos = asin
            ang.append(acos)
        elif (c<=0 and s>=0):
            ang.append(acos)
        elif (c >= 0 and s < 0): # asin = -acos
            ang.append(asin)
        elif (c<=0 and s<0):
            ang.append(-acos)
    return ang

def plot_dataset(datasets, key, dimension, group, run, title, preprocessing=None, setlim=None):

    all_data = {}
    for gp in group:
        all_data[gp] = {}
        for case in datasets:
            all_data[gp][case] = None
    for case in datasets:
        path = datasets[case]
        temp = os.listdir(path)
        for t in temp:
            # print(temp, run, t.strip(".csv"))
            if run == t.strip(".csv"):
                trace = t

        data = list2array(pd.read_csv(os.path.join(path, trace), dtype={'info': str})[key])
        if setlim is not None:
            data = data[setlim[0]: setlim[1]]
        for gp in group:
            if gp != preprocessing:
                dims = group[gp]
                x = data[:, dims[0]]
                y = data[:, dims[1]]
                all_data[gp][case] = [x, y]

    if preprocessing is not None:
        if preprocessing == "angle":
            for case in datasets:
                th1 = get_angle(all_data["theta1"][case])
                th2 = get_angle(all_data["theta2"][case])
                all_data["angle"][case] = [th1, th2]

            all_data.pop("theta1")
            all_data.pop("theta2")
            group.pop("theta1")
            group.pop("theta2")

        else:
            raise NotImplementedError

    case_key = list(datasets.keys())
    for gp in group:
        plt.figure()
        for idx in range(len(case_key)):
            x, y = all_data[gp][case_key[idx]]
            plt.scatter(x, y, s=5, color=cmap(case_key[idx], float(idx)/len(case_key)), marker=mmap(case_key[idx], idx) , label=case_key[idx])#, alpha=0.5)

        plt.xlabel(dimension[group[gp][0]])
        plt.ylabel(dimension[group[gp][1]])
        plt.legend()
        plt.savefig("{}_{}_{}.png".format(title, gp, run))
        plt.close()
        plt.clf()

    return

def plot_termination(datasets, types, run, title):
    plt.figure()
    y_count = 0
    y_label = [[], []]

    all_data = {}
    for cidx, case in enumerate(list(datasets.keys())):
        path = datasets[case]
        temp = os.listdir(path)
        for t in temp:
            if run == t.strip(".csv"):
                trace = t
        termin_type = pd.read_csv(os.path.join(path, trace), dtype={'info': str})["info"]

        all_data[case] = {}
        for tidx, tp in enumerate(types):
            all_data[case][tp] = [] # step of termination
            for idx in range(len(termin_type)):
                if termin_type[idx] == tp:
                    all_data[case][tp].append(idx)
            plt.scatter(all_data[case][tp], [y_count] * len(all_data[case][tp]),
                        color=cmap(case, float(cidx)/len(list(datasets.keys()))), marker=mmap(tp, tidx),
                        )# label="{}-{}".format(case, tp))
            y_label[1].append("{}-{}".format(case, tp))
            y_label[0].append(y_count)
            # print(y_label[-1], y_count)
            y_count += 0.5

        y_count += 0.5

    # plt.xlim(0, 5000)
    plt.xlabel("timestep")
    plt.ylim(-0.5, y_count)
    plt.yticks(y_label[0], y_label[1], rotation=50)
    # plt.legend()
    plt.savefig("{}_{}.png".format(title, run))
    # plt.show()
    plt.close()
    plt.clf()
    return

def plot_termination_perc(datasets, types, run, title):
    # plt.figure()

    all_data = {}
    for cidx, case in enumerate(list(datasets.keys())):
        path = datasets[case]
        temp = os.listdir(path)
        for t in temp:
            if run == t.strip(".csv"):
                trace = t
        termin_type = pd.read_csv(os.path.join(path, trace), dtype={'info': str})["info"]

        all_data[case] = {}

        t_count = {}
        for tp in types:
            t_count[tp] = 0

        total_term = 0

        t_perc = {}
        for tp in types:
            t_perc[tp] = []

        xs = []

        for idx in range(len(termin_type)):
            if termin_type[idx] in types:
                t_count[termin_type[idx]] += 1
                total_term += 1

                xs.append(idx)
                for tp in types:
                    t_perc[tp].append(t_count[tp] / float(total_term))

        plt.figure()
        for tidx, tp in enumerate(types):
            plt.plot(xs, t_perc[tp],
                     # color=cmap(case, float(cidx)/len(list(datasets.keys()))), marker=mmap(tp, tidx),
                     color=cmap(tp, float(tidx)/len(types)), marker=mmap(tp, tidx),
                     label="{}-{}".format(case, tp))
        plt.xlim(0, 20000)
        plt.xlabel("timestep")
        plt.legend()
        plt.savefig("{}_{}_{}.png".format(title, case, run))
        # plt.show()
        plt.close()
        plt.clf()
    return