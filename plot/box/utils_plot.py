import os
import sys
cwd = os.getcwd()
sys.path.insert(0, cwd+'/../..')
from plot.box.utils_data import *
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import FormatStrFormatter

# c_default = matplotlib.cm.get_cmap('cool')
# c_default = matplotlib.cm.get_cmap('hsv')

# colorblind friendly color cycle https://gist.github.com/thriveth/8560036
c_default =  ['#377eb8', '#ff7f00', '#4daf4a',
              '#f781bf', '#a65628', '#984ea3',
              '#999999', '#e41a1c', '#dede00']

c_default_Adam = ["#0077bb", "#33bbee", "#009988", "#ee7733", "#cc3311", "#ee3377", "#bbbbbb"]


c_dict = {
    "calibration": '#377eb8',
    "calibration (grid search)": '#377eb8',
    
    "fqi": '#4daf4a',
    "cem": '#f781bf',
    "Calibration (cem)": '#a65628',

    "no Adversarial": '#984ea3',
    "no Ensemble": '#999999',

    "15k": "#377eb8",
    "10k": "#a65628",
    "5k": "#dede00",
    "2k": "#4daf4a",
    "1k": "#984ea3",

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

    "random data": c_default_Adam[5],

    "knn": '#f781bf',
    "knn(laplace)": '#377eb8',
    "network": '#4daf4a',
    "network(laplace)": '#dede00',
    "network(scaled)": '#984ea3',
    "network(scaled+laplace)": '#e41a1c',

    "15k knn": c_default_Adam[1],
    "15k knn(laplace)": c_default_Adam[1],
    # "15k network(scaled+laplace)": c_default_Adam[1],
    "10k knn": c_default_Adam[3],
    "10k knn(laplace)": c_default_Adam[3],
    # "10k network(scaled+laplace)": c_default_Adam[3],
    "5k knn": c_default_Adam[0],
    "5k knn(laplace)": c_default_Adam[0],
    # "5k network(scaled+laplace)": c_default_Adam[0],
    "2.5k knn": c_default_Adam[2],
    "2.5k knn(laplace)": c_default_Adam[2],
    "1k knn": c_default_Adam[4],
    "1k knn(laplace)": c_default_Adam[4],
    "500 knn": c_default_Adam[5],
    "500 knn(laplace)": c_default_Adam[5],

    "15k network": c_default_Adam[1],
    "15k network(laplace)": c_default_Adam[1],
    "10k network": c_default_Adam[3],
    "10k network(laplace)": c_default_Adam[3],
    "5k network": c_default_Adam[0],
    "5k network(laplace)": c_default_Adam[0],
    "2.5k network": c_default_Adam[2],
    "2.5k network(laplace)": c_default_Adam[2],
    "1k network": c_default_Adam[4],
    "1k network(laplace)": c_default_Adam[4],
    "500 network": c_default_Adam[6],
    "500 network(laplace)": c_default_Adam[6],


    "Calibration (raw)": c_default_Adam[1],
    "Calibration": c_default_Adam[0],
    "NN Calibration (raw)": c_default_Adam[2],
    "NN Calibration (laplace)": c_default_Adam[3],
    "FQI": c_default_Adam[4],

    "Size = 5000": c_default_Adam[0],
    "Size = 2500": c_default_Adam[2],
    "Size = 1000": c_default_Adam[4],
    "Size = 500": c_default_Adam[6],

    "Optimal policy": c_default_Adam[0],
    # "Average policy": c_default_Adam[3],
    "Medium policy": c_default_Adam[3],
    "Bad policy": c_default_Adam[6],

    "KNN (laplace)": c_default_Adam[0],
    "network (laplace)": c_default_Adam[3],
    "Calibration-KNN": c_default_Adam[0],
    "Calibration-NN": c_default_Adam[3],
    "Random": c_default_Adam[6],
    "Random selection": c_default_Adam[6],

    "Esarsa transfer (true)": c_default_Adam[2],
    "Esarsa transfer (calibration)": c_default_Adam[6],
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
                    outer=None, sparse_reward=None, max_len=np.inf, label_ncol=10):

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
    plot_boxs(filtered, te_thrd, ranges, title, ylim=ylim, yscale=yscale, res_scale=res_scale, label_ncol=label_ncol)
    #plot_violins(filtered, te_thrd, ranges, title, ylim=ylim, yscale=yscale, res_scale=res_scale)

def plot_compare_top(te, cms, fqi, rand_lst, source, title,
                     cem=None,
                     ylim=None, yscale="linear", res_scale=1, outer=None, sparse_reward=None, max_len=np.inf,
                     ylabel="", right_ax=None, label_ncol=10, plot="box"):
    ranges = [0]
    # true env data dictionary
    te_data = loading_average(te, source, outer=outer, sparse_reward=sparse_reward, max_len=max_len)
    te_data = average_run(te_data["true"])

    # fqi data
    # # all performance
    if fqi is not None:
        fqi_data_all = loading_average(fqi, source, outer=outer, sparse_reward=sparse_reward, max_len=max_len)["FQI"] # 30 runs in total, but different parameters
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
    if rand_lst != []:
        rand_data = performance_by_param(rand_lst, te_data)

    # top true env data performance
    te_thrd = []
    for perc in ranges:
        te_thrd.append(percentile_avgeraged_run(te_data, perc))

    filtered = {}
    cms_data = loading_average(cms, source, outer=outer, sparse_reward=sparse_reward, max_len=max_len)
    models_rank = ranking_allruns(cms_data)
    for model in cms_data.keys():
        ranks = models_rank[model]
        filtered[model] = []
        for perc in ranges:
            target = percentile_worst(ranks, perc, te_data)
            data = [item[2] for item in target]
            filtered[model].append(data)


    # if cem is not None and fqi is not None and rand_lst != []:
    #     bsl = {"Random selection": [rand_data], "FQI": [fqi_data], "calibration (cem)": [cem_data]}
    # elif fqi is not None and rand_lst != []:
    #     bsl = {"Random selection": [rand_data], "FQI": [fqi_data]}
    # elif cem is not None and rand_lst != []:
    #     bsl = {"Random selection": [rand_data], "calibration (cem)": [cem_data]}
    # elif rand_lst != []:
    #     bsl = {"Random selection": [rand_data]}
    # else:
    #     bsl = {}
    bsl = {}
    if rand_lst != []:
        bsl["Random"] = [rand_data]
    if cem is not None:
        bsl["Calibration (cem)"] = [cem_data]
    if fqi is not None:
        bsl["FQI"] = [fqi_data]
    for k in bsl:
        filtered[k] = bsl[k]

    if plot == "box":
        plot_boxs(filtered, te_thrd, ranges, title, ylim=ylim, yscale=yscale, res_scale=res_scale, ylabel=ylabel, right_ax=right_ax, label_ncol=label_ncol)
    elif plot == "bar":
        plot_bars(filtered, te_thrd, ranges, title, ylim=ylim, yscale=yscale, res_scale=res_scale, ylabel=ylabel, right_ax=right_ax, label_ncol=label_ncol)

# def plot_compare_agents(te, cms, fqi, rand_lst, source, title,
#                         cem=None, ylim=None, yscale="linear", res_scale=1, outer=None, sparse_reward=None, max_len=np.inf,
#                         ylabel="", right_ax=None, label_ncol=10, plot="box"):
#     ranges = [0]
#     # true env data dictionary
#     te_data_all = loading_average(te, source, outer=outer, sparse_reward=sparse_reward, max_len=max_len)
#     te_data = {}
#     for agent in te_data_all:
#         te_data[agent] = average_run(te_data[agent])
#
#     # fqi data
#     # # all performance
#     if fqi is not None:
#         fqi_data_all = loading_average(fqi, source, outer=outer, sparse_reward=sparse_reward, max_len=max_len)["FQI"] # 30 runs in total, but different parameters
#         fqi_data = []
#         for rk in fqi_data_all.keys():
#             for pk in fqi_data_all[rk].keys():
#                 fqi_data.append(fqi_data_all[rk][pk])
#
#
#     if cem is not None:
#         # cem_data_all = loading_average(cem, source, outer=outer, sparse_reward=sparse_reward, max_len=max_len)["cem"] # 30 runs in total, but different parameters
#         cem_data_all = loading_average(cem, source, outer=outer, sparse_reward=sparse_reward, max_len=max_len)["calibration (cem)"] # 30 runs in total, but different parameters
#         # cem_rank = ranking_allruns(cem_data_all)["cem"]
#         cem_data = []
#         for rk in cem_data_all.keys():
#             for pk in cem_data_all[rk].keys():
#                 cem_data.append(cem_data_all[rk][pk])
#
#     # # random data list
#     # if rand_lst != []:
#     #     rand_data = performance_by_param(rand_lst, te_data)
#
#     # top true env data performance
#     te_thrd = []
#     for perc in ranges:
#         te_thrd.append(percentile_avgeraged_run(te_data, perc))
#
#     filtered = {}
#     cms_data = loading_average(cms, source, outer=outer, sparse_reward=sparse_reward, max_len=max_len)
#     models_rank = ranking_allruns(cms_data)
#     for model in cms_data.keys():
#         ranks = models_rank[model]
#         filtered[model] = []
#         for perc in ranges:
#             target = percentile_worst(ranks, perc, te_data)
#             data = [item[2] for item in target]
#             filtered[model].append(data)
#
#
#     # if cem is not None and fqi is not None and rand_lst != []:
#     #     bsl = {"Random selection": [rand_data], "FQI": [fqi_data], "calibration (cem)": [cem_data]}
#     # elif fqi is not None and rand_lst != []:
#     #     bsl = {"Random selection": [rand_data], "FQI": [fqi_data]}
#     # elif cem is not None and rand_lst != []:
#     #     bsl = {"Random selection": [rand_data], "calibration (cem)": [cem_data]}
#     # elif rand_lst != []:
#     #     bsl = {"Random selection": [rand_data]}
#     # else:
#     #     bsl = {}
#     bsl = {}
#     if rand_lst != []:
#         bsl["Random"] = [rand_data]
#     if cem is not None:
#         bsl["Calibration (cem)"] = [cem_data]
#     if fqi is not None:
#         bsl["FQI"] = [fqi_data]
#     for k in bsl:
#         filtered[k] = bsl[k]
#
#     if plot == "box":
#         plot_boxs(filtered, te_thrd, ranges, title, ylim=ylim, yscale=yscale, res_scale=res_scale, ylabel=ylabel, right_ax=right_ax, label_ncol=label_ncol)
#     elif plot == "bar":
#         plot_bars(filtered, te_thrd, ranges, title, ylim=ylim, yscale=yscale, res_scale=res_scale, ylabel=ylabel, right_ax=right_ax, label_ncol=label_ncol)

def performance_by_param(rand_lst, data):
    perf = []
    for i in rand_lst:
        pk = "param_{}".format(i)
        perf.append(data[pk])
    return perf

def plot_bars(filtered, thrd, xlabel, title, ylim=[], yscale='linear', res_scale=1, ylabel="", right_ax=[], label_ncol=10):
    all_models = list(filtered.keys())
    xlocations = range(len(filtered[all_models[0]]))
    width = 0.2 / len(all_models) if len(xlocations) > 2 else 0.05
    space = 0.1

    # fig, ax = plt.subplots(figsize=(6.4*max(1, len(all_models)/5), 4.8))
    fig, ax = plt.subplots(figsize=(6*max(1, len(all_models)/5), 4.8))

    for i in range(len(thrd)):
        ax.plot([xlocations[0]-width-space*2, (width+space)*len(all_models)], [thrd[i] * res_scale]*2, "--", color="black", linewidth=1.4)

    min_x = np.inf
    max_x = -np.inf
    offset = -10e6 if ylim==[] else ylim[0][0]
    info = {}
    for idx in range(len(all_models)):
        perct = filtered[all_models[idx]]
        perct = [np.array(x) * res_scale for x in perct]
        positions_group = np.array([x+(width*2)*idx for x in xlocations])
        med = [np.median(x) - offset for x in perct]
        quant25 = [np.quantile(x, 0.25) for x in perct]
        quant75 = [np.quantile(x, 0.75) for x in perct]
        ax.bar(positions_group, med, width=width, color=cmap(all_models[idx], idx/len(all_models)), bottom=offset)
        info[all_models[idx]] = {"color": cmap(all_models[idx], idx/len(all_models)), "style": "-"}

        for i in range(len(positions_group)):
            pos = positions_group[i]
            ax.plot([pos-0.01, pos+0.01], [quant25[i]]*2, color="black")
            ax.plot([pos-0.01, pos+0.01], [quant75[i]]*2, color="black")
            ax.plot([pos, pos], [quant25[i], quant75[i]], color="black")

        if positions_group.min() < min_x:
            min_x = positions_group.min()
        if positions_group.max() > max_x:
            max_x = positions_group.max()

    ax.set_xticks([])
    # ax.set_xticklabels(all_models, rotation=45, ha="right")
    if ylim != [] and yscale != "log":
        ax.set_ylim(ylim[0])

    ax.set_xlim([min_x-space, max_x+space])
    if len(thrd) > 0:
        ymin, ymax = ax.get_ylim()
        xmin, xmax = ax.get_xlim()

        true_perf_pos = thrd[0] * res_scale
        if ymax - true_perf_pos > true_perf_pos - ymin:
            true_perf_pos -= (true_perf_pos - ymin)*0.7
            # true_perf_pos = ymin
        else:
            true_perf_pos += (ymax - true_perf_pos)*0.05
        ax.text(xmin+(xmax-xmin)*0.005, true_perf_pos, "True performance", c="black", fontsize=15)

    ax.set_ylabel(ylabel, fontsize=20)
    # plt.show()
    plt.savefig("{}.pdf".format(title), dpi=300, bbox_inches='tight')
    plt.close()
    plt.clf()

    draw_label(info, title, label_ncol)
    return

"""
input:
    filtered: {
        modelname: [10 percentile data list, 20 percentile data list, 30 percentile data list],
        ...
        }
    thrd: [10 percentile threshold, 20 percentile threshold, 30 percentile threshold]
"""
def plot_boxs(filtered, thrd, xlabel, title, ylim=[], yscale='linear', res_scale=1, ylabel="", right_ax=[], label_ncol=10):

    all_models = list(filtered.keys())
    xlocations = range(len(filtered[all_models[0]]))
    width = 0.2 / len(all_models) if len(xlocations) > 2 else 0.05
    space = 0.1

    # fig, ax = plt.subplots(figsize=(6.4*max(1, len(all_models)/5), 4.8))
    fig, ax = plt.subplots(figsize=(6*max(1, len(all_models)/5), 4.8))
    rhs_axs = ax.twinx()

    info = {}
    for i in range(len(thrd)):
        ax.plot([xlocations[0]-width-space*2, (width+space)*len(all_models)], [thrd[i] * res_scale]*2, "--", color="black", linewidth=1.4)
    # if len(thrd) > 0:
    #     # plt.plot([], "--", c="black", label="true performance")
    #     info["true performance"] = {"color": "black", "style": "--"}

    vertline = None
    min_x = np.inf
    max_x = -np.inf

    for idx in range(len(all_models)):
        perct = filtered[all_models[idx]]
        perct = [np.array(x) * res_scale for x in perct]
        # positions_group = np.array([x+(width+space)*idx for x in xlocations])
        positions_group = np.array([x+(width*2)*idx for x in xlocations])
        if all_models[idx] in right_ax:
            vertline = positions_group[0] - width if vertline is None else vertline
            temp_pos = positions_group+space
            bp = rhs_axs.boxplot(perct, positions=temp_pos, widths=width, patch_artist=True, vert=True)
        else:
            temp_pos = positions_group-space
            bp = ax.boxplot(perct, positions=temp_pos, widths=width, patch_artist=True, vert=True)
        if temp_pos.min() < min_x:
            min_x = temp_pos.min()
        if temp_pos.max() > max_x:
            max_x = temp_pos.max()
        set_box_color(bp, cmap(all_models[idx], idx/len(all_models)))

        # plt.plot([], c=cmap(all_models[idx], idx/len(all_models)), label=all_models[idx])
        info[all_models[idx]] = {"color": cmap(all_models[idx], idx/len(all_models)), "style": "-"}

    if vertline:
        plt.axvline(x=vertline, color='brown', linestyle=':', linewidth=1.4)

    xtcs = []
    ax.set_xticks(xtcs)
    # ax.set_xticklabels(xlabel)

    # Acrobot plotting (Please do not delete)
    #loc, labels = plt.yticks()
    #labels = [str(-1.0 *loc[i]) for i in range(len(loc))]
    #plt.yticks(loc[1:-1], labels[1:-1])
    #plt.title('Acrobot: ' + title.split('/')[-1])
    #plt.xlabel('Top percentile', labelpad=35)
    #plt.ylabel('Steps to\nsuccess (AUC)', rotation=0, labelpad=55)

    # ax.set_xlim([xlocations[0]-width*2-space, (width+space)*len(all_models)-width])
    # rhs_axs.set_xlim([xlocations[0]-width*2-space, (width+space)*len(all_models)-width])
    ax.set_xlim([min_x-space, max_x+space])
    rhs_axs.set_xlim([min_x-space, max_x+space])

    if ylim != [] and yscale != "log":
        ax.set_ylim(ylim[0])
        # ax.set_ybound(lower=ylim[0][0], upper=ylim[0][1])
    if not vertline:
        rhs_axs.set_visible(False)
    elif vertline and len(thrd)>0:
        align_yaxis(ax, thrd[0] * res_scale, rhs_axs, thrd[0] * res_scale)

    ax.set_yscale(yscale)
    rhs_axs.set_yscale(yscale)
    if yscale!="log":
        ymin, ymax = ax.get_ylim()
        ytcs = []
        ytcs.append(ymin)
        step = (ymax - ymin) / 3
        for i in range(len(thrd)):
            ytcs.append(thrd[i] * res_scale)
        for j in np.arange(ymin+step, ymax+step, step):
            ytcs.append(j)

        ax.set_yticks(ytcs)
        ax.set_yticklabels(ytcs)

        if len(thrd) > 0:
            true_perf_pos = thrd[0] * res_scale
            if ymax - true_perf_pos > true_perf_pos - ymin:
                true_perf_pos -= (true_perf_pos - ymin)*0.7
                # true_perf_pos = ymin
            else:
                true_perf_pos += (ymax - true_perf_pos)*0.05
            ax.text(min_x-space, true_perf_pos, "True performance", c="black", fontsize=15)

        ymin, ymax = rhs_axs.get_ylim()
        ytcs = []
        ytcs.append(ymin)
        step = (ymax - ymin) / 3
        for i in range(len(thrd)):
            ytcs.append(thrd[i] * res_scale)
        for j in np.arange(ymin+step, ymax+step, step):
            ytcs.append(j)
        rhs_axs.set_yticks(ytcs)
        # rhs_axs.set_yticklabels(ytcs)

    if yscale == "log":
        ylabel += " (log)"
    ax.set_ylabel(ylabel, fontsize=20)

    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    plt.setp(ax.get_yticklabels(), fontsize=15)
    plt.setp(rhs_axs.get_yticklabels(), fontsize=15)

    # plt.legend()
    plt.tight_layout()
    # plt.show()
    plt.savefig("{}.pdf".format(title))
    plt.close()
    plt.clf()

    draw_label(info, title, label_ncol)
    return

def draw_label(info, save_path, ncol):
    plt.figure(figsize=(0.05, 2.5))
    for label in info:
        plt.plot([], color=info[label]["color"], linestyle=info[label]["style"], label=label)
    plt.axis('off')
    plt.legend(ncol=ncol)
    plt.savefig("{}_labels.pdf".format(save_path), dpi=300, bbox_inches='tight')
    # plt.savefig("plot/img/{}.png".format(save_path), dpi=300, bbox_inches='tight')
    # plt.show()
    plt.close()
    plt.clf()

# https://stackoverflow.com/questions/10481990/matplotlib-axis-with-two-scales-shared-origin
def align_yaxis(ax1, v1, ax2, v2):
    """adjust ax2 ylimit so that v2 in ax2 is aligned to v1 in ax1"""
    _, y1 = ax1.transData.transform((0, v1))
    _, y2 = ax2.transData.transform((0, v2))
    inv = ax2.transData.inverted()
    _, dy = inv.transform((0, 0)) - inv.transform((0, y1-y2))
    miny, maxy = ax2.get_ylim()
    ax2.set_ylim(miny+dy, maxy+dy)

def set_box_color(bp, color):
    # plt.setp(bp['boxes'], color=color)
    # plt.setp(bp['whiskers'], color=color)
    # plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color="black", linewidth=2)
    plt.setp(bp["fliers"], markeredgecolor=color)

    for patch in bp['boxes']:
        patch.set_facecolor(color)
        patch.set(facecolor=color)

def plot_violins(filtered, thrd, xlabel, title, ylim=None, yscale="linear", res_scale=1):
    all_models = list(filtered.keys())
    xlocations = range(len(filtered[all_models[0]]))
    width = 0.8 / len(all_models) if len(xlocations) > 2 else 0.2

    # fig, ax = plt.subplots(figsize=(6.4*max(1, len(all_models)/5), 4.8))
    fig, ax = plt.subplots(figsize=(6.4*max(1, len(all_models)/5), 2.5))

    for idx in range(len(all_models)):
        perct = filtered[all_models[idx]]
        perct = [np.array(x) * res_scale for x in perct]
        positions_group = [x-(width+0.01)*idx for x in xlocations]
        # print(perct,all_models[idx], positions_group, "---")
        vp = ax.violinplot(perct, positions=positions_group, widths=width)
        set_voilin_color(vp, cmap(all_models[idx], idx/len(all_models)))

        plt.plot([], c=cmap(all_models[idx], idx/len(all_models)), label=all_models[idx])

    for i in range(len(thrd)):
        ax.plot([-(width+0.01)*len(all_models), xlocations[-1]+width], [thrd[i] * res_scale]*2, "--", color="black", linewidth=0.75)
        #ax.plot([-(width+0.01)*len(all_models), xlocations[-1]+width], [-26.11002104240409 * res_scale]*2, color="black", linewidth=2.0)
        #ax.plot([-(width+0.01)*len(all_models), xlocations[-1]+width], [-139.3 * res_scale]*2, color="black", linewidth=2.0)

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

    plt.legend(ncol=3)
    fig.tight_layout(pad=1.0)
    fontP = FontProperties()
    # fontP.set_size('xx-small')
    plt.legend(bbox_to_anchor=(0.5, 1.18), loc='center', prop=fontP)
    # plt.show()
    # plt.savefig("{}.png".format(title))
    plt.savefig("{}.pdf".format(title), dpi=300, bbox_inches='tight')
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

def plot_learning_perform(paths, source, title, ylim=[], yscale="linear", outer=None, sparse_reward=None, max_len=np.inf, res_scale=1,
                          ylabel="", right_ax=None, label_ncol=10,
                          fqi=None, true_perf=None):
    te_thrd = []
    if true_perf:
        # true env data dictionary
        te_data = loading_average(true_perf, source, outer=outer, sparse_reward=sparse_reward, max_len=max_len)
        te_data = average_run(te_data["true"])

        # top true env data performance
        te_thrd.append(percentile_avgeraged_run(te_data, 0))

    data = loading_average(paths, source, outer=outer, sparse_reward=sparse_reward, max_len=max_len)
    models_rank = ranking_allruns(data)
    filtered = {}
    for model in data.keys():
        ranks = models_rank[model]
        filtered[model] = [[]]
        for run in ranks:
            filtered[model][0].append(ranks[run][0][1])

    bsl = {}
    if fqi is not None:
        fqi_data_all = loading_average(fqi, source, outer=outer, sparse_reward=sparse_reward, max_len=max_len)["FQI"] # 30 runs in total, but different parameters
        fqi_data = []
        for rk in fqi_data_all.keys():
            for pk in fqi_data_all[rk].keys():
                fqi_data.append(fqi_data_all[rk][pk])
        bsl["FQI"] = [fqi_data]
    for k in bsl:
        filtered[k] = bsl[k]

    plot_boxs(filtered, te_thrd, "", title, ylim=ylim, yscale=yscale, res_scale=res_scale, ylabel=ylabel, right_ax=right_ax, label_ncol=label_ncol)

def plot_param_sweep(paths, source, title, ylim=None, yscale="linear", outer=None, sparse_reward=None, max_len=np.inf, res_scale=1):
    assert len(list(paths.keys())) == 1
    # data = loading_average(paths, source, outer=outer, sparse_reward=sparse_reward, max_len=max_len)
    # models_rank = ranking_allruns(data)
    # filtered = {}
    # for model in data.keys():
    #     ranks = models_rank[model]
    #     filtered[model] = []
    #     for e in ranks[list(ranks.keys())[0]]:
    #         filtered[model].append([])
    #     for run in ranks:
    #         for e in ranks[run]:
    #             param = int(e[0].split("_")[1])
    #             filtered[model][param].append(ranks[run][param][1])
    # plot_violins(filtered, [], "", title, ylim=ylim, yscale=yscale, res_scale=res_scale)

    avg_data = loading_average(paths, source, outer=outer, sparse_reward=sparse_reward, max_len=max_len)
    avg_data = average_run(avg_data[list(paths.keys())[0]])
    ranked = sorted(avg_data.items(), key=lambda item: item[1])
    print("ranked")
    print(ranked)
