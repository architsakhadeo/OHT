import os
import sys
cwd = os.getcwd()
sys.path.insert(0, cwd+'/../..')
from plot.box.utils_data import *
import matplotlib
import matplotlib.pyplot as plt
cmap = matplotlib.cm.get_cmap('cool')


def plot_generation(te, cms, ranges, title, ylim=None):

    te_data = loading_pessimistic(te)
    te_data = average_run(te_data["true"])

    te_thrd = []
    for perc in ranges:
        te_thrd.append(percentile_avgeraged_run(te_data, perc))

    cms_data = loading_pessimistic(cms)
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
    plot_boxs(filtered, te_thrd, ranges, title, ylim=ylim)
    # plot_violins(filtered, te_thrd, ranges, title, ylim=ylim)


"""
input:
    filtered: {
        modelname: [10 percentile data list, 20 percentile data list, 30 percentile data list],
        ...
        }
    thrd: [10 percentile threshold, 20 percentile threshold, 30 percentile threshold]
"""
def plot_boxs(filtered, thrd, xlabel, title, ylim=None, yscale='linear'):

    all_models = list(filtered.keys())
    xlocations = range(len(filtered[all_models[0]]))
    width = 0.8 / len(all_models) if len(xlocations) > 2 else 0.2

    fig, ax = plt.subplots(figsize=(6.4*max(1, len(all_models)/5), 4.8))
    
    if yscale=="log" and ylim[0] < 0 and ylim[1] <= 0:
        res_scale = -1
    elif ylim[0] >= 0 and ylim[1] > 0:
        res_scale = -1
    else:
        res_scale = 1

    for idx in range(len(all_models)):
        perct = filtered[all_models[idx]]
        perct = [np.array(x) * res_scale for x in perct]
        positions_group = [x-(width+0.01)*idx for x in xlocations]

        # bp = ax.boxplot(perct, positions=positions_group, widths=width, patch_artist=True)
        bp = ax.boxplot(perct, positions=positions_group, widths=width)
        set_box_color(bp, cmap(idx/len(all_models)))

        plt.plot([], c=cmap(idx/len(all_models)), label=all_models[idx])

    for i in range(len(thrd)):
        ax.plot([-(width+0.01)*len(all_models), xlocations[-1]+width], [thrd[i] * res_scale]*2, "--", color="red", linewidth=0.75)

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
    return

def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)
    plt.setp(bp["fliers"], markeredgecolor=color)

    # for patch in bp['boxes']:
    #     patch.set_facecolor(color)

def plot_violins(filtered, thrd, xlabel, title, ylim=None, yscale="linear"):
    all_models = list(filtered.keys())
    xlocations = range(len(filtered[all_models[0]]))
    width = 0.8 / len(all_models) if len(xlocations) > 2 else 0.2

    fig, ax = plt.subplots(figsize=(6.4*max(1, len(all_models)/5), 4.8))
    if yscale=="log" and ylim[0] < 0 and ylim[1] <= 0:
        res_scale = -1
    else:
        res_scale = 1
    for idx in range(len(all_models)):
        perct = filtered[all_models[idx]]
        perct = [np.array(x) * res_scale for x in perct]
        positions_group = [x-(width+0.01)*idx for x in xlocations]
        vp = ax.violinplot(perct, positions=positions_group, widths=width)
        set_voilin_color(vp, cmap(idx/len(all_models)))

        plt.plot([], c=cmap(idx/len(all_models)), label=all_models[idx])

    for i in range(len(thrd)):
        print(thrd)
        ax.plot([-(width+0.01)*len(all_models), xlocations[-1]+width], [thrd[i] * res_scale]*2, "--", color="red", linewidth=0.75)

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

    plt.legend()
    plt.tight_layout()
    # plt.show()
    plt.savefig("{}.png".format(title))
    return

def set_voilin_color(violin_parts, color):
    for pc in violin_parts['bodies']:
        pc.set_facecolor(color)
        pc.set_edgecolor(color)
    for partname in ('cbars','cmins','cmaxes'):
        vp = violin_parts[partname]
        vp.set_edgecolor(color)
        vp.set_linewidth(1)