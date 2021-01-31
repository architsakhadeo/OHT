import os
import sys
cwd = os.getcwd()
sys.path.insert(0, cwd+'/../..')
from plot.box.utils_data import *
import matplotlib
import matplotlib.pyplot as plt
cmap = matplotlib.cm.get_cmap('cool')

def plot_each_run(te, cms, title, ylim=None):
    te_data = loading_pessimistic(te)
    te_rank = ranking_allruns(te_data)
    te_rank = te_rank["true"]

    cms_data = loading_pessimistic(cms)

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
    for idx in range(len(all_data)):
        model_data = all_data[idx]
        # print(np.array(model_data).shape, idx)
        for run in model_data:
            plt.scatter([i+1 for i in range(len(run))], run,
                        s=2, color=cmap(idx/len(all_data)), alpha=0.3)

        model_data = np.array(model_data)
        # print(model_data.shape)
        avg = model_data.mean(axis=0)
        # print(label[idx], avg)
        plt.scatter([i+1 for i in range(len(avg))], avg, marker='^', facecolors='none', s=35,
                    label=label[idx], edgecolors=cmap(idx/len(all_data)), alpha=1)
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
                    label=label[idx], s=5, color=cmap(idx/len(data)))
        max_idx = np.array(d).argmax()
        plt.scatter([max_idx+1], np.array(d[max_idx]), facecolors='none', edgecolors=cmap(idx/len(data)), s=160)

    plt.legend()
    plt.tight_layout()
    plt.savefig("{}.png".format(title))
    plt.close()
    plt.clf()

def plot_generation(te, cms, ranges, title, ylim=None):

    te_data = loading_pessimistic(te, 'episode')
    te_data = average_run(te_data["true"])

    te_thrd = []
    for perc in ranges:
        te_thrd.append(percentile_avgeraged_run(te_data, perc))

    cms_data = loading_pessimistic(cms, 'episode')
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
    
    '''
    if ylim is not None and ylim[0] >= 0 and ylim[1] > 0:
        res_scale = -1
    else:
        res_scale = 1
    '''
    res_scale = -1

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
    if ylim[0] >= 0 and ylim[1] > 0:
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
