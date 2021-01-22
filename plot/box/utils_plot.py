import matplotlib
import matplotlib.pyplot as plt
cmap = matplotlib.cm.get_cmap('cool')


"""
input:
    filtered: {
        modelname: [10 percentile data list, 20 percentile data list, 30 percentile data list],
        ...
        }
    thrd: [10 percentile threshold, 20 percentile threshold, 30 percentile threshold]
"""
def plot_boxs(filtered, thrd, xlabel):
    fig, ax = plt.subplots()

    all_models = list(filtered.keys())
    xlocations  = range(len(filtered[all_models[0]]))
    width = 0.2
    for idx in range(len(all_models)):
        perct = filtered[all_models[idx]]
        positions_group = [x-(width+0.01)*idx for x in xlocations]

        bp = ax.boxplot(perct, positions=positions_group, widths = width)
        set_box_color(bp, cmap(idx/len(all_models)))

        plt.plot([], c=cmap(idx/len(all_models)), label=all_models[idx])

    for i in range(len(thrd)):
        ax.plot([-(width+0.01)*len(all_models), xlocations[-1]+width*3], [thrd[i]]*2, "--", color="red")

    ax.set_xticklabels(xlabel)
    ax.set_xlim([-(width+0.01)*len(all_models)-width, xlocations[-1]+width*len(all_models)])
    ax.set_ylim([-0.02, 0])
    plt.legend()
    plt.show()
    return

def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)