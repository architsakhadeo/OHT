import numpy as np
import matplotlib.pyplot as plt

# --- Your data, e.g. results per algorithm:
data1 = [5,5,4,3,3,5]
data2 = [6,6,4,6,8,5]
data3 = [7,8,4,5,8,2]
data4 = [6,9,3,6,8,4]

# --- Combining your data:
data_group1 = [data1, data2]
data_group2 = [data3, data4]

# --- Labels for your data:
labels_list = ['a','b']
xlocations  = range(len(data_group1))
width       = 0.3
symbol      = 'r+'
ymin        = 0
ymax        = 10

ax = plt.gca()
ax.set_ylim(ymin,ymax)
ax.set_xticklabels( labels_list, rotation=0 )
ax.grid(True, linestyle='dotted')
ax.set_axisbelow(True)
ax.set_xticks(xlocations)
plt.xlabel('X axis label')
plt.ylabel('Y axis label')
plt.title('title')

# --- Offset the positions per group:
positions_group1 = [x-(width+0.01) for x in xlocations]
positions_group2 = xlocations

plt.boxplot(data_group1,
            sym=symbol,
            labels=['']*len(labels_list),
            positions=positions_group1,
            widths=width,
            #           notch=False,
            #           vert=True,
            #           whis=1.5,
            #           bootstrap=None,
            #           usermedians=None,
            #           conf_intervals=None,
            #           patch_artist=False,
            )

plt.boxplot(data_group2,
            labels=labels_list,
            sym=symbol,
            positions=positions_group2,
            widths=width,
            #           notch=False,
            #           vert=True,
            #           whis=1.5,
            #           bootstrap=None,
            #           usermedians=None,
            #           conf_intervals=None,
            #           patch_artist=False,
            )

# plt.savefig('boxplot_grouped.png')
# plt.savefig('boxplot_grouped.pdf')    # when publishing, use high quality PDFs
plt.show()                   # uncomment to show the plot.
