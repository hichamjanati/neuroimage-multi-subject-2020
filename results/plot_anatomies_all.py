"""Boxplot simulation data."""
import pandas as pd
import matplotlib as mpl

mpl.use('Agg')

from matplotlib import pyplot as plt
import seaborn.apionly as sns
from matplotlib.lines import Line2D


name = "5sources_camcan"
path = "data/"
data_a = pd.read_csv(path + "%s.csv" % name, index_col=0)
data_a = data_a[data_a.model != "mtw"]
params = {"legend.fontsize": 20,
          "axes.titlesize": 17,
          "axes.labelsize": 22,
          "xtick.labelsize": 20,
          "ytick.labelsize": 20,
          "pdf.fonttype": 42}
sns.set_context("paper", rc=params)
plt.rcParams.update(params)
x = "n_tasks"
xname = "# of subjects"
hue = "model"
ys = ["auc", "aucabs", "ot", "otabs", "mse"]
y_names = [r"AUC($\hat{\theta}, \theta^\star$)",
           r"AUC($|\hat{\theta}|, |\theta^\star|$)",
           r"EMD($\hat{\theta}, \theta^\star$ in cm)",
           r"EMD($|\hat{\theta}|, |\theta^\star|$) in cm",
           r"MSE($\hat{\theta}, \theta^\star$)"]
# ys = ["aucabs", "otabs", "mse"]
# y_names = [r"AUC($|\hat{\theta}|, |\theta^\star|$)",
#            r"EMD($|\hat{\theta}|, |\theta^\star|$) in cm",
#            r"MSE($\hat{\theta}, \theta^\star$)"]

model_names = ["MTW2", "GroupLasso", "MLL", "Dirty", "STL", "AdaSTL"]

models = [s.lower() for s in model_names]
model_names[0] = "MTW"
model_names[-2] = "Lasso"
model_names[-1] = "RW-Lasso"

data_a.model = data_a.model.replace(to_replace=models,
                                    value=model_names)

markers = ["^", "*", "+", "o", "+", "x"]
colors = ["forestgreen", "black", "indianred", "cornflowerblue", "purple",
          "gold"]
linestyles = ["--", "-"]
# linestyles = ["-"]

anatomies = ["Same X", "Different Xs"]
# anatomies = ["Different Xs"]

designs = [True, False]
# designs = [False]

legend_anatomies = [Line2D([0], [0], color='k', lw=4, ls=s,
                    label=t) for s, t in zip(linestyles, anatomies)]
legend_models = [Line2D([0], [0], color="w", marker="o",
                        markerfacecolor=color, markersize=15,
                        label=name)
                 for color, name in zip(colors, model_names)]
for y, yname in zip(ys, y_names):
    f, ax = plt.subplots(1, 1, figsize=(10, 6))
    for design, ls in zip(designs, linestyles):
        sns.pointplot(x=x, y=y, data=data_a[data_a.same_design == design],
                      hue=hue, legend=False, palette=colors, linestyles=ls,
                      hue_order=model_names)
    ax.legend(handles=legend_models, ncol=len(models) // 2,
              bbox_to_anchor=(0.9, +1.5), frameon=False)
    ax2 = ax.twinx()
    ax2.legend(handles=legend_anatomies, bbox_to_anchor=[0.9, +1.2],
               ncol=2, handlelength=2., frameon=False)
    ax.set_ylabel(yname)
    ax.grid('on')
    ax.set_xlabel(xname)
    if "auc" in y:
        ax.set_ylim([0.2, 1.])
    ax.locator_params(axis='y', nbins=10)
    fig_fname = "fig/%s-%s.pdf" % (name, y)
    f.set_tight_layout(True)
    plt.savefig(fig_fname, bbox_inches="tight")
    plt.close('all')


# f, axes = plt.subplots(2, 2, sharey="col", sharex=True, figsize=(16, 6))
# for i, (axcol, metric_id) in enumerate(zip(axes.T, ids)):
#     metric = metrics[metric_id]
#     met_name = ylabels[metric_id]
#     for ax, b, ti in zip(axcol.ravel(), [True, False], titles):
#         sns.pointplot(y=metric, x="n_tasks", hue="model",
#                       hue_order=model_names,
#                       palette=colors,
#                       markers=ms,
#                       linestyles=ls,
#                       data=df[df.same_design == b], ax=ax)
#         if b:
#             ax.set_xlabel("")
#             ax.set_title(met_name)
#         else:
#             ax.set_title("")
#             ax.set_xlabel(xlabel)
#         if i == 0:
#             ax.set_ylabel(ti)
#         else:
#             ax.set_ylabel("")
#         if i == 2:
#             ax.ticklabel_format(style='sci', axis='y', scilimits=(-3, -3))
#         ax.grid(True)
#         if i == 0 and not b:
#             ax.legend(handles=legend_models,
#                       ncol=6, bbox_to_anchor=[2.7, -0.35],
#                       labelspacing=0.15,
#                       columnspacing=0.5)
#         else:
#             ax.get_legend().remove()
