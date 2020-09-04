import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter


rc = {"legend.fontsize": 11,
      "axes.titlesize": 15,
      "axes.labelsize": 11,
      "xtick.labelsize": 10,
      "ytick.labelsize": 10,
      "pdf.fonttype": 42}
plt.rcParams.update(rc)

dataset_names = ["Cam-CAN", "DS117"]
datasets = ["camcan", "ds117"]
ls = ["-", ":"]
dataset_colors = ["k", "k"]
lambdas = [0.3, 0.3]
tasks = ["auditory", "visual"]

model_names = [r"MWE$_{0.5}$", r"MWE$_1$"]

colors = ["forestgreen", "cornflowerblue"]

ms = ["*", "o"]

legend_models = [Line2D([0], [0], color=color, marker=m,
                        markerfacecolor=color, markersize=5,
                        linewidth=2,
                        linestyle="-",
                        label=name)
                 for color, name, m in zip(colors, model_names, ms)]

legend_datasets = [Line2D([0], [0], color=c,
                          markerfacecolor=c, markersize=1,
                          linewidth=2,
                          linestyle=ll,
                          label=name)
                   for c, name, ll in zip(dataset_colors, dataset_names, ls)]
legend = legend_models + legend_datasets

f1, ax1 = plt.subplots(1, 1, figsize=(5.5, 3))
f2, ax2 = plt.subplots(1, 1, figsize=(4, 3))
f3, ax3 = plt.subplots(1, 1, figsize=(4, 3))

alphas = [0., 0.5, 1., 2., 3., 4., 5., 7., 10.]
ylabels = ["Number of active sources", "Max amplitude", "R2"][:1]
for dataset, ls, task, lambda_ in zip(datasets, ls, tasks, lambdas):
    df = pd.read_csv("stats/stats_%s-%s.csv" % (dataset, task), index_col=0)
    df = df[df.beta == lambda_]
    df = df[df.alpha.isin(alphas)]
    df = df[df.model.str.contains("mwe")]
    n_tasks = 32
    if dataset == "ds117":
        n_tasks = 16

    models = ["re-mwe-S%s" % n_tasks, "mwe-S%s" % n_tasks]

    df.model = df.model.replace(models, model_names)
    cols = ["model", "subject", "alpha", "beta"]
    df_ns = df.groupby(cols).ns.sum().reset_index()
    df_value = df.groupby(cols).source_max.max().reset_index()
    df_gof = df.groupby(cols).gof.max().reset_index()

    for data, y, ax, ylabel in zip([df_ns, df_value, df_gof],
                                   ["ns", "source_max", "gof"],
                                   [ax1, ax2, ax3],
                                   ylabels):
        ax = sns.pointplot(y=y, x="alpha", hue="model",
                           hue_order=model_names,
                           palette=colors, linestyles=ls, markers=ms,
                           data=data, ax=ax, ci=None)
        ax.get_legend().set_visible(False)
        ax.grid(True)
        ax.set_ylabel(ylabel)
        ax.set_xlabel(r"OT regularization $\mu$")
        ax.legend(handles=legend, frameon=False, loc=2,
                  ncol=4, bbox_to_anchor=[-0.06, 1.15],
                  labelspacing=1,
                  columnspacing=1.)
        #
        # ax.set_xticks(alphas)
        # ax.set_xticklabels([str(a) for a in alphas])

for ax in [ax1, ax2]:
    ax.set_yscale("symlog")
    formatter = ScalarFormatter()
    formatter.set_scientific(False)
    ax.yaxis.set_major_formatter(formatter)
ax1.set_ylim([1, 20000])
ax1.set_yticks([1, 5, 10, 50, 500, 1000, 2500, 5000])
ax2.set_yticks([1, 5, 10, 20, 30])
# plt.show()
#
f1.savefig("fig/model-selection.pdf", bbox_inches="tight")
# f2.savefig("fig/amplitude.pdf", bbox_inches="tight")
# f3.savefig("fig/gof.pdf", bbox_inches="tight")
