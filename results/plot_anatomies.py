import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn.apionly as sns
from matplotlib.lines import Line2D
from matplotlib import ticker


dataset = "camcan"
name = "ico4_%s" % dataset
path = "data/"
df = pd.read_csv(path + "%s.csv" % name, index_col=0)
params = {"legend.fontsize": 16,
          "axes.titlesize": 16,
          "axes.labelsize": 16,
          "xtick.labelsize": 14,
          "ytick.labelsize": 14,
          "pdf.fonttype": 42}
plt.rcParams.update(params)
sns.set_context("paper", rc=params)

xlabel = "# subjects"
metrics = ["auc", "aucabs", "ot", "otabs", "mse"]
tick_bases = [0.1, 0.1, 0.2, 0.1, 0.5]
ids = [1, 3, -1]

ylabels = ["AUC", "AUC(|.|)", "EMD in cm",
           "EMD(|.|) in cm", "MSE"]
titles = ["Same Leadfield", "Different Leadfields"]
all_models = ["re-mwe", "mwe", "dirty", "grouplasso", "lasso", "re-lasso"]
all_model_names = [r"MWE$_{0.5}$", r"MWE$_1$", "Dirty", "GL",
                   r"Lasso $\ell_1$", r"Re-Lasso $\ell_{0.5}$"
                   ]
colors_all = ["forestgreen", "black", "indianred", "gold", "cornflowerblue",
              "purple", "cyan", "magenta", "yellow"]
ms_all = ["*", "o", "s", "^", "P", "D", "*", "o", "^"]
ls_all = ["-", "--", "-.", ":", "-", "-", "--", ":", "-."]

df = df[(df.model != "mll")]


def get_plot_params(df):
    models = [x.lower() for x in list(np.unique(df.model.values))]
    if len(models) < len(all_models):
        idx_models = [all_models.index(model) for model in models]
    else:
        idx_models = np.arange(len(models))
        models = list(map(str, np.array(all_models)[idx_models]))
        model_names = list(map(str, np.array(all_model_names)[idx_models]))
        colors = np.array(colors_all)[idx_models]
        ms = np.array(ms_all)[idx_models]
        ls = np.array(ls_all)[idx_models]
    return models, model_names, colors, ms, ls, idx_models

models, model_names, colors, ms, ls, idx_models = get_plot_params(df)
legend_models = [Line2D([0], [0], color=color, marker=m,
                        markerfacecolor=color, markersize=13,
                        linewidth=3,
                        linestyle=ll,
                        label=name)
                 for color, name, m, ll in zip(colors, model_names, ms, ls)]
f, axes = plt.subplots(3, 2, sharey="row", sharex=True, figsize=(10, 7))


for i, (axrow, metric_id) in enumerate(zip(axes, ids)):
    metric = metrics[metric_id]
    met_name = ylabels[metric_id]
    for ax, b, ti in zip(axrow.ravel(), [True, False], titles):
        dfb = df[df.same_design == b]
        models_, model_names, colors, ms, ls, _ = get_plot_params(dfb)
        dfb.model = dfb.model.replace(to_replace=models_, value=model_names)
        sns.pointplot(y=metric, x="n_tasks", hue="model",
                      hue_order=model_names,
                      palette=colors,
                      markers=ms,
                      linestyles=ls,
                      data=dfb, ax=ax)
        if b:
            ax.set_ylabel(met_name)
        else:
            ax.set_ylabel("")
        if i == 0:
            ax.set_title(ti)
        else:
            ax.set_title("")
        if i == 2:
            ax.set_xlabel(xlabel)
        else:
            ax.set_xlabel("")
        ax.grid(True)
        if i == 0 and not b:
            ax.legend(handles=legend_models, frameon=False,
                      ncol=1, bbox_to_anchor=[1.05, 0.5],
                      labelspacing=1.7,
                      columnspacing=0.5)
        else:
            ax.get_legend().remove()

        if i < 2:
            formatter = ticker.MultipleLocator(tick_bases[metric_id])
            ax.yaxis.set_major_locator(formatter)
            ax.yaxis.set_major_formatter(ticker.ScalarFormatter())


# f.set_tight_layout(True)
# plt.show()
plt.savefig("fig/%s-anatomies.pdf" % name, bbox_inches="tight")

# for metric, yl in zip(metrics, ylabels):
#     f, axes = plt.subplots(1, 2, sharey=True, figsize=(12, 4))
#     for ax, b, ti in zip(axes.ravel(), [True, False], titles):
#         sns.pointplot(y=metric, x="n_tasks", hue="model",
#                       hue_order=model_names,
#                       palette=colors,
#                       data=df[df.same_design == b], ax=ax)
#         if "auc" in metric:
#             ax.set_ylim([0.2, 1])
#         if "mse" in metric:
#             ax.set_title(ti)
#         if b:
#             ax.set_ylabel(yl)
#         else:
#             ax.set_ylabel("")
#         if "ot" in metric:
#             ax.set_xlabel(xlabel)
#         else:
#             ax.set_xlabel("")
#         ax.grid('on')
#         if not b and "mse" in metric:
#             ax.legend(ncol=1, bbox_to_anchor=[1., 1.05], labelspacing=1.,
#                       columnspacing=None)
#         else:
#             ax.get_legend().remove()
#     f.set_tight_layout(True)
#     plt.savefig("fig/ipmi/anatomies-%s.pdf" % metric, bbox_inches="tight")
