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
df = df[(~df.same_design)]


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
f, axes = plt.subplots(1, 3, sharex=True, figsize=(17, 4))

for i, (ax, metric_id) in enumerate(zip(axes, ids)):
    metric = metrics[metric_id]
    met_name = ylabels[metric_id]
    dfb = df.copy()
    models_, model_names, colors, ms, ls, _ = get_plot_params(dfb)
    dfb.model = dfb.model.replace(to_replace=models_, value=model_names)
    sns.pointplot(y=metric, x="n_tasks", hue="model",
                  hue_order=model_names,
                  palette=colors,
                  markers=ms,
                  linestyles=ls,
                  data=dfb, ax=ax)
    ax.set_ylabel(met_name)
    ax.set_title("")
    ax.set_xlabel(xlabel)
    ax.grid(True)
    if i == 1:
        ax.legend(handles=legend_models, frameon=False,
                  ncol=6, bbox_to_anchor=[1.9, 1.25],
                  labelspacing=2., handlelength=2.5,
                  columnspacing=1.5)
    else:
        ax.get_legend().remove()

    if i < 2:
        formatter = ticker.MultipleLocator(tick_bases[metric_id])
        ax.yaxis.set_major_locator(formatter)
        ax.yaxis.set_major_formatter(ticker.ScalarFormatter())


# f.set_tight_layout(True)
# plt.show()
plt.savefig("fig/%s-scores.pdf" % name, bbox_inches="tight")
