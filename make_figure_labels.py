import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.lines import Line2D

dataset = "ds117"
params = {"legend.fontsize": 15,
          "axes.titlesize": 15,
          "axes.labelsize": 14,
          "xtick.labelsize": 12,
          "ytick.labelsize": 12}
sns.set_context("paper", rc=params)
plt.rcParams.update(params)

path = "brain_plots/%s/" % dataset
dmeg = pd.read_csv("results/data/real_data_stats.csv", index_col=0)
dfmri = pd.read_csv("results/data/fmri_st.csv", index_col=0)
df = pd.concat((dmeg, dfmri))
hemis = ["lh", "rh"]
names = ["ffg"]
colors = ["indianred", "cornflowerblue", "forestgreen"]
titles = ["%s hemisphere" % s for s in ["Left", "Right"]]
model_names = ["Lasso", "fMRI", "MWE"]
ms = ["+", "+", "*"]
ls = ["--", ":", "-"]
legend_models = [Line2D([0], [0], color=color, marker=m,
                        markerfacecolor=color, markersize=13,
                        linestyle=ll, linewidth=3,
                        label=name)
                 for color, name, m, ll in zip(colors, model_names, ms, ls)]

betas = np.sort(df.beta.unique())
betas = [0.1, 0.15, 0.2, 0.3]
n_b = len(betas)
width = 6
height = n_b * width // 4
for name in names:
    f, axes = plt.subplots(n_b, 2, sharey=True,
                           figsize=(2 * width, height))
    yticks = [0.25, 0.5, 0.75, 1.]
    for r, (beta, axrow) in enumerate(zip(betas, axes)):
        dtmp = df[df.beta == beta]
        alphas = np.sort(dtmp.alpha.unique())
        for i, (hemi, ax, title) in enumerate(zip(hemis, axrow, titles)):
            lasso_v = dtmp[(dtmp.hemi == hemi) & (dtmp.model == "lasso")]
            lasso_v = lasso_v[[name]].mean().values
            fmri_v = dtmp[(dtmp.hemi == hemi) & (dtmp.model == "fmri")]
            fmri_v = fmri_v[[name]].mean().values
            sns.pointplot(y=name, x="alpha",
                          data=dtmp[dtmp.hemi == hemi],
                          ax=ax, color=colors[2], ci=0,
                          markers=ms[2], linestyles=ls[2])
            ax.plot(alphas, len(alphas) * lasso_v, color=colors[0],
                    marker=ms[0], linewidth=3, ls=ls[0])
            ax.plot(alphas, len(alphas) * fmri_v, color=colors[1],
                    marker=ms[1], linewidth=3, ls=ls[1])
            ax.set_ylim([0.2, 1.05])
            ax.set_yticks(yticks)

            if i == 0:
                bf = int(100 * beta)
                ax.set_ylabel(r"$\frac{\lambda}{\lambda_{\max}} = %d$%%" % bf)
            else:
                ax.set_ylabel("")
            if r == 0:
                ax.set_title(title)

            if r == len(betas) - 1:
                ax.set_xlabel(r"OT regularization $\mu$")
            else:
                ax.set_xlabel("")
                ax.set_xticklabels([])
            ax.grid(True)
            if r == len(betas) - 1 and hemi == "lh":
                ax.legend(handles=legend_models, ncol=3,
                          bbox_to_anchor=[1.65, -0.46])
    # f.set_tight_layout(True)
    plt.savefig("results/fig/%s.pdf" % name, bbox_inches="tight")
