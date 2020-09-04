import os.path as op

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from config import get_subjects_list
from plot_averages import get_argmin
import utils
from matplotlib.patches import Patch

import seaborn as sns


dataset = "ds117"
task = "visual"
subjects = get_subjects_list(dataset)
n_tasks = 16
n_sources = 6
min_amplitude = 0.1

params = {"legend.fontsize": 12,
          "axes.titlesize": 12,
          "axes.labelsize": 12,
          "xtick.labelsize": 10,
          "ytick.labelsize": 10}
sns.set_context("paper", rc=params)
plt.rcParams.update(params)

dmeg = pd.read_csv("stats/stats_%s-%s.csv" % (dataset, task), index_col=0)
dfmri = pd.read_csv("stats/stats_fmri.csv", index_col=0)
data_path = "data/ds117-visual/"
save_path = "data/ds117-visual/glass-brains"
dgroup = dmeg.groupby(["beta", "subject", "model", "alpha"])
dns = dgroup.ns.sum().reset_index()
dvalue = dgroup.source_max.max().reset_index()

models = ["re-mwe-S%s" % n_tasks, "lasso", "adalasso", "gala"]
alphas = [0., 0.25, 0.5, 0.75, 1., 2., 3.]
dmodels = []
for model in models:
    if "mwe" in model:
        dmodel = dmeg[dmeg.model == model].copy()
        dmodel = dmodel[dmodel.alpha.isin(alphas)]
        no_sources = dmodel.groupby(["alpha", "beta"]).ns.mean()
        no_sources = no_sources[no_sources == 0]
        alpha_betas_ns = no_sources[["aloha", "beta"]].values
        low_sources = dvalue[dvalue.model == model]
        low_sources = low_sources.set_index(["alpha", "beta"])
        amplitude_condition = low_sources.source_max <= min_amplitude
        low_sources = low_sources[amplitude_condition].reset_index()
        alpha_betas_v = low_sources[["alpha", "beta"]].values
        filters_ns = [(dmodel.alpha == ab[0]) & (dmodel.beta == ab[1])
                      for ab in alpha_betas_ns]
        filters_v = [(dmodel.alpha == ab[0]) & (dmodel.beta == ab[1])
                     for ab in alpha_betas_v]
        dmodel = dmodel[~sum(filters_v + filters_ns).astype(bool)]
        # dmodel = dmodel[dmodel.alpha > 0.]
        dmodel = dmodel.groupby(["alpha", "beta", "subject"]).ns.sum()
        dmodel = dmodel.reset_index().groupby(["alpha", "beta"]).ns.mean()
        idx = utils.get_argmin(dmodel.values, n_sources)
        alpha, beta = dmodel.index[idx]
        print(model, alpha, beta)
        # alpha = 0.5, beta = 0.2
        dmodel = dmeg[(dmeg.alpha == alpha) &
                      (dmeg.beta == beta) & (dmeg.model == model)]
        dmodels.append(dmodel)
    elif "gala" in model:
        dmodels.append(dmeg[dmeg.model == model])
    else:
        dmodel = dns[dns.model == model].copy()
        dmodel = dmodel.pivot(index="subject", columns="beta", values="ns")
        assert len(set(subjects) - set(list(dmodel.index))) == 0
        betas = dict(dmodel.apply(get_argmin, axis=1))
        # print(">>>>>> \n \n ", dmodel.ns, betas)
        # print("........\n")
        filters = [(dmeg.subject == s) & (dmeg.beta == b)
                   for s, b in betas.items()]
        dmodel = dmeg[sum(filters).astype(bool) & (dmeg.model == model)]
        dmodels.append(dmodel)
df = pd.concat(dmodels + [dfmri])

for i, subject in enumerate(subjects):
    for model in models:
        df_ = df[(df.model == model) & (df.subject == subject)]
        if "mwe" in model:
            alpha, beta = df_.alpha.data[0], df_.beta.data[0]
            beta = int(100 * beta)
            alpha = str(alpha)
            fname = "p10-b%d-a%s.npy" % (beta, alpha)
            coefs_path = op.join(data_path, "re-mwe-S16/coefs/", fname)
            # coef = np.load(coefs_path)[:, i]
            save_fname = op.join(save_path, "remwe/%s.npy" % subject)
            # np.save(save_fname, coef)
        elif "gala" in model:
            pass
        else:
            print(df_.ns)

            beta = df_.beta.data[0]
            beta = int(1000 * beta)
            fname = "alpha-%s.npy" % (beta)
            coefs_path = op.join(data_path,
                                 "%s/subjects/%s/%s" % (model, subject, fname))
            # coef = np.load(coefs_path)
            save_fname = op.join(save_path, "%s/%s.npy" % (model, subject))
            # np.save(save_fname, coef)

df = df.rename(columns=dict(ffg_geod="FFG",
               ffa_geod="FFA", V1_geod="V1"))

models = ["re-mwe-S%s" % n_tasks, "fmri", "lasso", "adalasso", "gala"]
model_names = [r"MWE$_{0.5}$", "fMRI", r"MCE ($\ell_1$)",
               r"Reweighted MCE ($\ell_{0.5}$)", "GALA"]

df.model = df.model.replace(models, model_names)

dl = df[df.hemi == "lh"]
dr = df[df.hemi == "rh"]
cols = ["model", "FFG", "V1"]
dl = pd.melt(dl[cols], "model", var_name="Label", value_name="Geodesic (cm)")
dr = pd.melt(dr[cols], "model", var_name="Label", value_name="Geodesic (cm)")

colors = ["forestgreen", "gold", "cornflowerblue", "purple", "indianred"]

legend_models = [Patch(edgecolor=color, facecolor=color, color=color,
                       label=name)
                 for color, name in zip(colors, model_names)]

plt.figure(figsize=(10, 1))
plt.legend(handles=legend_models, ncol=5)
plt.axis("off")
plt.savefig("fig/legend-geodesics.pdf")

f, axes = plt.subplots(1, 2, figsize=(5, 4), sharey=True)
for ax, data, hemi in zip(axes.ravel(), [dl, dr], ["Left", "Right"]):
    ax = sns.stripplot(x="Geodesic (cm)", y="Label", hue="model", data=data,
                       dodge=True, jitter=True, alpha=0.3, ax=ax,
                       palette=colors, hue_order=model_names,)
    data_ = data[data["Geodesic (cm)"] < 20]
    ax = sns.pointplot(x="Geodesic (cm)", y="Label", hue="model", data=data_,
                       markers="d", join=False, ax=ax, dodge=0.532,
                       palette=colors, hue_order=model_names, ci=None,
                       scale=.75)
    ax.set_title("%s hemisphere" % hemi)
    ax.get_legend().remove()
    ax.grid(True, axis="x")
    ax.set_xticks(np.r_[np.arange(0, 21, 4), 23.6])
    ax.set_xticklabels([str(i) for i in np.arange(0, 21, 4)] + [r"$+\infty$"])
ax.set_ylabel("")
plt.savefig("fig/geodesic.pdf", bbox_inches="tight")
