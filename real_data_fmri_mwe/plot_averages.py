import numpy as np
import pandas as pd
from plot_brains import plot_source_estimate
from config import get_subjects_list
import os
from utils import get_argmin, get_labels


if __name__ == "__main__":
    dataset = "ds117"
    subjects = get_subjects_list(dataset)
    n_sources = 6
    min_amplitude = 0.1
    perc = 0.05

    labels = []
    if dataset == "ds117":
        labels.append([])
        views = "ventral"
        n_subjects = 16
        hemis = ["both"]
        task = "visual"
        for hemi in ["lh", "rh"]:
            labels[0].extend(get_labels(["V1"], hemi=hemi,
                                        subjects_dir=None))
            labels[0].extend(get_labels(["G_oc-temp_lat-fusifor"], hemi=hemi,
                                        annot_name="aparc.a2009s",
                                        subjects_dir=None))
            # labels[0].extend(get_labels(["ffg"], hemi=hemi,
            #                             annot_name="neurosynth"))
    else:
        views = "lateral"
        n_subjects = 32
        hemis = ["lh", "rh"]
        task = "auditory"
        for hemi in ["lh", "rh"]:
            labels.append(get_labels(["auditory-cortex"],
                                     hemi=hemi, annot_name="neurosynth"))

    df = pd.read_csv("stats/stats_%s-%s.csv" % (dataset, task), index_col=0)
    dgroup = df.groupby(["beta", "subject", "model", "alpha"])
    dns = dgroup.ns.sum().reset_index()
    dvalue = dgroup.source_max.max().reset_index()

    models = ["re-mwe-S%s" % n_subjects]
    # models = ["lasso", "adalasso"]
    models = ["lasso", "adalasso"]
    dmodels = []
    for model in models:
        root = "%s-%s/%s/" % (dataset, task, model)
        if not os.path.exists(root + "coefs"):
            os.makedirs(root + "coefs")
            os.makedirs(root + "coefs/img")

        if "mwe" in model:
            path = root + "barycenter/"
            dmodel = df[df.model == model]
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
            dmodel = dmodel.groupby(["alpha", "beta"]).ns.mean()
            idx = get_argmin(dmodel.values, n_sources)
            alpha, beta = dmodel.index[idx]
            name = "p10-b%d-a%d" % (100 * beta, alpha)
            for fname, mean_type in zip(["euclidean-", ""],
                                        ["euclidean", "ot"]):
                mean = np.load(path + "%s%s.npy" % (fname, name))
                for hemi, labels_hemi in zip(hemis, labels):
                    img_fname = "fig/%s/%s-%s-%s.png" % (dataset, model,
                                                         mean_type, hemi)
                    plot_source_estimate(mean, hemi=hemi, views=views,
                                         perc=perc,
                                         dataset=dataset,
                                         save_fname=img_fname,
                                         labels=labels_hemi)

        else:
            path = root + "subjects/"
            dmodel = dns[dns.model == model]
            dmodel = dmodel.pivot(index="subject", columns="beta", values="ns")
            assert len(set(subjects) - set(list(dmodel.index))) == 0
            betas = dict(dmodel.apply(get_argmin, axis=1, args=(n_sources,)))
            coefs = []
            for subject, beta in betas.items():
                coef = np.load(path + subject +
                               "/alpha-%d.npy" % (1000 * beta))
                coefs.append(coef)
            mean = np.array(coefs).mean(0)
            save_fname = "%s-%d" % (model, n_sources)
            np.save(root + "coefs/%s.npy" % save_fname, mean)
            for hemi, labels_hemi in zip(hemis, labels):
                img_fname = "fig/%s/%s-%s.png" % (dataset, model, hemi)
                plot_source_estimate(mean, hemi=hemi, views=views, perc=perc,
                                     dataset=dataset,
                                     labels=labels_hemi,
                                     save_fname=img_fname)
