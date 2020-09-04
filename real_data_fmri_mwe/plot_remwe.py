import numpy as np
from plot_brains import plot_source_estimate, plot_blobs
from utils import get_labels


top_vertices = 2000
perc = 0.05
plot_support = True  # set to false to show barycentes

if plot_support:
    plot = plot_blobs
    subfolder = "coefs"
    prefixes = [""]
else:
    plot = plot_source_estimate
    subfolder = "barycenter"
    prefixes = ["", "euclidean-"]

alphas = [2]
remwe_list = dict(ds117=["p10-b20-a%s.npy" % i for i in alphas],
                  camcan=["p10-b30-a%s.npy" % i for i in alphas])

for dataset in ["camcan"]:
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
        hemis = ["lh", "rh"][:1]
        task = "auditory"
        for hemi in ["lh", "rh"]:
            labels.append(get_labels(["auditory-cortex"],
                                     hemi=hemi, annot_name="neurosynth"))

    path = "data/%s-%s/re-mwe-S%d/%s/" % (dataset, task, n_subjects, subfolder)
    for file_name in remwe_list[dataset]:
        file_name_ = file_name
        for prefix in prefixes:
            coefs = np.load(path + prefix + file_name)
            if not plot_support:
                file_name_ = "bar-" + prefix + file_name
            fig_fname = "fig/%s/gif/%s" % (dataset, file_name_[:-4])
            for hemi, labels_hemi in zip(hemis, labels):
                plot(coefs, hemi=hemi, views=views,
                     save_fname=fig_fname + "-%s.png" % hemi,
                     labels=labels_hemi, top_vertices=top_vertices,
                     perc=perc, dataset=dataset)
