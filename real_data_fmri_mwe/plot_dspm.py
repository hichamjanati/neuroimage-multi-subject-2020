import numpy as np
from plot_brains import plot_source_estimate
from utils import get_labels


dspm_list = dict(ds117=["euclidean-snr-%d.npy" % i for i in [3]],
                 camcan=["euclidean-snr-%d.npy" % i for i in [3]])

for dataset in ["ds117"]:
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

    path = "%s-%s/dspm/barycenter/" % (dataset, task)
    for file_name in dspm_list[dataset]:
        coefs = np.load(path + file_name)
        fig_fname = "fig/%s/dspm-%s" % (dataset, file_name[:-4])
        for hemi, labels_hemi in zip(hemis, labels):
            plot_source_estimate(coefs, hemi=hemi, views=views,
                                 save_fname=fig_fname + "-%s.png" % hemi,
                                 labels=labels_hemi, lims=[2., 4., 8],
                                 dataset="camcan")
