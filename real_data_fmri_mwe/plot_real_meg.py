import numpy as np
import os
from plot_brains import plot_source_estimate, plot_blobs
import glob


dataset = "camcan"
views = "lateral"
n_subjects = [32]
hemis = ["lh", "rh"]
task = "auditory"
if dataset == "ds117":
    views = "ventral"
    n_subjects = [16]
    hemis = ["both"]
    label_names = ["V1"]
    task = "visual"
    annot_name = "aparc.a2009s"
for n in n_subjects:
    path = "%s-%s/re-mwe-S%d/coefs/" % (dataset, task, n)
    # path = "%s/adalasso/subjects/sub017/" % (dataset)
    one_subject = False
    perc = 0.05
    one_subject = ""
    subject = "CC110033"
    lims = None
    if "dspm" in path:
        lims = [2, 5, 7]
    exists_ok = True
    plot_this = ["p10-b30-a%s.npy" % i for i in [0.1]]
    # plot_this = ["euclidean-snr-3.npy"]
    plot_this = []
    top_vertices = 2000

    for f in glob.glob(path + "*.npy"):
        file_name = f.split("/")[-1]
        root = "/".join(f.split("/")[:-1])
        if plot_this and file_name not in plot_this:
            continue
        if f.split(".")[-1] == "npy":
            if not os.path.exists(root + "/img/"):
                os.makedirs(root + "/img/")
            coefs = np.load(f)
            # if "dataset" == "ds117" or "dspm" in path:
            #     coefs = abs(coefs)
            fnamev = root + "/img/"
            if isinstance(one_subject, int):
                coefs = coefs[:, one_subject: one_subject + 1]
                fnamev += subject + "/"
                if not os.path.exists(fnamev):
                    os.makedirs(fnamev)

            for hemi in hemis:
                fname_h = fnamev + f"{hemi}-" + file_name[:-4]
                #
                if os.path.exists(fname_h + ".png") and exists_ok:
                    continue
                if top_vertices:
                    plot_blobs(coefs, hemi=hemi,
                               views=views,
                               save_fname=fname_h + ".png",
                               labels=labels,
                               top_vertices=top_vertices,
                               dataset=dataset)
                else:
                    plot_source_estimate(coefs, hemi=hemi,
                                         views=views,
                                         save_fname=fname_h + ".png",
                                         labels=labels,
                                         perc=perc,
                                         lims=lims,
                                         dataset=dataset)
