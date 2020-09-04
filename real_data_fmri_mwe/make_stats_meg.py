import os
import os.path as op

import pickle

import numpy as np
from scipy.io import loadmat

import pandas as pd
import mne

import config as cfg
from get_real_data import get_dataset
import glob
import utils


dataset = "ds117"
task = "auditory"
spacing = "ico4"
if os.path.exists("/home/parietal/"):
    data_path = "/home/parietal/hjanati/data/"
else:
    data_path = "~/Dropbox/neuro_transport/code/mtw_experiments/meg/"
    data_path += dataset + "/"
    data_path = os.path.expanduser(data_path)
subjects_dir = data_path + "subjects/"
os.environ['SUBJECTS_DIR'] = subjects_dir

metric_fname = data_path + "metrics/metric_fsaverage_%s_lrh.npy" % spacing
M_ = np.load(metric_fname)
M = M_ * 100
labels_dir = "~/Dropbox/neuro_transport/code/mtw_experiments/neurosynth-labels"
labels_dir = os.path.expanduser(labels_dir)

n_tasks = 32
if dataset == "ds117":
    n_tasks = 16
    task = "visual"
X, y, group_info, subjects, _ = get_dataset(dataset, n_tasks, task=task,
                                            n_jobs=n_tasks)
y = y.squeeze()


if __name__ == "__main__":
    hemis = ["lh", "rh"]
    file_path = data_path + "leadfields/group_info_%s.pkl" % spacing
    group_info_file = open(file_path, "rb")
    group_info = pickle.load(group_info_file)
    verts = group_info["vertno_ref"]
    n_l = len(verts[0])
    hemis = ["lh", "rh"]

    if dataset == "ds117":
        fname = "aparca2009s"
        labels_dict_fname = data_path + "label/%s-%s.pkl" % (fname, spacing)
        with open(labels_dict_fname, "rb") as ff:
            labels = pickle.load(ff)

        labels["ffg-lh"] = labels["G_oc-temp_lat-fusifor-lh"]
        labels["ffg-rh"] = labels["G_oc-temp_lat-fusifor-rh"]
        v1_name = "V1"

        for hemi in hemis:
            v1_fname = subjects_dir + \
                "fsaverage/label/%s.V1_exvivo.label" % hemi
            v1 = mne.read_label(v1_fname)
            v1 = v1.morph("fsaverage", subject_to="fsaverage", grade=4,
                          subjects_dir=subjects_dir)
            labels[v1_name + "-%s" % hemi] = v1
            ffa_fname = labels_dir + "/ffa-%s.label" % hemi
            ffa = mne.read_label(ffa_fname)
            labels["ffa" + "-%s" % hemi] = ffa

    path_names = ["%s-%s/mwe-%s" % (dataset, task, n_tasks)]
    models = ["re-mwe-S%s" % n_tasks, "mwe-S%s" % n_tasks]
    label_names = ["ffg", "V1", "ffa"]
    depth = 0.9
    df = pd.DataFrame()
    subjects = cfg.get_subjects_list(dataset)
    n_tasks = len(subjects)

    for model in models:
        print("Doing %s ... " % model)
        ss = "data/%s-%s/%s" % (dataset, task, model)
        coefs_path = "%s/coefs/" % ss
        all_coefs = glob.glob(coefs_path + "*.npy")
        for i, f in enumerate(all_coefs):
            fff = f.split("/")[-1].split(".npy")[0].split("-")
            coefs = np.load(f)
            alpha = fff[2][1:]
            beta = int(fff[1][1:]) / 100
            power = int(fff[0][1:]) / 10
            for j in range(n_tasks):
                subject = subjects[j]
                r2 = utils.compute_r2(coefs[:, j], X[j], y[j])
                coefs1, coefs2 = coefs[:n_l, j], coefs[n_l:, j]
                for k, (hemi, coefs_h) in enumerate(zip(hemis,
                                                        [coefs1, coefs2])):
                    ns = (abs(coefs_h) > 0.).sum()
                    max_h = abs(coefs_h).max()
                    d = dict(n_tasks=n_tasks, depth=depth, alpha=alpha,
                             beta=beta, power=power, subject=subject,
                             hemi=hemi, ns=ns, model=model,
                             gof=r2, source_max=max_h)
                    if dataset == "ds117":
                        for name in label_names:
                            ratio, max_in = utils.count_sources(coefs_h,
                                                                name, verts,
                                                                labels, k)
                            geod = utils.get_geodesic(name, labels, coefs_h,
                                                      verts, M, k)
                            d[name + "_count"] = ratio
                            d[name + "_max"] = max_in
                            d[name + "_geod"] = geod

                            if abs(coefs[:, j]).max():
                                d[name + "_max"] /= abs(coefs[:, j]).max()

                    df = df.append(d, ignore_index=True)

    for model in ["lasso", "adalasso"]:
        print("Doing %s ... " % model)
        lasso_path = "data/%s-%s/%s/" % (dataset, task, model)
        for j, subject in enumerate(subjects):
            coefs_path_lasso = lasso_path + "subjects/" + subject + "/"
            all_coefs = glob.glob(coefs_path_lasso + "*.npy")
            n_files = len(all_coefs)
            for i, f in enumerate(all_coefs):
                coefs = np.load(f)[:, 0]
                f = f.split("/")[-1]
                alpha = int(f.split('.')[0].split("-")[-1]) / 1000
                coefs1, coefs2 = coefs[:n_l], coefs[n_l:]
                r2 = utils.compute_r2(coefs, X[j], y[j])
                for k, (hemi, coefs_h) in enumerate(zip(hemis,
                                                        [coefs1, coefs2])):
                    ns = (abs(coefs_h) > 0.0).sum()
                    max_h = abs(coefs_h).max()
                    d = dict(depth=depth, beta=alpha, alpha=0.,
                             subject=subject, hemi=hemi, ns=ns,
                             power=0., gof=r2,
                             model=model, source_max=max_h)
                    if dataset == "ds117":
                        for name in label_names:
                            ratio, max_in = utils.count_sources(coefs_h,
                                                                name, verts,
                                                                labels, k)
                            geod = utils.get_geodesic(name, labels, coefs_h,
                                                      verts, M, k)
                            d[name + "_count"] = ratio
                            d[name + "_max"] = max_in
                            d[name + "_geod"] = geod
                            if abs(coefs).max():
                                d[name + "_max"] /= abs(coefs).max()
                    df = df.append(d, ignore_index=True)

    # Gala
    print("Doing Gala ...")
    model = "gala"
    if op.exists("/parietal"):
        root = "/storage/store/work/hjanati/datasets/ds117-gala/"
    else:
        root = "/Users/hichamjanati/Dropbox/ds117-gala/"
    stc_path = op.join(root, "stc")
    with open(op.join(root, "data/vertno_ref.pkl"), "rb") as vertno_file:
        vertices = pickle.load(vertno_file)
    n_l = vertices[0].size
    for ii, subject in enumerate(subjects):
        stc_fname = op.join(stc_path, "%s-stc.mat" % subject)
        coefs = loadmat(stc_fname)["SS"][0, 0][0][:, 0]
        coefs1, coefs2 = coefs[:n_l], coefs[n_l:]
        r2 = utils.compute_r2(coefs, X[ii], y[ii])
        # r2 = 0
        for k, (hemi, coefs_h) in enumerate(zip(hemis,
                                                [coefs1, coefs2])):
            ns = (abs(coefs_h) > 0.0).sum()
            max_h = abs(coefs_h).max()
            d = dict(depth=depth, beta=0., alpha=0.,
                     subject=subject, hemi=hemi, ns=ns,
                     power=0., gof=r2,
                     model=model, source_max=max_h)
            if dataset == "ds117":
                for name in label_names:
                    ratio, max_in = utils.count_sources(coefs_h,
                                                        name, verts,
                                                        labels, k)
                    geod = utils.get_geodesic(name, labels, coefs_h,
                                              verts, M, k)
                    d[name + "_count"] = ratio
                    d[name + "_max"] = max_in
                    d[name + "_geod"] = geod
                    if abs(coefs).max():
                        d[name + "_max"] /= abs(coefs).max()
            df = df.append(d, ignore_index=True)
    df.to_csv("stats/stats_%s-%s.csv" % (dataset, task))
