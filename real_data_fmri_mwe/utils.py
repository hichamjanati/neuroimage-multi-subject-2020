import numpy as np
import os
import mne


def compute_r2(coefs, x, y):
    coefs = coefs.squeeze() * 1e-9
    y_pred = x.dot(coefs)
    explained = ((y - y_pred) ** 2).sum()
    total_var = (y ** 2).sum()
    gof = 1 - explained / total_var
    return gof


def get_multiple_indices(values, search_list):
    indices = []
    for v in values:
        indices.extend(np.where(search_list == v)[0].tolist())
    return indices


def count_sources(coefs, name, vertices, labels, hemi_indx=0, active_set=10):
    hemis = ["lh", "rh"]
    sources_best = np.argsort(abs(coefs).squeeze())[:-active_set:-1]
    support = np.where(coefs != 0.)[0]
    sources_best = np.array(list(set(sources_best) & set(support)))
    ll = labels[name + "-%s" % hemis[hemi_indx]]
    count_ratio, max_ratio = 0., 0.
    if len(sources_best):
        active = set(vertices[hemi_indx][sources_best])
        n = len(set(ll.vertices) & active)
        if len(active):
            count_ratio = n / len(active)
    active = set(vertices[hemi_indx][support])
    ids = get_multiple_indices(list(set(ll.vertices) & active),
                               vertices[hemi_indx])
    if len(ids):
        max_ratio = abs(coefs[ids]).max()

    return count_ratio, max_ratio


def get_geodesic(name, labels, coefs, vertices, M, hemi_indx=0):
    hemis = ["lh", "rh"]
    if abs(coefs.flatten()).max():
        argmax = np.argmax(abs(coefs).flatten())
        n_l = len(vertices[0])
        ll = labels[name + "-%s" % hemis[hemi_indx]]
        geod = 0
        ids = get_multiple_indices(list(set(ll.vertices)), vertices[hemi_indx])
        for vertex in ids:
            geod += M[hemi_indx * n_l + argmax, hemi_indx * n_l + vertex]
        geod /= len(ids)
    else:
        geod = M.max()
    return geod


def get_argmin(x, n_sources=4):
    y = x.copy()
    y[y == 0.] = 10000
    return np.argmin(abs(y - n_sources))


def get_labels(label_names, annot_name=None, hemi="lh",
               subjects_dir=None):
    if subjects_dir is None:
        subjects_dir = mne.datasets.sample.data_path() + "/subjects/"
    labels = []
    if annot_name is None:
        for label_name in label_names:
            fname = subjects_dir + \
                "fsaverage/label/%s.%s.label" % (hemi, label_name)
            label = mne.read_label(fname)
            labels.append(label)
    elif annot_name == "neurosynth":
        labels_dir = "~/Dropbox/neuro_transport/code/" + \
            "mtw_experiments/neurosynth-labels"
        labels_dir = os.path.expanduser(labels_dir)

        for name in label_names:
            label_name = os.path.join(labels_dir, name)
            label = mne.read_label(label_name + "-%s.label" % hemi)
            labels.append(label)
    else:
        labels_raw = mne.read_labels_from_annot("fsaverage", annot_name,
                                                subjects_dir=subjects_dir)
        for l in labels_raw:
            for name in label_names:
                if name + "-" + hemi in l.name:
                    labels.append(l)
    return labels


def merge_labels(labels_list):
    sum_labels = labels_list[0]
    if len(labels_list) > 1:
        for l in labels_list[1:]:
            sum_labels += l
    return sum_labels
