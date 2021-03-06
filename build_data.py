import os
import numpy as np
import mne

from numpy.linalg import norm
from fractions import Fraction
import config as cfg


n_tasks_max = 32


def build_coefs(n_tasks, n_sources=1, overlap=100, seed=None, positive=False,
                hemi="lh", illustration=False, labels_type="any",
                dataset="camcan", spacing="ico4"):
    params = cfg.get_params(dataset)
    data_path = params["data_path"]
    labels = np.load(data_path +
                     "label/labels-%s-%s-%s.npy" % (labels_type,
                                                    spacing, hemi))
    n_labels, n_features = labels.shape
    rng = np.random.RandomState(seed)
    # simulate random activation (one per label)
    coefs = np.zeros((n_features, n_tasks_max))
    if overlap < 0. or overlap > 100:
        raise ValueError("Overlap must be in 0-100%. Got %s." % overlap)

    frac = Fraction(overlap, 100)
    denom = frac.denominator
    numer = frac.numerator
    for l in range(n_sources):
        labels_idx, = np.where(labels[l])
        n_source_in_label = len(labels_idx)
        choices = np.arange(n_source_in_label)
        permutation = rng.permutation(choices)
        ido = permutation[0]
        if positive:
            sign = 1
        else:
            sign = (- 1) ** rng.randint(2)
        idx_tasks_o = np.arange(n_tasks_max)
        mod = idx_tasks_o % denom < numer
        idx_tasks_o = idx_tasks_o[mod]
        vals = 10. * (2 + rng.rand(len(idx_tasks_o)))
        coefs[labels_idx[ido], idx_tasks_o] = sign * vals
        idx_no = list(set(np.arange(n_tasks_max)) - set(idx_tasks_o))
        no = len(idx_no)
        if no:
            choices_no = permutation[1:no + 1]
            vals = 10. * (2 + rng.rand(no))
            idno = rng.choice(choices_no, size=no)
            coefs[labels_idx[idno], np.array(idx_no)] = sign * vals
    coefs = coefs[:, :n_tasks]
    return coefs


def build_dataset(coefs, std=0.2, seed=None, same_design=False,
                  randomize_subjects=False, hemi="lh", dataset="camcan",
                  age_min=0, age_max=30, spacing="ico4"):
    """Build multi-task regression data."""
    params = cfg.get_params(dataset)
    data_path = params["data_path"]
    rng = np.random.RandomState(seed)
    n_features, n_tasks = coefs.shape
    subjects = cfg.get_subjects_list(dataset, age_min, age_max)[:n_tasks]
    if randomize_subjects:
        subjects = rng.permutation(subjects)
    if same_design:
        s_id = rng.randint(0, n_tasks)
        subjects = n_tasks * [subjects[s_id]]
    x_names = [data_path + "leadfields/X_%s_%s_%s.npy" %
               (s, hemi, spacing) for s in subjects]
    X = np.stack([np.load(x_name) for x_name in x_names], axis=0)
    X = X.astype(np.float64)
    y = [x.dot(coef) for x, coef in zip(X, coefs.T)]
    y = np.array(y)
    y *= 1e4  # get Y in fT/cm
    n_samples = y.shape[1]
    std *= np.std(y, axis=1).mean()
    noise = std * rng.randn(n_tasks_max, n_samples)
    y += noise[:n_tasks]
    X = X[:n_tasks]
    y = y[:n_tasks]

    return X, y


def get_subjects(n_tasks, seed=None):
    """Build multi-task regression data."""
    rng = np.random.RandomState(seed)
    ids = np.arange(1, n_tasks_max)
    ids = rng.permutation(ids)[:n_tasks_max]
    subjects = ["sub%03d" % i for i in ids]
    return subjects[:n_tasks]


def get_snrs(X, Y, coefs):
    signal = [x.dot(coef) for x, coef in zip(X, coefs.T)]
    snrs = [norm(s) / norm(yy - s) for s, yy in zip(signal, Y)]
    return snrs


if __name__ == "__main__":
    dataset = "camcan"
    plot = True
    topo = False
    hemi = "lh"
    n_features, n_tasks = 642, 32
    if hemi == "rh":
        n_features = 642
    seed = 76737383
    overlap = 50
    n_sources = 5
    grad_only = True
    spacing = "ico4"
    # coefs = build_coefs(n_tasks=n_tasks, overlap=overlap,
    #                     n_sources=n_sources, seed=seed)
    coefs = build_coefs(n_tasks=n_tasks, overlap=overlap,
                        n_sources=n_sources, seed=seed, hemi=hemi,
                        labels_type="any", dataset=dataset,
                        spacing=spacing)

    # X, y = build_dataset(coefs, std=0.5, seed=seed,
    #                      same_design=False, hemi=hemi,
    #                      dataset=dataset, spacing=spacing)
    # params = cfg.get_params(dataset)
    # info = params["info"]
    # grad_ind = params["grad_indices"]
    # info_grad = mne.pick_info(info, grad_ind)

    # if not os.path.exists("/home/parietal/"):
    #
    #     if topo:
    #         if grad_only:
    #             ev = mne.evoked.EvokedArray(y.T[:, :1] * 1e-13,
    #                                         info=info_grad)
    #         else:
    #             ev = mne.evoked.EvokedArray(y.T[:, :1] * 1e-13, info=info)
    #         ev.plot_topomap()
    #
    #     if plot:
    #         from plot_brains import plot_source_estimate
    #         plot_source_estimate(coefs, hemi="lh", views="lateral",
    #                              figsize=(400, 400))

    # coefs_f = _build_coefs_old(n_tasks=4, overlap=0.5,
    #                            n_sources=2, seed=seed, mode="full")

    # Xf, yf = build_dataset(coefs_f, mode="full")
