import numpy as np
import os

from smtr import STL, utils, AdaSTL
from get_real_data import get_dataset
from joblib import delayed, Parallel
from smtr.estimators.solvers.solver_mtw import barycenterkl, barycenterkl_log
import warnings
import pickle
from config import get_params


if os.path.exists("/home/parietal/"):
    try:
        import cupy as cp
    except:
        pass

n_tasks = 16
spacing = "ico4"
dataset = "camcan"
task = "auditory"
if dataset == "camcan":
    n_tasks = 32
params = get_params(dataset)
subjects_dir = params["subjects_dir"]
data_path = params["data_path"]
M_fname = data_path + "metrics/metric_fsaverage_%s_lrh.npy" % spacing
M_ = np.load(M_fname)

X, y, _, subjects, ts = get_dataset(dataset, n_tasks, spacing,
                                    task, n_jobs=n_tasks)
y = y.squeeze()

alpha_frs = [0.35]
depth = 0.9
n_features = len(M_)
epsilon = 10. / n_features
gamma = 1.


def get_dirs(dataset, method):
    root_dir = "%s-%s/%s/" % (dataset, task, method)
    ss_dir = root_dir + "subjects/"
    bar_path = root_dir + "barycenter/"
    log_path = root_dir + "log/"
    if not os.path.exists(root_dir):
        os.makedirs(root_dir, exist_ok=True)
    if not os.path.exists(ss_dir):
        os.makedirs(ss_dir, exist_ok=True)
    if not os.path.exists(bar_path):
        os.makedirs(bar_path, exist_ok=True)
        os.makedirs(bar_path + "img/", exist_ok=True)
    if not os.path.exists(log_path):
        os.makedirs(log_path, exist_ok=True)
    return ss_dir, bar_path, log_path


def run_lasso(subject, alpha_fr, Xi, yi, model="lasso"):
    print(">> Entering worker alpha", alpha_fr)

    ss_dir, bar_path, log_path = get_dirs(dataset, model)
    coefs_path = ss_dir + "%s/" % subject
    if not os.path.exists(coefs_path):
        os.makedirs(coefs_path, exist_ok=True)

    norms = np.linalg.norm(Xi, axis=1) ** depth
    X_scaled = Xi / norms[:, None, :]
    alphamax = max([abs(x.T.dot(yy)).max() for (x, yy) in zip(X_scaled, yi)])
    alphamax /= Xi.shape[1]
    alpha = alpha_fr * alphamax
    if model == "lasso":
        model = STL
    else:
        model = AdaSTL
    stl = model(alpha=[alpha], positive=False)
    print(X_scaled.shape, yi.shape, norms.shape)
    stl.fit(X_scaled, yi)
    coefs_stl = stl.coefs_ * 1e9 / norms.T
    alpha = int(1000 * alpha_fr)
    coefs_fname = coefs_path + "alpha-%s.npy" % alpha
    np.save(coefs_fname, coefs_stl)
    print(">> Leaving worker alpha", alpha_fr)

    return 0.


def lasso_average(alpha_fr, power=1, model="lasso", device=0):
    ss_dir, bar_path, log_path = get_dirs(dataset, model)
    alpha = int(1000 * alpha_fr)
    coefs = []
    for subj in subjects:
        f = ss_dir + "%s/alpha-%s.npy" % (subj, alpha)
        c = np.load(f)
        coefs.append(c)
    coefs = np.concatenate(coefs, axis=-1)
    coefs_mean = coefs.mean(axis=-1)[:, None]
    f = bar_path + "euclidean-alpha-%s.bpy" % alpha
    np.save(f, coefs_mean)

    log1, log2, logl1, logl2 = {"cstr": []}, {"cstr": []}, {"cstr": []}, \
        {"cstr": []}

    M = M_.copy() ** power  # convert to mm
    M /= np.median(M)
    M = - M / epsilon
    n_features = len(M)
    with cp.cuda.Device(device):
        M = cp.asarray(M)
        coefs1, coefs2 = utils.get_unsigned(coefs)
        if coefs1.max(0).all():
            fot1, log1, _, b1, bar1 = barycenterkl(coefs1 + 1e-100, M, epsilon,
                                                   gamma, tol=1e-7,
                                                   maxiter=5000)
            utils.free_gpu_memory(cp)

            if fot1 is None or not coefs1.max(0).all():
                warnings.warn("""Nan found when averagin, re-fit in
                                 log-domain.""")
                b1 = cp.log(b1 + 1e-100, out=b1)
                fot1, logl1, m1, b1, bar1 = \
                    barycenterkl_log(coefs1, M, epsilon, gamma,
                                     b=b1, tol=1e-5, maxiter=1000)
                utils.free_gpu_memory(cp)
        else:
            bar1 = np.zeros(n_features)
        if coefs2.max(0).all():
            fot2, log2, _, b2, bar2 = barycenterkl(coefs2 + 1e-100, M, epsilon,
                                                   gamma, tol=1e-7,
                                                   maxiter=5000)
            utils.free_gpu_memory(cp)

            if fot2 is None or not coefs2.max(0).all():
                warnings.warn("""Nan found when averagin, re-fit in
                                 log-domain.""")
                b2 = cp.log(b2 + 1e-100, out=b2)
                fot2, logl2, m2, b2, bar2 = \
                    barycenterkl_log(coefs2, M, epsilon, gamma,
                                     b=b2, tol=1e-5, maxiter=1000)
                utils.free_gpu_memory(cp)
        else:
            bar2 = np.zeros(n_features)
        bar = bar1 - bar2
    bar = bar[:, None]
    fname = "ot-alpha-%s.npy" % alpha
    np.save(bar_path + fname, bar)
    fname = "alpha-%s.pkl" % alpha
    try:
        logs = [log1["cstr"], log2["cstr"], logl1["cstr"], logl2["cstr"]]
        with open(log_path + fname, "wb") as ff:
            pickle.dump(logs, ff)
    except:
        pass
    print(">> LEAVING worker alpha", alpha)
    return 0.

if __name__ == "__main__":
    args = [(s, a, X[i: i + 1], y[i: i + 1], m) for i, s in enumerate(subjects)
            for a in alpha_frs for m in ["lasso", "adalasso"]]

    pll = Parallel(n_jobs=50, backend="multiprocessing")
    dell = delayed(run_lasso)
    it = (dell(i, a, xx, yy, m) for i, a, xx, yy, m in args)
    out = pll(it)

    # pll = Parallel(n_jobs=10)
    # it = (delayed(lasso_average)(a, 1, m, d % 4)
    #       for d, a in enumerate(alpha_frs)
    #       for m in ["lasso", "adalasso"])
    # out = pll(it)
