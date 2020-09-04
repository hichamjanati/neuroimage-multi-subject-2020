import numpy as np
import os

from smtr import MTW, utils
from get_real_data import get_dataset
from joblib import Parallel, delayed
from time import time
from smtr.estimators.solvers.solver_mtw import barycenterkl, barycenterkl_log
import warnings
import pickle
from config import get_params, get_subjects_list


if os.path.exists("/home/parietal/"):
    try:
        import cupy as cp
        gpu = True
        derivatives_path = "/storage/store/work/hjanati/neuroimage19/sources"
    except:
        warnings.warn("No GPUs found.")
        gpu = False
        derivatives_path = "data"
        pass

n_tasks = 32
spacing = "ico4"
dataset = "camcan"
task = "auditory"
if dataset == "ds117":
    n_tasks = 16
    task = "visual"
params = get_params(dataset)
subjects = get_subjects_list(dataset)[:n_tasks]
subjects_dir = params["subjects_dir"]
data_path = params["data_path"]
metric_fname = data_path + "metrics/metric_fsaverage_%s_lrh.npy" % spacing
M_ = np.load(metric_fname)

depth = 0.9
X_full, y_full, _, subjects, ts = get_dataset(dataset, n_tasks, spacing,
                                              task, n_jobs=n_tasks)
y_full = y_full.squeeze()
n_tasks, n_samples, n_features = X_full.shape

epsilon = 10. / n_features
gamma = 1.
depth_ = int(100 * depth)
reweighting_tol = 1e-2
reweighting_steps = 100


def setup_data(n_tasks):
    X_ = X_full[:n_tasks].copy()
    norms = np.linalg.norm(X_, axis=1) ** depth
    X_scaled = X_ / norms[:, None, :]
    y = y_full[:n_tasks].copy()
    sigma00 = np.linalg.norm(y, axis=1).min() / (n_samples ** 0.5)
    sigma0 = 0.01
    betamax = min([abs(x.T.dot(yy)).max() for (x, yy) in zip(X_scaled, y)])
    betamax /= n_samples

    if sigma0:
        betamax /= sigma00

    return X_scaled, y, norms, sigma0, betamax


def setup_path(dataset, n_tasks):
    if reweighting_steps > 1:
        model = "re-mwe-S%s" % n_tasks
    else:
        model = "mwe-S%s-right" % n_tasks
    ss_dir = "%s/%s-%s/%s/" % (derivatives_path, dataset, task, model)
    coefs_path = ss_dir + "coefs/"
    subjects_path = ss_dir + "subjects/"
    bar_path = ss_dir + "barycenter/"
    thetabar_path = ss_dir + "thetabar/"
    log_path = ss_dir + "log/"

    if not os.path.exists(ss_dir):
        os.makedirs(ss_dir, exist_ok=True)
    if not os.path.exists(coefs_path):
        os.makedirs(coefs_path, exist_ok=True)
        os.makedirs(coefs_path + "img/", exist_ok=True)
    subject_paths = []
    for sub in subjects[:n_tasks]:
        subpath = subjects_path + sub + "/"
        subject_paths.append(subpath)
        if not os.path.exists(subpath):
            os.makedirs(subpath, exist_ok=True)
    if not os.path.exists(bar_path):
        os.makedirs(bar_path, exist_ok=True)
        os.makedirs(bar_path + "img/", exist_ok=True)
    if not os.path.exists(thetabar_path):
        os.makedirs(thetabar_path, exist_ok=True)
        os.makedirs(thetabar_path + "img/", exist_ok=True)
    if not os.path.exists(log_path):
        os.makedirs(log_path, exist_ok=True)
    path = dict(subjects=subject_paths, coefs=coefs_path,
                thetabar=thetabar_path, bar=bar_path, log=log_path)
    return path


def mwe_run(alpha, beta_fr, power, device, n_tasks=n_tasks,
            average=False, gpu=gpu):
    X_scaled, y, norms, sigma0, betamax = setup_data(n_tasks)
    path = setup_path(dataset, n_tasks)
    p = int(10 * power)
    b = int(100 * beta_fr)
    a = str(alpha)
    print(f">> Entering worker :")
    print(f"alpha={a}, beta={b}, n_tasks={n_tasks}, device={device}")
    beta = beta_fr * betamax
    M = M_.copy() ** power  # convert to mm
    M /= np.median(M)
    M = - M / epsilon
    if gpu:
        with cp.cuda.Device(device):
            mwe = MTW(M=M, epsilon=epsilon, gamma=gamma, sigma0=sigma0,
                      stable=False, tol_ot=1e-4, maxiter_ot=30, tol=1e-4,
                      maxiter=4000, positive=False, cython=True, gpu=True,
                      n_jobs=2, alpha=alpha, beta=beta,
                      reweighting_steps=reweighting_steps,
                      reweighting_tol=reweighting_tol)
            mwe.fit(X_scaled, y)
            utils.free_gpu_memory(cp)

    else:
        mwe = MTW(M=M, epsilon=epsilon, gamma=gamma, sigma0=sigma0,
                  stable=False, tol_ot=1e-4, maxiter_ot=30, tol=1e-4,
                  maxiter=4000, positive=False, cython=True, gpu=False,
                  n_jobs=2, alpha=alpha, beta=beta,
                  reweighting_steps=reweighting_steps,
                  reweighting_tol=reweighting_tol)
        mwe.fit(X_scaled, y)
    print("TIMES %s, %s : OT = %f s, CD = %f" %
          (a, b, mwe.log_["t_ot"], mwe.log_["t_cd"]))
    coefs_mwe = mwe.coefs_
    coefs_mwe = 1e9 * mwe.coefs_ / norms.T

    fname = "p%d-b%d-a%s.npy" % (p, b, a)
    for coef, subpath in zip(coefs_mwe.T, path["subjects"]):
        np.save(subpath + fname, coef)
    np.save(path["coefs"] + fname, coefs_mwe)
    np.save(path["thetabar"] + fname, mwe.barycenter_[:, None])
    keys = ["log_", "sigmas_", "coefs_", "gamma", "epsilon", "barycenter_"]
    log = dict()
    for k in keys:
        log[k] = getattr(mwe, k)
    fname = "p%d-b%d-a%s-ave.pkl" % (p, b, a)
    with open(path["log"] + fname, "wb") as ff:
        pickle.dump(log, ff)

    if average:
        compute_average(alpha, beta_fr, power, gpu, n_tasks)
    if gpu:
        utils.free_gpu_memory(cp)
    print("Leaving worker")
    print(f"alpha={a}, beta={b}, n_tasks={n_tasks}, device={device}")

    return 0.


def compute_average(alpha, beta_fr, power, device, n_tasks=n_tasks):
    print("Computing barycenter ...")
    log1, log2, logl1, logl2 = {"cstr": []}, {"cstr": []}, {"cstr": []}, \
        {"cstr": []}
    p = int(10 * power)
    b = int(100 * beta_fr)
    a = str(alpha)
    path = setup_path(dataset, n_tasks)
    fname = "p%d-b%d-a%s.npy" % (p, b, a)
    coefs = np.load(path["coefs"] + fname)
    M = M_.copy() ** power  # convert to mm
    M /= np.median(M)
    M = - M / epsilon
    coefs1, coefs2 = utils.get_unsigned(coefs)
    mean = coefs1.mean(-1) - coefs2.mean(-1)
    mean = mean[:, None]
    f = path["bar"] + "euclidean-%s" % fname
    np.save(f, mean)
    with cp.cuda.Device(device):
        M = cp.asarray(M)
        fot1, log1, _, b1, bar1 = barycenterkl(coefs1 + 1e-100, M, epsilon,
                                               gamma, tol=1e-7,
                                               maxiter=3000)
        utils.free_gpu_memory(cp)

        if fot1 is None or not coefs1.max(0).all():
            warnings.warn("""Nan found when averagin, re-fit in
                             log-domain.""")
            b1 = cp.log(b1 + 1e-100, out=b1)
            fot1, logl1, m1, b1, bar1 = \
                barycenterkl_log(coefs1, M, epsilon, gamma,
                                 b=b1, tol=1e-7, maxiter=3000)
            utils.free_gpu_memory(cp)

        fot2, log2, _, b2, bar2 = barycenterkl(coefs2 + 1e-100, M, epsilon,
                                               gamma, tol=1e-7,
                                               maxiter=3000)
        utils.free_gpu_memory(cp)

        if fot2 is None or not coefs2.max(0).all():
            warnings.warn("""Nan found when averagin, re-fit in
                             log-domain.""")
            b2 = cp.log(b2 + 1e-100, out=b2)
            fot2, logl2, m2, b2, bar2 = \
                barycenterkl_log(coefs2, M, epsilon, gamma,
                                 b=b2, tol=1e-7, maxiter=3000)
            utils.free_gpu_memory(cp)

        bar = bar1 - bar2
    bar = bar[:, None]
    np.save(path["bar"] + fname, bar)
    fname = "p%d-b%d-a%s-bar.pkl" % (p, b, a)
    logs = [log1["cstr"], log2["cstr"], logl1["cstr"], logl2["cstr"]]
    with open(path["log"] + fname, "wb") as ff:
        pickle.dump(logs, ff)
    print(">> Barycenter computed for alpha, beta, p", a, b / 100, p / 10)
    return 0.


if __name__ == "__main__":

    t = time()
    betafrs = np.array([0.3])
    n_betas = len(betafrs)
    alphas = [0., 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.12, 0.14,
              0.16, 0.18, 0.25, 0.4]
    n_alphas = len(alphas)
    powers = [1.]
    n_jobs = 20
    n_tasks_all = [n_tasks]
    average = True
    args = [[a, b, p, t] for a in alphas for b in betafrs for p in powers
            for t in n_tasks_all]
    # args = args[9:]
    if gpu:
        n_jobs = 15
    pll = Parallel(n_jobs=n_jobs, backend="multiprocessing")
    dell = delayed(mwe_run)
    # dell = delayed(compute_average)
    it = (dell(a, b, p, d % 2 + 2, t, True)
          for d, (a, b, p, t) in enumerate(args))
    output = pll(it)

    t = time() - t
    print("=======> TIME: %.2f" % t)
