# brain data
import os
import numpy as np
import mne

from mne.morph import compute_source_morph as morph
from mne.minimum_norm import make_inverse_operator, apply_inverse
from joblib import Parallel, delayed
from config import (get_subjects_list, get_ave_fname,
                    get_cov_fname, get_params, get_fwd_fname)
from smtr import utils
from smtr.estimators.solvers.solver_mtw import barycenterkl, barycenterkl_log
import warnings
import pickle
if os.path.exists("/home/parietal/"):
    try:
        import cupy as cp
        device = 0
    except ImportError:
        pass

dataset = "ds117"
params = get_params(dataset)
subjects = get_subjects_list(dataset, simu=False)
subjects_dir = params["subjects_dir"]
data_path = params["data_path"]

condition = None
hemi = "both"
spacing = "ico4"

method = "dSPM"
depth = 0.9
tmin_tmax = dict(ds117=[0.150, 0.190], camcan=[0.08, 0.12])


def get_dirs(dataset, method):
    root_dir = "real_data/%s/%s/" % (dataset, method)
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


def mne_source_localization(subject, dataset=dataset, time_point="auto",
                            task="passive", snr=3, method=method):
    params = get_params(dataset)
    subjects_dir = params["subjects_dir"]
    ss_dir, bar_path, log_path = get_dirs(dataset, method)
    coefs_path = ss_dir + "%s/" % subject

    if not os.path.exists(coefs_path):
        os.makedirs(coefs_path, exist_ok=True)
    evoked_fname = get_ave_fname(dataset, subject, task)
    cov_fname = get_cov_fname(dataset, subject)
    noise_cov = mne.read_cov(cov_fname, verbose=False)
    evoked = mne.read_evokeds(evoked_fname, condition=condition,
                              verbose=False)
    if dataset == "camcan":
        evoked = evoked[:3]
        evoked = mne.combine_evoked(evoked, "nave")
    else:
        evoked = evoked[3]
    evoked = evoked.pick_types("grad", eeg=False)
    fwd_fname = get_fwd_fname(dataset, subject)
    fwd = mne.read_forward_solution(fwd_fname)
    forward = mne.convert_forward_solution(fwd, surf_ori=True,
                                           use_cps=True)
    if dataset == "camcan":
        evoked.shift_time(-0.05)
    if time_point == "auto":
        tmin, tmax = tmin_tmax[dataset]
        _, time_point = evoked.get_peak("grad", tmin=tmin, tmax=tmax,
                                        merge_grads=True)
    elif isinstance(time_point, int):
        time_point = time_point / 1000  # get time in seconds
    evoked = evoked.crop(tmin=time_point, tmax=time_point)
    inverse_operator = make_inverse_operator(evoked.info, forward, noise_cov,
                                             loose=0., depth=depth)
    lambda2 = 1. / snr ** 2
    stc = apply_inverse(evoked, inverse_operator, lambda2,
                        method=method, pick_ori=None)
    morph_stc = morph(stc, subject_to="fsaverage", spacing=4,
                      subjects_dir=subjects_dir)
    stc = morph_stc.apply(stc)
    coefs_fname = coefs_path + "%s.npy" % (subject)
    np.save(coefs_fname, stc.data)
    return stc


def mne_average(snr, power=1):
    coefs = []
    coefs_path = ss_dir + "snr%s/" % snr

    for subj in subjects:
        coefs_fname = coefs_path + "%s.npy" % (subj)
        c = np.load(coefs_fname)
        coefs.append(c)
    coefs = np.concatenate(coefs, axis=-1)
    coefs_mean = coefs.mean(axis=-1)[:, None]
    f = bar_path + "euclidean-snr-%s.npy" % snr
    np.save(f, coefs_mean)
    return 0
    log1, log2, logl1, logl2 = {"cstr": []}, {"cstr": []}, {"cstr": []}, \
        {"cstr": []}

    M = M_.copy() ** power  # convert to mm
    M /= np.median(M)
    M = - M / epsilon
    with cp.cuda.Device(device):
        M = cp.asarray(M)
        coefs1, coefs2 = utils.get_unsigned(coefs)
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

        bar = bar1 - bar2
    bar = bar[:, None]
    fname = "ot-snr-%s.npy" % snr
    np.save(bar_path + fname, bar)
    fname = "snr-%s.pkl" % snr
    logs = [log1["cstr"], log2["cstr"], logl1["cstr"], logl2["cstr"]]
    with open(log_path + fname, "wb") as ff:
        pickle.dump(logs, ff)
    print(">> LEAVING worker snr", snr)
    return 0.


pll = Parallel(n_jobs=45)
it = (delayed(mne_source_localization)(s, snr=snr) for s in subjects
      for snr in [1, 2, 3])
out = pll(it)
it = (delayed(mne_average)(s) for s in [1, 2, 3])
out = pll(it)
