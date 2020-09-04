# brain data
import os
import mne
import config as cfg
from os import path as op
from joblib import Memory
from groupmne import group_model
from mne.parallel import parallel_func
import pickle


if os.path.exists("/home/parietal/"):
    location = '~/data/'
else:
    location = "~/Dropbox/neuro_transport/code/mtw_experiments/meg/"
location = os.path.expanduser(location)
memory = Memory(location, verbose=0)

condition = None
hemi = "lrh"
spacing = "ico4"
time_point = "auto"
task = "visual"
tmin_tmax = dict(visual=[0.150, 0.190], auditory=[0.08, 0.12])


def get_subject_data(dataset, subject, time_point=time_point, task=task):
    # read data
    evoked_fname = cfg.get_ave_fname(dataset, subject)
    evoked = mne.read_evokeds(evoked_fname, condition=condition,
                              verbose=False)
    if dataset == "camcan":
        if task == "auditory":
            evoked = evoked[:3]
            evoked = mne.combine_evoked(evoked, "nave")
        else:
            evoked = evoked[3]
    else:
        evoked = evoked[3]
        assert task == "visual"
        assert evoked.comment == "contrast"
    # epo_fname = cfg.get_epo_fname(dataset, subject)
    cov_fname = cfg.get_cov_fname(dataset, subject)
    # epochs = mne.read_epochs(epo_fname)
    noise_cov = mne.read_cov(cov_fname, verbose=False)

    t = -1.
    if dataset == "camcan":
        evoked.shift_time(-0.05)
        # epochs.shift_time(-0.05)
    # noise_cov = mne.compute_covariance(epochs, tmin=None, tmax=0.0,
    #                                    method="auto", verbose=True, n_jobs=1,
    #                                    projs=None)
    if isinstance(time_point, int):
        t = time_point / 1000  # get time in seconds
        evoked.crop(tmin=t, tmax=t)
    elif time_point == "auto":
        tmin, tmax = tmin_tmax[task]
        _, t = evoked.get_peak("grad", tmin=tmin, tmax=tmax,
                               merge_grads=True, time_as_index=False)
        evoked.crop(tmin=t, tmax=t)
    else:
        pass
    t *= 1000
    return evoked, noise_cov, t


def build_dataset(dataset, n_subjects, spacing="ico4", task="auditory",
                  time_point="auto", n_jobs=1):
    import os
    subjects_dir = cfg.get_subjects_dir(dataset)
    os.environ['SUBJECTS_DIR'] = subjects_dir
    src_ref = group_model.get_src_reference(spacing=spacing,
                                            subjects_dir=subjects_dir)
    subjects = cfg.get_subjects_list(dataset)[:n_subjects]
    parallel, run_func, _ = parallel_func(group_model.compute_fwd,
                                          n_jobs=n_jobs)

    raw_name_s = [cfg.get_raw_fname(dataset, s) for s in subjects]
    trans_fname_s = [cfg.get_trans_fname(dataset, s) for s in subjects]
    bem_fname_s = [cfg.get_bem_fname(dataset, s) for s in subjects]

    fwds = parallel(run_func(s, src_ref, info, trans, bem,  mindist=3,
                             meg=True, eeg=False)
                    for s, info, trans, bem in zip(subjects, raw_name_s,
                                                   trans_fname_s, bem_fname_s))

    parallel, run_func, _ = parallel_func(get_subject_data,
                                          n_jobs=n_jobs)
    ev_and_cov = parallel(run_func(dataset, s, time_point, task)
                          for s in subjects)
    evoked_s, noise_cov_s, ts = list(zip(*ev_and_cov))

    gains, M, group_info = \
        group_model.compute_inv_data(fwds, src_ref, evoked_s, noise_cov_s,
                                     ch_type="grad", tmin=None, tmax=None)
    group_info["task"] = task
    save_dir = "~/data/%s/" % dataset
    save_dir = op.expanduser(save_dir)
    ff = open(save_dir + "leadfields/group_info_%s.pkl" % spacing,
              "wb")
    pickle.dump(group_info, ff)

    return gains, M, group_info, subjects, ts


get_dataset = memory.cache(build_dataset)


if __name__ == "__main__":
    dataset = "ds117"
    # Xss, yss, group_info, sss, tss = build_dataset("camcan", 32, "visual", 32)
    XXs, yys, group_info, sss, tts = get_dataset(dataset, 16, n_jobs=16)
