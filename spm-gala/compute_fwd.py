import os
import os.path as op

import numpy as np
from scipy import io

import mne
from mne.parallel import parallel_func

from groupmne import group_model
import config as cfg

import nibabel as nib


plot = False

subjects_dir = cfg.get_subjects_dir("camcan")
os.environ['SUBJECTS_DIR'] = subjects_dir

derivatives_path = "/storage/store/work/hjanati/datasets/camcan-spm/data"
fwd_path = op.join(derivatives_path, "forwards")
permutations_path = op.join(derivatives_path, "permutations")
evokeds_path = op.join(derivatives_path)
mri_path = op.join(derivatives_path, "mri")

resolution = 4
spacing = "ico%d" % resolution
subject = "fsaverage"

# src_ref = group_model.get_src_reference(spacing=spacing,
#                                         subjects_dir=subjects_dir)


def compute_fwd(subject, meg=True, eeg=True, mindist=2,
                subjects_dir=None, n_jobs=1):
    """Compute forward."""

    # src = mne.morph_source_spaces(src_ref, subject_to=subject,
    #                               subjects_dir=subjects_dir)
    # bem_fname = cfg.get_bem_fname("camcan", subject)
    # bem = mne.read_bem_solution(bem_fname)
    # trans_fname = cfg.get_trans_fname("camcan", subject)
    evoked_fname = cfg.get_ave_fname("camcan", subject)
    evokeds = mne.read_evokeds(evoked_fname, verbose=False)
    evoked = evokeds[:3]
    evoked = mne.combine_evoked(evoked, "nave")
    evoked.shift_time(-0.05)
    tmin = 0.08
    tmax = 0.12
    _, t = evoked.get_peak("grad", tmin=tmin, tmax=tmax,
                           merge_grads=True, time_as_index=False)
    evoked.crop(tmin=t, tmax=t + 1 / evoked.info["sfreq"])
    new_fname = op.join(evokeds_path, "%s-ave.fif" % subject)
    mne.write_evokeds(new_fname, evoked)
    # fwd = mne.make_forward_solution(evoked.info, trans=trans_fname, src=src,
    #                                 bem=bem, meg=meg, eeg=eeg,
    #                                 mindist=mindist,
    #                                 n_jobs=n_jobs)
    # fwd_fname = op.join(fwd_path, "%s-fwd.fif" % subject)
    # mne.write_forward_solution(fwd_fname, fwd, overwrite=True)
    #
    # return fwd

###################################################################
# the function group_model.compute_fwd morphs the source space src_ref to the
# surface of each subject by mapping the sulci and gyri patterns
# and computes their forward operators.


subjects = cfg.get_subjects_list("camcan")
for s in subjects:
    compute_fwd(s)
# n_jobs = len(subjects)
# parallel, run_func, _ = parallel_func(compute_fwd, n_jobs=n_jobs)
#
# fwds = parallel(run_func(s, mindist=3)
#                 for s in subjects)
#
# gains, group_info = group_model._group_filtering(fwds, src_ref,
#                                                  noise_covs=None)
# n_sources = 2562
# assert gains.shape[-1] == 2 * n_sources
#
# hemis = ["lh", "rh"]
# permutations = []
# for ii, subject in enumerate(subjects):
#     permutation = []
#     for jj, hemi in enumerate(hemis):
#         vertno = group_info["vertno_%s" % hemi][ii]
#         permutation_hemi = np.argsort(np.argsort(vertno))
#         # if right hemi, add 2562 before concatenating
#         if jj:
#             permutation_hemi += n_sources
#         permutation.append(permutation_hemi)
#     permutation = np.concatenate(permutation)
#     data_dict = dict(permutation=permutation)
#     permut_name = op.join(permutations_path, "%s-permutation.mat" % subject)
#     io.savemat(permut_name, data_dict)
#     permutations.append(permutation)
#
# for i in range(2):
#     fwd = mne.convert_forward_solution(fwds[i], surf_ori=True,
#                                        force_fixed=True,
#                                        use_cps=True)
#     new_gain = fwd["sol"]["data"][:, permutations[i]]
#
#     # check that permutated gains match with the ones returned by groupmne
#     assert abs(gains[i] - new_gain).max() == 0
#
#
# if plot:
#     # see if vertices correspond across subjects
#     # and if they're on the right hemi
#     vertex = 5000
#     for i, s in enumerate(subjects[:5]):
#         ev_fname = cfg.get_ave_fname("camcan", s)
#         data = gains[i, :, vertex]
#         info = mne.read_evokeds(ev_fname)[0].info
#         evoked = mne.EvokedArray(data[:, None], info=info)
#         evoked.plot_topomap()
#
# subjects_fname = op.join(derivatives_path, "subjects.mat")
# data_dict = dict(subjects=subjects)
# io.savemat(subjects_fname, data_dict)
#
#
# for subject in subjects:
#     mri_fname = op.join(subjects_dir, subject, "mri", "T1.mgz")
#     nii_fname = op.join(mri_path, "%s-T1.nii" % subject)
#
#     mri = nib.load(mri_fname)
#     nib.save(mri, nii_fname)
