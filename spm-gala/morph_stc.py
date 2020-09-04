import os
import os.path as op

import mne

from groupmne import group_model
import config as cfg

from mne.parallel import parallel_func


plot = False

subjects_dir = cfg.get_subjects_dir("camcan")
os.environ['SUBJECTS_DIR'] = subjects_dir


stc_path = "/storage/store/work/hjanati/datasets/camcan-spm/data/"
derivatives_path = "/Users/hichamjanati/Dropbox/camcan-spm/fig"

subjects = cfg.get_subjects_list("camcan")

# src_ref = group_model.get_src_reference(spacing=spacing,
#                                         subjects_dir=subjects_dir)

# surfer_kwargs = dict(subject="fsaverage", background="white",
#                      foreground='black', cortex=("gray", -1, 6, True),
#                      smoothing_steps=15, views="lateral")
#


def morph_stc(subject):
    stc_fname = op.join(stc_path, subject)
    stc = mne.read_source_estimate(stc_fname)
    stc.vertices[0] -= 1
    stc.vertices[1] -= 1

    morph = mne.compute_source_morph(stc, subject_from=subject,
                                     spacing=4)
    stc_morphed = morph.apply(stc)
    stc_morphed_fname = op.join(stc_path, subject + "-morphed")
    stc_morphed.save(stc_morphed_fname)


n_jobs = 10
parallel, run_func, _ = parallel_func(morph_stc, n_jobs=n_jobs)

parallel(run_func(s) for s in subjects)
