import os
import os.path as op

import numpy as np

import mne

from mayavi import mlab

import config as cfg


plot = False

subjects_dir = cfg.get_subjects_dir("camcan")
os.environ['SUBJECTS_DIR'] = subjects_dir


surfer_kwargs = dict(subject="fsaverage", background="white",
                     foreground='black', cortex=("gray", -1, 6, True),
                     smoothing_steps=10, views="lateral")

root = "/Users/hichamjanati/Dropbox/camcan-spm/"
ev_fname = op.join(root, "data/CC110033-ave.fif")

results_path = op.join(root, "results")
times = mne.read_evokeds(ev_fname, verbose=False)[0].times

subjects_all = ['CC221373',
                'CC220526',
                'CC210519',
                'CC122172',
                'CC121428',
                'CC121411',
                'CC121144',
                'CC121111',
                'CC121106',
                'CC120727',
                'CC120795',
                'CC120640',
                'CC120550',
                'CC120469',
                'CC120376',
                'CC120319',
                'CC120313',
                'CC120309',
                'CC120264',
                'CC120218',
                'CC120182',
                'CC120166',
                'CC120120',
                'CC120049',
                'CC120061',
                'CC120008',
                'CC112141',
                'CC110606',
                'CC110411',
                'CC110101',
                'CC110187',
                'CC110033']

simu = False
if simu:
    subjects_all = ['CC121111', 'CC120640', 'CC120309']
    n_subjects_all = [3]
else:
    n_subjects_all = [2, 4, 8]
for n_subjects in n_subjects_all:
    subjects = subjects_all[:n_subjects]
    print("Doing %s subjects ..." % n_subjects)
    if simu:
        stc_path = "/Users/hichamjanati/Dropbox/camcan-spm/simulation/spm/"
        fig_path = op.join(stc_path, "fig")
    else:
        stc_path = op.join(results_path, "%ssubjects" % n_subjects)
        fig_path = "/Users/hichamjanati/Dropbox/camcan-spm/fig/spm/"
        fig_path = op.join(fig_path, "%ssubjects" % n_subjects)
    if not op.exists(fig_path):
        os.makedirs(fig_path)

    subject = subjects[0]
    stc_fname = op.join(stc_path, subject)
    stc_all = mne.read_source_estimate(stc_fname)
    vertices = [np.arange(2562), np.arange(2562)]

    for subject in subjects:
        stc_fname = op.join(stc_path, subject)
        stc = mne.read_source_estimate(stc_fname)
        stc.vertices = vertices
        # stc_all += stc
        # m = abs(stc.data).max()
        # clim = dict(kind='value', pos_lims=[0., 0.2 * m, m])
        for hemi in ["lh", "rh"]:
            stc.plot(hemi=hemi, **surfer_kwargs)
            fig_fname = op.join(fig_path, subject + "-" + hemi + ".png")
            mlab.savefig(fig_fname)
    stc_all /= len(subjects)

    # for hemi in ["lh", "rh"]:
    #     stc.plot(hemi=hemi, **surfer_kwargs)
    #     fig_fname = op.join(fig_path, "average-" + hemi + ".png")
    #     mlab.savefig(fig_fname)
