import os
import os.path as op

import numpy as np

import mne

from mayavi import mlab

from hdf5storage import loadmat

import config as cfg


plot = False

subjects_dir = cfg.get_subjects_dir("camcan")
os.environ['SUBJECTS_DIR'] = subjects_dir


surfer_kwargs = dict(subject="fsaverage", background="white",
                     foreground='black', cortex=("gray", -1, 6, True),
                     smoothing_steps=10, views="lateral")

root = "/Users/hichamjanati/Dropbox/camcan-spm-gala/"
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


# subjects_all = ['CC121111', 'CC120640', 'CC120309']

stc_path = "/Users/hichamjanati/Dropbox/camcan-spm-gala/results/gala/stc"
fig_path = "/Users/hichamjanati/Dropbox/camcan-spm-gala/results/gala/fig"
mat_path = "/Users/hichamjanati/Dropbox/camcan-spm-gala/results/gala/mat"

subject = subjects_all[0]
# stc_fname = op.join(stc_path, subject)
# stc_all = mne.read_source_estimate(stc_fname)
vertices = [np.arange(2562), np.arange(2562)]

for i, subject in enumerate(subjects_all):
    stc_fname = op.join(stc_path, subject)
    data = loadmat(op.join(mat_path, "%s.mat" % subject))["SS"][0, 0][0]

    stc = mne.SourceEstimate(data=data, vertices=vertices, tmin=times[0],
                             tstep=times[1] - times[0])
    if i == 0:
        stc_all = [stc]
    else:
        stc_all.append(stc)
    for i, hemi in enumerate(["lh", "rh"]):
        col = i * 2562
        m = abs(stc.data[col:]).max()
        clim = dict(kind='value', pos_lims=[0., 0.8 * m, m])
        stc.plot(hemi=hemi, clim=clim, **surfer_kwargs)
        fig_fname = op.join(fig_path, subject + "-" + hemi + ".png")
        mlab.savefig(fig_fname)
stc_all = sum(stc_all) / len(subjects_all)

for i, hemi in enumerate(["lh", "rh"]):
    col = i * 2562
    m = abs(stc_all.data[col:]).max()
    clim = dict(kind='value', pos_lims=[0., 0.8 * m, m])
    stc_all.plot(hemi=hemi, clim=clim, **surfer_kwargs, time_label="")
    fig_fname = op.join(fig_path, "average-" + hemi + ".png")
    mlab.savefig(fig_fname)
