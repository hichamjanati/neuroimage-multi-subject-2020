import os
import os.path as op

import numpy as np
from hdf5storage import loadmat

import mne

from mayavi import mlab


plot = False

root_dir = op.expanduser("~/Dropbox/hf_dataset_gala/")
subjects_dir = op.join(root_dir, "subjects")
os.environ['SUBJECTS_DIR'] = subjects_dir


surfer_kwargs = dict(subject="fsaverage", background="white",
                     foreground='black', cortex=("gray", -1, 6, True),
                     smoothing_steps=10, views="lateral",
                     initial_time=0.02)

ev_fname = op.join(root_dir, "data/subject_a-ave.fif")

times = mne.read_evokeds(ev_fname, verbose=False)[0].times

subjects = ['subject_a', 'subject_b']

stc_path = op.join(root_dir, "stc")
fig_path = op.join(root_dir, "fig")
data_path = op.join(root_dir, "data")


# subject = subjects[0]
# stc_fname = op.join(stc_path, subject)
# stc_all = mne.read_source_estimate(stc_fname)
vertices = [np.arange(2562), np.arange(2562)]
#
for subject in subjects:
    stc_fname = op.join(stc_path, subject)
    # stc_all += stc
    data = loadmat(op.join(root_dir, "%s-stc.mat" % subject))['JG']
    m = abs(data).max()
    clim = dict(kind='value', pos_lims=[0., 0.1 * m, m])
    stc = mne.SourceEstimate(data=data, vertices=vertices, tmin=times[0],
                             tstep=times[1] - times[0])
    for hemi in ["lh", "rh"]:
        if not op.exists(stc_fname):
            stc.save(stc_fname)
        stc.plot(hemi=hemi, **surfer_kwargs)
        fig_fname = op.join(fig_path, subject + "-" + hemi + ".png")
        mlab.savefig(fig_fname)
