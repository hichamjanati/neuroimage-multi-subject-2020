import os
import os.path as op

import mne

from mayavi import mlab

from hdf5storage import loadmat

import config as cfg

import pickle

plot = False

subjects_dir = cfg.get_subjects_dir("ds117")
os.environ['SUBJECTS_DIR'] = subjects_dir


surfer_kwargs = dict(subject="fsaverage", background="white",
                     foreground='black', cortex=("gray", -1, 6, True),
                     smoothing_steps=10, views="ventral", time_label="")

root = "/Users/hichamjanati/Dropbox/ds117-gala/"
ev_fname = op.join(root, "data/sub002-ave.fif")

results_path = op.join(root, "results")
times = mne.read_evokeds(ev_fname, verbose=False)[0].times

subjects_all = ["sub%03d" % i for i in range(1, 20) if i not in [1, 5, 16]]


stc_path = "/Users/hichamjanati/Dropbox/ds117-gala/stc"
fig_path = "/Users/hichamjanati/Dropbox/ds117-gala/fig"

subject = subjects_all[0]
# stc_fname = op.join(stc_path, subject)
# stc_all = mne.read_source_estimate(stc_fname)
with open(op.join(root, "data/vertno_ref.pkl"), "rb") as vertno_file:
    vertices = pickle.load(vertno_file)

for i, subject in enumerate(subjects_all):
    stc_fname = op.join(stc_path, subject)
    data = loadmat(op.join(stc_path, "%s-stc.mat" % subject))["SS"][0, 0][0]
    m = abs(data).max()
    clim = dict(kind='value', pos_lims=[0., 0.2 * m, m])
    stc = mne.SourceEstimate(data=data, vertices=vertices, tmin=times[0],
                             tstep=times[1] - times[0])
    if i == 0:
        stc_all = [stc]
    else:
        stc_all.append(stc)
    # stc.plot(hemi="both", clim=clim, **surfer_kwargs)
    # fig_fname = op.join(fig_path, subject + ".png")
    # mlab.savefig(fig_fname)
stc_all = sum(stc_all) / len(subjects_all)

m = abs(stc_all.data).max()
clim = dict(kind='value', pos_lims=[0., 0.8 * m, m])
stc_all.plot(hemi="both", clim=clim, **surfer_kwargs)
fig_fname = op.join(fig_path, "average.png")
mlab.savefig(fig_fname)
