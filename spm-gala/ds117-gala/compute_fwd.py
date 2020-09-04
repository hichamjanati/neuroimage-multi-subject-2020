import os
import os.path as op

import numpy as np
import scipy

from scipy import io

import mne
from mne.parallel import parallel_func

from groupmne import group_model

import config as cfg

import pickle

plot = False
save_dir = "/storage/store/work/hjanati/datasets/ds117-gala"
root_dir = "/storage/store/work/hjanati/datasets/mne-biomag-group-demo"
# fwd_path = op.expanduser("forwards/")
# ave_path = op.expanduser("ave/")
# meg_path = op.expanduser("MEG/")

# subjects_dir = op.expanduser("subjects/")
subjects_dir = op.join(root_dir, "subjects")
meg_path = op.join(root_dir, "MEG")

os.environ['SUBJECTS_DIR'] = subjects_dir


resolution = 4
spacing = "ico%d" % resolution
subject = "fsaverage"

# src_ref = group_model.get_src_reference(spacing=spacing,
#                                         subjects_dir=subjects_dir)

src_ref = mne.setup_source_space(subject=subject,
                                 spacing=spacing,
                                 subjects_dir=subjects_dir,
                                 add_dist=False)


def compute_fwd(subject, src_ref, info, trans_fname, bem_fname,
                meg=True, eeg=False, mindist=2, subjects_dir=None,
                n_jobs=1):
    """Morph the source space of fsaverage to a subject.

    Parameters
    ----------
    subject : str
        Name of the reference subject.
    src_ref : instance of SourceSpaces
        Source space of the reference subject. See `get_src_reference.`
    info : str | instance of mne.Info
        Instance of an MNE info file or path to a raw fif file.
    trans_fname : str
        Path to the trans file of the subject.
    bem_fname : str
        Path to the bem solution of the subject.
    mindist : float
        Safety distance from the outer skull. Sources below `mindist` will be
        discarded in the forward operator.
    subjects_dir : str
        Path to the freesurfer `subjects` directory.

    """
    print("Processing subject %s" % subject)

    src = mne.morph_source_spaces(src_ref, subject_to=subject,
                                  subjects_dir=subjects_dir)
    bem = mne.read_bem_solution(bem_fname)
    fwd = mne.make_forward_solution(info, trans=trans_fname, src=src,
                                    bem=bem, meg=meg, eeg=eeg,
                                    mindist=mindist,
                                    n_jobs=n_jobs)
    return fwd

###################################################################
# the function group_model.compute_fwd morphs the source space src_ref to the
# surface of each subject by mapping the sulci and gyri patterns
# and computes their forward operators.


subjects = cfg.get_subjects_list("ds117")
ave_name_s = [cfg.get_ave_fname("ds117", s) for s in subjects]
trans_fname_s = [cfg.get_trans_fname("ds117", s) for s in subjects]
bem_fname_s = [cfg.get_bem_fname("ds117", s) for s in subjects]
fwd_fname_s = [op.join(save_dir, "%s-fwd.fif" % s) for s in subjects]

n_jobs = 16
parallel, run_func, _ = parallel_func(compute_fwd, n_jobs=n_jobs)

fwds_free = parallel(run_func(s, src_ref, info, trans, bem, mindist=3)
                     for s, info, trans, bem in zip(subjects, ave_name_s,
                                                    trans_fname_s,
                                                    bem_fname_s))
fwds = []
for fwd_fname, fwd in zip(fwd_fname_s, fwds_free):
    fwd = mne.convert_forward_solution(fwd, surf_ori=True,
                                       force_fixed=True,
                                       use_cps=True)
    fwds.append(fwd)
    # mne.write_forward_solution(fwd_fname, fwd, overwrite=True)

gains, group_info = group_model._group_filtering(fwds_free, src_ref,
                                                 noise_covs=None)
vertno_ref = group_info["vertno_ref"]

with open(op.join(save_dir, "data/vertno_ref.pkl"), "wb") as vertno_file:
    pickle.dump(vertno_ref, vertno_file)

n_sources = 2562
# assert gains.shape[-1] == 2 * n_sources


for ii, subject in enumerate(subjects):
    data_dict = dict(gain=gains[ii])
    io.savemat(op.join(save_dir, "data/%s-gain.mat" % subject), data_dict)
    cov = mne.read_cov(cfg.get_cov_fname("ds117", subject))
    cov.save(op.join(save_dir, "%s-cov.fif" % subject))
adjacency = []
coords = []
for i in range(2):
    # new_gain = fwds[i]["sol"]["data"][:, permutations[i]]
    # check that permutated gains match with the ones returned by groupmne
    # assert abs(gains[i] - new_gain).max() == 0
    tris = src_ref[0]["use_tris"]
    vertno = src_ref[0]["vertno"]
    points = src_ref[0]["rr"][vertno]
    A = mne.surface.mesh_dist(tris, points).toarray()
    A = A[vertno_ref[i]][:, vertno_ref[i]]
    points = points[vertno_ref[i]]
    A[A != 0] = 1.
    adjacency.append(A)
    coords.append(points)

graph = scipy.sparse.block_diag(adjacency)
coords = np.concatenate(coords)
data_dict = dict(graph=graph, Vertices=coords)
io.savemat(op.join(save_dir, "data/graph.mat"), data_dict)

if plot:
    # see if vertices correspond across subjects
    # and if they're on the right hemi
    vertex = 5000
    for i, ev_fname in enumerate(ave_name_s):
        data = gains[i, :, vertex]
        info = mne.read_evokeds(ev_fname)[0].info
        evoked = mne.EvokedArray(data[:, None], info=info)
        evoked.plot_topomap()
