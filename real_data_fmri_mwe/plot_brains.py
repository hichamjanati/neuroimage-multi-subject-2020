import os
import numpy as np
from surfer import Brain
from mayavi import mlab
import mne
import warnings
from matplotlib import pyplot as plt
import pickle
from mayavi.core.module_manager import ModuleManager


def get_data_dir(dataset):
    data_path = "~/Dropbox/neuro_transport/code/"
    data_path += "mtw_experiments/meg/%s/" % dataset
    data_path = os.path.expanduser(data_path)
    subjects_dir = data_path + 'subjects/'
    os.environ['SUBJECTS_DIR'] = subjects_dir

    return data_path


def plot_source_estimate(coefs=None, subject='fsaverage', save_fname="",
                         title=None, surface="inflated", text="", hemi="lh",
                         background="white", colorbar=True, perc=0.2,
                         views="ventral", dataset="camcan", labels=[],
                         lims=None, top_vertices=None,
                         figsize=(800, 800), brain=None, spacing="ico4",
                         smoothing_steps=10):
    data_path = get_data_dir(dataset)
    subjects_dir = data_path + "subjects/"
    foreground = "black"
    if background == "black":
        foreground = "white"
    if title is None:
        title = "Source estimates - %s" % subject
    file_path = data_path + "leadfields/group_info_%s.pkl" % spacing
    group_info_file = open(file_path, "rb")
    group_info = pickle.load(group_info_file)
    if subject == "fsaverage":
        vertices_l, vertices_r = group_info["vertno_ref"]
    else:
        subject_id = group_info["subjects"].index(subject)
        vertices_l = group_info["vertno_lh"][subject_id]
        vertices_r = group_info["vertno_rh"][subject_id]
    order_l = np.argsort(vertices_l)
    vertices_l = np.sort(vertices_l)
    order_r = np.argsort(vertices_r)
    vertices_r = np.sort(vertices_r)
    n_l, n_r = group_info["n_sources"]
    verts = [vertices_l, vertices_r]
    coefs_l = coefs[:n_l][order_l].squeeze()
    coefs_r = coefs[n_l:][order_r].squeeze()
    coefs_ = np.hstack((coefs_l, coefs_r))

    m_l = np.abs(coefs_l).max()
    m_r = np.abs(coefs_r).max()
    if hemi == "lh":
        m = m_l
    elif hemi == "rh":
        m = m_r
    else:
        assert n_l + n_r == len(coefs)

        m = max(m_l, m_r)
    if m <= 0:
        warnings.warn("SourceEstimate all zero for %s " % save_fname)
    if lims is None:
        lims = (0., perc * m, m)
    if coefs_.min() < 0:
        clim = dict(kind="value", pos_lims=lims)
    else:
        clim = dict(kind="value", lims=lims)

    surfer_kwargs = dict(subject=subject, surface=surface,
                         subjects_dir=subjects_dir,
                         views=views, time_unit='s', size=(800, 800),
                         smoothing_steps=smoothing_steps, transparent=True,
                         alpha=0.8, clim=clim, colorbar=colorbar,
                         time_label=None, background=background,
                         foreground=foreground, cortex=("gray", -1, 6, True))

    stc = mne.SourceEstimate(data=coefs_.copy(), vertices=verts, tmin=0.17,
                             tstep=0.)
    # return stc, surfer_kwargs
    brain = stc.plot(hemi=hemi, **surfer_kwargs)
    engine = mlab.get_engine()
    parents = engine.scenes[0].children
    for parent in parents:
        child = parent.children[0].children[0]
        if isinstance(child, ModuleManager):
            sc_lut_manager = child.scalar_lut_manager
            sc_lut_manager.scalar_bar.number_of_labels = 8
            sc_lut_manager.scalar_bar.label_format = '%.2f'
    for label in labels:
        brain.add_label(label, color=(0., 1., 0.), alpha=1., borders=True)

    if save_fname:
        mlab.savefig(save_fname)
        f = mlab.gcf()
        mlab.close(f)

    return brain


def plot_blobs(coefs=None, subject='fsaverage', save_fname="",
               title=None, surface="inflated", text="", hemi="lh",
               background="white",  views="ventral", top_vertices=None,
               dataset="camcan", labels=[],
               brain=None, spacing="ico4",
               figsize=(800, 800), perc=0.,):
    data_path = get_data_dir(dataset)
    subjects_dir = data_path + "subjects/"
    foreground = "black"
    if background == "black":
        foreground = "white"
    n_subjects = coefs.shape[-1]
    if top_vertices is None:
        top_vertices = int((abs(coefs) > 0).sum(0).mean())
    if title is None:
        title = "Source estimates - %s" % subject
    file_path = data_path + "leadfields/group_info_%s.pkl" % spacing
    group_info_file = open(file_path, "rb")
    group_info = pickle.load(group_info_file)
    if subject == "fsaverage":
        vertices_l, vertices_r = group_info["vertno_ref"]
    else:
        subject_id = group_info["subjects"].index(subject)
        vertices_l = group_info["vertno_lh"][subject_id]
        vertices_r = group_info["vertno_rh"][subject_id]
    order_l = np.argsort(vertices_l)
    vertices_l = np.sort(vertices_l)
    order_r = np.argsort(vertices_r)
    vertices_r = np.sort(vertices_r)
    n_l, n_r = group_info["n_sources"]
    verts = [vertices_l, vertices_r]
    coefs_l = coefs[:n_l][order_l]
    coefs_r = coefs[n_l:][order_r]
    coefs_lr = [coefs_l, coefs_r]
    hemis = ["lh", "rh"]
    if hemi == "lh":
        hs = [0]
    elif hemi == "rh":
        hs = [1]
    else:
        hs = [0, 1]
    colors = plt.cm.hsv(np.linspace(0, 1, n_subjects))[:, :-1]
    scales = np.linspace(5, 10, n_subjects)[::-1]
    colors[0] = np.array([0, 0, 0])
    scales = np.array([12, 6, 9, 6, 6, 6])
    # scales = 5 * np.ones(n_subjects)
    f = mlab.figure(size=figsize)
    if brain is None:
        brain = Brain(subject, hemi, surface, subjects_dir=subjects_dir,
                      views=views, offscreen=False, background=background,
                      foreground=foreground, figure=f,
                      cortex=("gray", -1, 6, True))

    for h_index in hs:
        coefs = coefs_lr[h_index]
        threshs = np.sort(abs(coefs), axis=0)[-top_vertices]
        hemi_tmp = hemis[h_index]
        vertices = verts[h_index]
        for data, col, s, t in zip(coefs.T, colors, scales, threshs):
            surf = brain.geo[hemi_tmp]
            support = np.where(abs(data) > t)[0]
            sources = vertices[support]
            mlab.points3d(surf.x[sources], surf.y[sources],
                          surf.z[sources], color=tuple(col),
                          scale_factor=s, opacity=0.7, transparent=True)
        for label in labels:
            brain.add_label(label, color=(0., 1., 0.), alpha=1., borders=True)
    if save_fname:
        mlab.savefig(save_fname)
        f = mlab.gcf()
        mlab.close(f)

    return brain


# if __name__ == "__main__":
#     dataset = "camcan"
#     surface = "white"
#     spacing = "ico4"
#     data_path = "~/Dropbox/neuro_transport/code/"
#     data_path += "mtw_experiments/meg/%s/" % dataset
#     data_path = os.path.expanduser(data_path)
#     subjects_dir = data_path + "/subjects/"
#     subject = "fsaverage"
#     hemi = "both"
#     labels_type = "any"
#     spacing = 'ico4'
#     labels_lh = np.load(data_path +
#                         "label/labels-%s-%s-lh.npy" % (labels_type, spacing))
#     labels_rh = np.load(data_path +
#                         "label/labels-%s-%s-rh.npy" % (labels_type, spacing))
#     label_idx = 0
#     n_l = labels_lh.shape[1]
#     pos_l = np.where(labels_lh[label_idx])[0]
#     n_r = labels_rh.shape[1]
#     pos_r = np.where(labels_rh[label_idx])[0] + n_l
#     coefs = np.zeros((n_l + n_r, 2))
#     coefs[pos_l, 0] = np.random.rand(len(pos_l))
#     coefs[pos_r, 1] = np.random.rand(len(pos_r))
#
#     label_names = ["Brodmann.41", "Brodmann.42"]
#     label_names = ["auditory-cortex"]
#     labels_lh = utils.get_labels(label_names, annot_name="neurosynth", hemi="lh",
#                                  subjects_dir=subjects_dir)
#     labels_rh = utils.get_labels(label_names, annot_name="neurosynth", hemi="rh",
#                                  subjects_dir=subjects_dir)
#     labels = labels_lh + labels_rh
#     if 0:
#         b = plot_blobs(coefs, subject=subject, hemi=hemi, top_vertices=50)
#     else:
#         b = plot_source_estimate(coefs.mean(1), subject=subject, hemi=hemi,
#                                  labels=labels, views="lateral",
#                                  dataset=dataset, spacing=spacing, perc=0.3,
#                                  )
