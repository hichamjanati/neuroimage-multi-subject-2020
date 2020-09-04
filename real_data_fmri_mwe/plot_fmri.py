import numpy as np
import os
from mayavi import mlab
import mne
import pandas as pd
import utils


dataset = "ds117"
spacing = "ico4"
task = "visual"
if os.path.exists("/home/parietal/"):
    data_path = "/home/parietal/hjanati/data/"
else:
    data_path = "~/Dropbox/neuro_transport/code/mtw_experiments/meg/"
    data_path += dataset + "/"
    data_path = os.path.expanduser(data_path)
subjects_dir = data_path + "subjects/"
os.environ['SUBJECTS_DIR'] = subjects_dir

metric_fname = data_path + \
    "metrics/metric_fsaverage_full_ico4_lrh.npy"
M_ = np.load(metric_fname)
M = M_ * 100
zmap_path = data_path + "fmri/"
os.environ['SUBJECTS_DIR'] = subjects_dir
fmri_path = "%s-%s/fmri/" % (dataset, task)

plot = False
surfer_kwargs = dict(subject="fsaverage", surface="inflated",
                     subjects_dir=subjects_dir,
                     views='ventral', time_unit='s', size=(800, 800),
                     smoothing_steps=6, transparent=True, alpha=0.8)
vertices_large = 2 * [np.arange(163842)]
vertices = 2 * [np.arange(2562)]

fname = "aparca2009s"
labels_fname = subjects_dir + "fsaverage/label/%s.pkl" % fname

labels_raw = mne.read_labels_from_annot("fsaverage", "aparc.a2009s",
                                        subjects_dir=subjects_dir)
labels = dict()
name = "G_oc-temp_lat-fusifor"
for l in labels_raw:
    if name in l.name:
        labels["ffg-" + l.name.split('-')[-1]] = l
hemis = ["lh", "rh"]
v1_name = "V1"
for hemi in hemis:
    v1_fname = subjects_dir + "fsaverage/label/%s.V1_exvivo.label" % hemi
    v1 = mne.read_label(v1_fname)
    labels[v1_name + "-%s" % hemi] = v1


labels_dir = "~/Dropbox/neuro_transport/code/mtw_experiments/neurosynth-labels"
labels_dir = os.path.expanduser(labels_dir)

for hemi in hemis:
    ffa_fname = labels_dir + "/ffa-%s.label" % hemi
    ffa = mne.read_label(ffa_fname)
    labels["ffa" + "-%s" % hemi] = ffa

label_names = ["ffa", "V1", "ffg"]

dfmri = pd.DataFrame()
mean_fmri_lh = np.zeros(2562)
mean_fmri_rh = np.zeros(2562)

for s in range(1, 17):
    if s < 4:
        s_ = s + 1
    elif s < 13:
        s_ = s + 2
    else:
        s_ = s + 3
    subject = "sub%03d" % s_
    print("Doing %s" % subject)
    coefs = []
    for k, (h, hemi) in enumerate(zip(["left", "right"], hemis)):
        fname = zmap_path + "sub-%d_%s_surfacic_zmap_fsaverage.npy" % (s, h)
        coefs_h = np.load(fname)
        coefs.append(coefs_h)
    coefs_ = np.hstack(coefs)
    stc = mne.SourceEstimate(coefs_, vertices_large, tmin=0, tstep=0)
    morph = mne.morph.compute_source_morph(stc, subject_from="fsaverage",
                                           spacing=4)
    stc = morph.apply(stc)
    data = stc.data
    coefs = [data[:2562], data[2562:]]
    for k, (h, hemi) in enumerate(zip(["left", "right"], hemis)):
        d = dict(subject=subject, hemi=hemi, model="fmri")
        for name in label_names:
            ratio, max_ratio = utils.count_sources(coefs[k], name, vertices,
                                                   labels, k)
            geod = utils.get_geodesic(name, labels, coefs[k],
                                      vertices, M, k)
            d[name + "_geod"] = geod
            d[name + "_count"] = ratio
            m = abs(coefs_).max()
            d[name + "_max"] = max_ratio / m
        print(f"hemi = {hemi} - max = {max_ratio} - MAX = {m}")
        dfmri = dfmri.append(d, ignore_index=True)
    mean_fmri_lh += coefs[0].flatten()
    mean_fmri_rh += coefs[1].flatten()

    if plot:
        coefs = abs(np.vstack(coefs))
        m = np.abs(coefs).max()
        lims = (0., 0.7 * m, m)
        surfer_kwargs["clim"] = dict(kind="value", lims=lims)
        stc = mne.SourceEstimate(coefs, vertices=vertices, tstep=0, tmin=1)
        f = mlab.figure(size=(800, 800))
        brain = stc.plot(hemi="both", **surfer_kwargs, background="white",
                         foreground="black", figure=f, time_label=None)
        brain.add_label(labels["ffg-lh"], alpha=0.4, color="green",
                        borders=True)
        brain.add_label(labels["ffg-rh"], alpha=0.4, color="green",
                        borders=True)
        mlab.savefig(fmri_path + "%s.eps" % subject)
        mlab.close(f)
np.save(fmri_path + "mean_lh.npy", mean_fmri_lh / 16)
np.save(fmri_path + "mean_rh.npy", mean_fmri_rh / 16)

if plot:
    mean = abs(np.vstack([mean_fmri_lh[:, None], mean_fmri_rh[:, None]])) / 16
    m = np.abs(mean).max()
    lims = (0., 0.7 * m, m)
    surfer_kwargs["clim"] = dict(kind="value", lims=lims)
    stc = mne.SourceEstimate(mean, vertices=vertices, tstep=0, tmin=1)
    f = mlab.figure(size=(800, 800))
    brain = stc.plot(hemi="both", **surfer_kwargs, background="white",
                     foreground="black", figure=f, time_label=None)
    brain.add_label(labels["ffg-lh"], alpha=0.4, color="green",
                    borders=True)
    brain.add_label(labels["ffg-rh"], alpha=0.4, color="green",
                    borders=True)
    mlab.savefig(fmri_path + "%s.eps" % subject)
    mlab.close(f)
else:
    dfmri.to_csv("stats/stats_fmri.csv")
