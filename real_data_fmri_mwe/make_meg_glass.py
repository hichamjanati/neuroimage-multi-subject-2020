import os
import os.path as op
import pickle
import numpy as np
import mne

from scipy.io import loadmat

from joblib import Parallel, delayed

import pickle

from nilearn.image import (load_img, coord_transform, new_img_like, math_img,
                           smooth_img)
from nilearn import plotting

from matplotlib import pyplot as plt

import config as cfg


params = {"legend.fontsize": 16,
          "axes.titlesize": 16,
          "axes.labelsize": 15,
          "xtick.labelsize": 15,
          "ytick.labelsize": 15,
          "font.size": 20}
plt.rcParams.update(params)

dataset = "ds117"
spacing = "ico4"
task = "visual"
if os.path.exists("/home/parietal/"):
    data_path = "/storage/store/work/hjanati/datasets/mne-biomag-group-demo/"
else:
    data_path = "~/Dropbox/neuro_transport/code/mtw_experiments/meg/"
    data_path += dataset + "/"
    data_path = os.path.expanduser(data_path)
subjects_dir = "/storage/store/work/agramfort/mne-biomag-group-demo/"
subjects_dir += "subjects"
os.environ['SUBJECTS_DIR'] = subjects_dir

os.environ['SUBJECTS_DIR'] = subjects_dir
fmri_path = os.path.join(data_path, "fmri/")

file_path = "~/data/ds117/leadfields/group_info_ico4.pkl"
save_path = "data/ds117-visual/glass-brains"

file_path = op.expanduser(file_path)
group_info_file = open(file_path, "rb")
group_info = pickle.load(group_info_file)
vertices = group_info["vertno_ref"]
n_sources = vertices[0].size
models = ["adalasso", "remwe", "lasso", "gala"]
titles = ["fMRI"] + models
colors = ["gold", "purple", "forestgreen", "cornflowerblue", "indianred"]
linewidths = [2., 2.5, 3., 3.5, 4.]

mni_coords_all = []
for hemi_indx in [0, 1]:
    mni_coords = mne.vertex_to_mni(vertices[hemi_indx], hemis=hemi_indx,
                                   subject="fsaverage",
                                   subjects_dir=subjects_dir)
    mni_coords_all.append(mni_coords)

all_data = dict(models=titles)


def make_nifti(s):
    if s < 4:
        s_ = s + 1
    elif s < 14:
        s_ = s + 2
    else:
        s_ = s + 3
    subject = "sub%03d" % s_
    print(subject)
    imgs = []
    z_fname = os.path.join(fmri_path, "sub-%s_zmap.nii" % s)
    fmri = load_img(z_fname, False)
    fmri_data = abs(fmri.get_data())
    inv_trans = np.linalg.inv(fmri.affine)
    # img_fname = "fig/fmri/glass-%s.pdf" % s
    fmri_data_max = np.zeros_like(fmri_data)

    all_voxels = []
    for hemi_indx in [0, 1]:
        mni_coords = mni_coords_all[hemi_indx]
        mni_x, mni_y, mni_z = mni_coords.T
        vx, vy, vz = coord_transform(mni_x, mni_y, mni_z, inv_trans)
        voxels = np.vstack((vx, vy, vz)).astype(int)
        all_voxels.append(voxels.T)
        fmri_data_hemi = fmri_data[voxels[0], voxels[1], voxels[2]]
        voxel_max = np.where(fmri_data == fmri_data_hemi.max())
        if len(voxel_max[0]) > 1:
            voxel_max = (voxel_max[0][0], voxel_max[1][0], voxel_max[2][0])
        fmri_data_max[voxel_max] = fmri_data_hemi.max()
    fmri_img_max = new_img_like(fmri, fmri_data_max)
    fmri_img_max = smooth_img(fmri_img_max, fwhm=1.5)

    imgs.append(fmri_img_max)
    gala_root = "/storage/store/work/hjanati/datasets/ds117-gala"
    with open(op.join(gala_root, "data/vertno_ref.pkl"), "rb") as vertno_file:
        vertices = pickle.load(vertno_file)
    n_l = vertices[0].size
    for jj, model in enumerate(models):
        meg_vol = np.zeros_like(fmri_data)
        if "gala" in model:
            stc_path = op.join(gala_root, "stc")
            stc_fname = op.join(stc_path, "%s-stc.mat" % subject)
            coefs = loadmat(stc_fname)["SS"][0, 0][0][:, 0]
            coefs1, coefs2 = coefs[:n_l], coefs[n_l:]
            coefs_hemis = [coefs1, coefs2]
        else:
            coefs_fname = save_path + "/%s/%s.npy" % (model, subject)
            coefs = np.load(coefs_fname).flatten()
            coefs_hemis = [coefs[:n_sources], coefs[n_sources:]]
            # loc_max = np.argmax(abs(coefs))
            # hemi_indx = int(loc_max >= n_sources)
        for hemi_indx in [0, 1]:
            max_location = np.argmax(abs(coefs_hemis[hemi_indx]))
            v_x, v_y, v_z = all_voxels[hemi_indx][max_location]
            meg_vol[v_x, v_y, v_z] = \
                abs(coefs_hemis[hemi_indx][max_location])
        # for jj, v in enumerate(unique_voxels):
        #     support = np.where(((voxels - v[None, :]) == 0).all(1))[0]
        #     values_in_v = coefs_hemis[i][support]
        #     meg_vol[v[0], v[1], v[2]] = abs(values_in_v).max()
        meg_img = new_img_like(fmri, meg_vol)
        meg_img = smooth_img(meg_img, fwhm=linewidths[jj])
        imgs.append(meg_img)
    return subject, imgs


if __name__ == "__main__":
    iterator = (delayed(make_nifti)(s) for s in range(1, 17))
    out = Parallel(n_jobs=16)(iterator)
    subjects, imgs = zip(*out)
    for s, img in zip(subjects, imgs):
        all_data[s] = img
    with open("data/ds117-visual/glass-brains/data.pkl", "wb") as data_file:
        pickle.dump(all_data, data_file)
