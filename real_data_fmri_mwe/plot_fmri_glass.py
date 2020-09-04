import numpy as np
import os
from nilearn.image import load_img
from nilearn import plotting
from nilearn.image import math_img
from matplotlib import pyplot as plt


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
    data_path = "/home/parietal/hjanati/data/"
else:
    data_path = "~/Dropbox/neuro_transport/code/mtw_experiments/meg/"
    data_path += dataset + "/"
    data_path = os.path.expanduser(data_path)
subjects_dir = data_path + "subjects/"
os.environ['SUBJECTS_DIR'] = subjects_dir

zmap_path = data_path + "fmri/"
os.environ['SUBJECTS_DIR'] = subjects_dir
fmri_path = os.path.join(data_path, "fmri/")

plot = False
surfer_kwargs = dict(subject="fsaverage", surface="inflated",
                     subjects_dir=subjects_dir,
                     views='ventral', time_unit='s', size=(800, 800),
                     smoothing_steps=6, transparent=True, alpha=0.8)
vertices_large = 2 * [np.arange(163842)]
hemis = ["lh", "rh"]


imgs = []
for s in range(1, 17):
    z_fname = os.path.join(fmri_path, "sub-%s_zmap.nii" % s)
    img = load_img(z_fname, False)
    img_fname = "fig/fmri/glass-%s.pdf" % s
    title = "Subject %s" % s
    imgs.append(img)
    max_ = 6
    plotting.plot_glass_brain(img, threshold=max_, output_file=img_fname,

                              colorbar=True, title=title,
                              plot_abs=True)
# result_img = math_img("np.sum(imgs, 0)", imgs=imgs)
