import pickle
import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
from nilearn.image import (load_img, new_img_like, mean_img, resample_img)
from nilearn import plotting

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


colors = ["gold", "purple", "forestgreen", "cornflowerblue", "indianred"]
linewidths = [4.5, 4., 3, 1.5, 1.]


if __name__ == "__main__":
    # load one fmri img as a template
    plot_all = True
    subjects = ["sub%03d" % i for i in range(1, 20) if i not in [1, 5, 16]]
    with open("data/ds117-visual/glass-brains/data.pkl", "rb") as data_file:
        data = pickle.load(data_file)
    titles = data["models"]

    if plot_all:
        for subject in subjects:
            print("Doing subject %s ..." % subject)
            img_fname = "fig/fmri/glass-%s.pdf" % subject
            display = plotting.plot_glass_brain(None)
            imgs = data[subject]
            for color, stat_img, title, lw in zip(colors, imgs, titles,
                                                  linewidths):
                display.add_contours(stat_img, colors=color,
                                     alpha=1., filled=True)
            plt.savefig(img_fname)
    else:
        s = 1
        z_fname = os.path.join(fmri_path, "sub-%s_zmap.nii" % s)
        template = load_img(z_fname, False)
        fig = plt.figure(figsize=(12, 4))
        display = plotting.plot_glass_brain(None, figure=fig)
        img_fname = "fig/fmri/average-1.pdf"
        for i, model in enumerate(data["models"]):
            if "lasso" not in model:
                average_model = []
                for subject in subjects:
                    new_img = resample_img(data[subject][i], template.affine)
                    average_model.append(new_img)
                average = mean_img(average_model)
                display.add_contours(average, colors=colors[i],
                                     alpha=0.75, filled=True)
        plt.savefig(img_fname, bbox_inches="tight")

        fig = plt.figure(figsize=(12, 4))
        display = plotting.plot_glass_brain(None, figure=fig)
        img_fname = "fig/fmri/average-2.pdf"
        for i, model in enumerate(data["models"]):
            if "remwe" not in model and "gala" not in model:
                average_model = []
                for subject in subjects:
                    new_img = resample_img(data[subject][i], template.affine)
                    average_model.append(new_img)
                average = mean_img(average_model)
                display.add_contours(average, colors=colors[i],
                                     alpha=0.75, filled=True)
        plt.savefig(img_fname, bbox_inches="tight")
