from real_data.plot_brains import plot_blobs
from build_data import build_coefs
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from real_data.utils import get_labels


params = {"legend.fontsize": 19,
          "axes.titlesize": 17,
          "axes.labelsize": 16,
          "xtick.labelsize": 14,
          "ytick.labelsize": 14}
plt.rcParams.update(params)

n_subjects = 6
names = [f"Subject {i}" for i in range(1, n_subjects + 1)]
colors = plt.cm.hsv(np.linspace(0, 1, n_subjects))[:, :-1]
colors[0] = np.array([0, 0, 0])
legend_models = [Line2D([0], [0], color="w",
                        markerfacecolor=color, markersize=13,
                        linewidth=3,
                        marker="o",
                        label=name)
                 for color, name in zip(colors, names)]
f, ax = plt.subplots(1, 1, figsize=(8, 2))
ax.legend(handles=legend_models, ncol=3, frameon=False,
          labelspacing=1.5)
ax.axis("off")
plt.savefig("real_data/fig/coefs_legend.pdf")


dataset = "camcan"
seed = 0
overlap = 50
n_sources = 5

label_names = ["S_interm_prim-Jensen",
               "G_and_S_transv_frontopol",
               "S_oc_sup_and_transversal",
               "Lat_Fis-ant-Horizont",
               "S_collat_transv_post"]
labels = get_labels(label_names, annot_name="aparc.a2009s", hemi="lh")
coefs = build_coefs(n_tasks=n_subjects, overlap=overlap,
                    n_sources=n_sources, seed=seed,
                    positive=True, labels_type="any",
                    dataset=dataset)
coefs += np.random.randn(*coefs.shape) * 10

views = [dict(azimuth=-135.40236255883042,
              elevation=78.58822778745676,
              distance=437.24652099609347,
              focalpoint=np.array([0., 1.28382111, 0.]),
              roll=77.61442413729858), "lateral"]

for i, view in enumerate(views):
    fname = "real_data/fig/coefs_-v%i.png" % i
    plot_blobs(np.concatenate((coefs, coefs)), subject='fsaverage',
               save_fname=fname,
               title=None, surface="inflated", text="", hemi="lh",
               background="white",  views=view, top_vertices=2500,
               dataset=dataset, labels=labels,
               figsize=(800, 800))
