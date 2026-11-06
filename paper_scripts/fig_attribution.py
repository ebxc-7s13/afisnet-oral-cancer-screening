"""Fig. 9 (attribution): one correctly classified cell per class, input / occlusion map / Grad-CAM++.
Occlusion maps: occlusion_maps_highres.npz (final AFiS-Net, QC-clean, seed 42, out-of-fold).
Grad-CAM++: the saved overlay of the same model and image (explainability_final/gallery); the heat values are recovered
by inverting the overlay (0.44 * viridis(heat) + 0.55 * input; mean colour residual ~4/255) and drawn in the same style.
Usage: python fig_attribution.py <afis_results dir> <out prefix>"""
import glob
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colormaps
from PIL import Image
from scipy.ndimage import gaussian_filter, binary_fill_holes, binary_opening, label as cc_label
from skimage import filters, measure

E0, out = sys.argv[1], sys.argv[2]
PICKS = [("correct_cancer", 11, "Cancer"), ("correct_normal", 13, "Normal")]
d = np.load(E0 + "/explainability_final_highres/occlusion_maps_highres.npz")
V = colormaps["viridis"](np.linspace(0, 1, 256))[:, :3] * 255
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5, "pdf.fonttype": 42})


def stretch(b):
    lo, hi = np.percentile(b, 1), np.percentile(b, 99.7)
    return np.clip((b - lo) / (hi - lo + 1e-6), 0, 1)


def cell_mask(b):
    g = gaussian_filter(b.max(2), 2)
    m = binary_fill_holes(binary_opening(g > filters.threshold_otsu(g), iterations=2))
    lab, nl = cc_label(m)
    return lab == (1 + np.argmax(np.bincount(lab.ravel())[1:])) if nl > 1 else m


def overlay(b, s, sigma):
    s = gaussian_filter(np.clip(s, 0, None), sigma)
    s = (s - s.min()) / (s.max() - s.min() + 1e-8)
    heat = colormaps["inferno"](s)[..., :3]; a = np.clip((s - 0.25) / 0.75, 0, 1)[..., None] * 0.8
    return np.clip(b * (1 - a) + heat * a, 0, 1)


def gradcampp(cat, k):
    inp = np.asarray(Image.open(glob.glob(f"{E0}/explainability_final/gallery/{cat}/{k:02d}_*_input.png")[0]).convert("RGB"), float)
    g = np.asarray(Image.open(glob.glob(f"{E0}/explainability_final/gallery/{cat}/{k:02d}_*_multilevel_gradcampp.png")[0]).convert("RGB"), float)
    col = (g - 0.55 * inp) / 0.44
    dist = ((col.reshape(-1, 1, 3) - V[None]) ** 2).sum(2)
    res = np.sqrt(dist.min(1)).mean()
    return dist.argmin(1).reshape(g.shape[:2]) / 255.0, res


fig, axes = plt.subplots(2, 3, figsize=(3.7, 2.62), gridspec_kw={"wspace": 0.04, "hspace": 0.06})
fig.subplots_adjust(left=0.08, right=0.86, top=0.91, bottom=0.02)
for r, (cat, k, lab) in enumerate(PICKS):
    base = d[f"{cat}_{k:02d}_input"].astype(np.float32); occ = d[f"{cat}_{k:02d}"]
    b = stretch(base); m = cell_mask(b)
    gc, res = gradcampp(cat, k)
    print(cat, k, "Grad-CAM++ inversion residual", round(res, 2))
    panels = [b, overlay(b, occ, 2), overlay(b, gc, 1)]
    for c, im in enumerate(panels):
        ax = axes[r, c]; ax.imshow(im); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        if c:
            for cc in measure.find_contours(m.astype(float), 0.5):
                if len(cc) > 40:
                    ax.plot(cc[:, 1], cc[:, 0], c="#7fdbff", lw=0.6)
    axes[r, 0].set_ylabel(lab, fontsize=8.5)
for c, t in enumerate(["Input", "Occlusion", "Grad-CAM++"]):
    axes[0, c].set_title(t, fontsize=7.5, pad=3)
cax = fig.add_axes([0.88, 0.12, 0.022, 0.7])
cb = fig.colorbar(plt.cm.ScalarMappable(cmap="inferno"), cax=cax)
cb.set_ticks([0, 1]); cb.set_ticklabels(["low", "high"]); cb.ax.tick_params(labelsize=6.5, length=2)
fig.savefig(out + ".png", dpi=600); fig.savefig(out + ".pdf")
