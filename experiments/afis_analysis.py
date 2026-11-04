# =============================================================================
# afis_analysis.py  --  analyses, statistics, figures and galleries
# Every function is idempotent and writes under RESULTS/<section>/.
# =============================================================================
import os
import io
import sys
import json
import math
import glob
import time
import shutil
import zipfile
import zlib
import warnings
import subprocess
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

import afis_core as A

warnings.filterwarnings("ignore")
IMG_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
CLASSES = ("normal", "cancer")
DPI = 300
N_GALLERY = 15          # >= 10 candidates per comparison, as requested
SHEET_DPI = 120        # contact sheets are for browsing; every panel is also saved as a full-resolution PNG
plt.rcParams.update({"font.size": 10, "axes.titlesize": 10, "savefig.bbox": "tight"})


def _dev():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _list_images(d):
    d = A.ci_dir(d) if d else None            # class folders may be 'Cancer'/'Normal'
    if not d:
        return []
    return sorted(os.path.join(d, f) for f in os.listdir(d) if f.lower().endswith(IMG_EXTS))


def _savefig(fig, path_noext, dpi=None):
    Path(path_noext).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path_noext) + ".png", dpi=dpi or DPI)
    fig.savefig(str(path_noext) + ".pdf")
    plt.close(fig)


def _save_img(arr01_or_pil, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if isinstance(arr01_or_pil, Image.Image):
        arr01_or_pil.save(path)
    else:
        Image.fromarray((np.clip(arr01_or_pil, 0, 1) * 255).astype(np.uint8)).save(path)


def _done(flag):
    return (A.RESULTS / "_done" / flag).exists()


def _mark(flag):
    p = A.RESULTS / "_done" / flag
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(time.strftime("%Y-%m-%d %H:%M:%S"))


def _df_to_latex(df, path, caption, label):
    def cell(v):
        if isinstance(v, (float, np.floating)):
            v = float(v)
            return "" if np.isnan(v) else (f"{v:.2e}" if (v != 0 and abs(v) < 1e-3) else f"{v:.4g}")
        v = str(v)
        return v if "$" in v else v.replace("%", "\\%").replace("_", "\\_")
    head = " & ".join(cell(c) for c in df.columns) + " \\\\"
    lines = [" & ".join(cell(v) for v in row) + " \\\\" for row in df.itertuples(index=False)]
    body = ("\\begin{tabular}{" + "l" * len(df.columns) + "}\n\\toprule\n" + head + "\n\\midrule\n"
            + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")
    tex = ("\\begin{table}[htbp]\n\\centering\n\\caption{" + caption + "}\n\\label{" + label + "}\n"
           "\\footnotesize\n" + body + "\\end{table}\n")
    Path(path).write_text(tex)


def real_pool(paths):
    """Same pooling as kfold_classification.pool_real_data (basename dedup)."""
    seen = {}
    for split in ("train", "val", "test"):
        for c in CLASSES:
            for p in _list_images(os.path.join(paths[split], c)):
                b = os.path.basename(p)
                if b not in seen:
                    seen[b] = (p, c, split)
    return seen


def class_sets(paths, key):
    root = paths.get(key) if key in paths else None
    if key == "baseline_selected_qc":
        root = str(A.RESULTS / "baseline_qc" / "selected")
        if "synthetic_selected_baseline" in paths:
            root = paths["synthetic_selected_baseline"]
    if key == "synthetic_all_baseline" and root is None:
        gen = A.RESULTS / "baseline_qc" / "generated" / "final_generated"
        root = str(gen) if gen.exists() else None
    return {c: _list_images(os.path.join(root, c)) if root else [] for c in CLASSES}


# =============================================================================
# 0. Census
# =============================================================================
def census(paths):
    C, K = A.import_pipeline(paths)
    out = A.RESULTS / "tables"; out.mkdir(parents=True, exist_ok=True)
    rows = []
    for split in ("train", "val", "test"):
        for c in CLASSES:
            rows.append(dict(set=f"real/{os.path.basename(paths[split])}", cls=c,
                             n=len(_list_images(os.path.join(paths[split], c)))))
    pool = real_pool(paths)
    for c in CLASSES:
        names = [b for b, v in pool.items() if v[1] == c]
        groups = defaultdict(int)
        for b in names:
            groups[K.region_group(b)] += 1
        fr = np.array(list(groups.values())) if groups else np.array([0])
        rows.append(dict(set="real/pooled (CV)", cls=c, n=len(names), regions=len(groups),
                         frames_per_region_mean=round(float(fr.mean()), 2),
                         frames_per_region_median=float(np.median(fr)),
                         frames_per_region_max=int(fr.max())))
    for key in ("synthetic_selected", "synthetic_all_ntp", "synthetic_all_baseline", "synthetic_selected_baseline"):
        if key in paths:
            for c in CLASSES:
                rows.append(dict(set=key, cls=c, n=len(_list_images(os.path.join(paths[key], c)))))
    if "screening" in paths:
        for g in sorted(os.listdir(paths["screening"])):
            d = os.path.join(paths["screening"], g)
            if os.path.isdir(d):
                rows.append(dict(set="screening", cls=g, n=len(_list_images(d))))
    df = pd.DataFrame(rows)
    df.to_csv(out / "census.csv", index=False)
    sizes = defaultdict(int)
    for b, (p, c, s) in list(pool.items())[:200]:
        with Image.open(p) as im:
            sizes[f"{im.size[0]}x{im.size[1]} {im.mode}"] += 1
    (out / "census_image_sizes.json").write_text(json.dumps(sizes, indent=1))
    print(df.to_string(index=False))
    print("Image sizes (first 200 real images):", dict(sizes))
    return df


# =============================================================================
# 1. DINOv2 embeddings (fold-aware selection E6 + memorisation analysis E8)
# =============================================================================
def _dino(device):
    import torch
    if A.SMOKE:
        from transformers import Dinov2Config, Dinov2Model
        torch.manual_seed(0)
        m = Dinov2Model(Dinov2Config(hidden_size=32, num_hidden_layers=1, num_attention_heads=2,
                                     intermediate_size=64, image_size=224, patch_size=14))
        return m.to(device).eval(), None
    from transformers import AutoModel, AutoImageProcessor
    name = "facebook/dinov2-base"     # same backbone as the paper's QC run
    return AutoModel.from_pretrained(name).to(device).eval(), AutoImageProcessor.from_pretrained(name)


def _embed(paths_list, model, proc, device, bs=32):
    import torch
    import torch.nn.functional as F
    out = []
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    for i in range(0, len(paths_list), bs):
        ims = [Image.open(p).convert("RGB") for p in paths_list[i:i + bs]]
        if proc is not None:
            px = proc(images=ims, return_tensors="pt")["pixel_values"]
        else:
            px = torch.stack([torch.from_numpy(np.asarray(im.resize((224, 224), Image.BICUBIC),
                                                          dtype=np.float32) / 255.).permute(2, 0, 1) for im in ims])
            px = (px - mean) / std
        with torch.no_grad():
            o = model(pixel_values=px.to(device))
            e = o.pooler_output if getattr(o, "pooler_output", None) is not None else o.last_hidden_state[:, 0]
        out.append(F.normalize(e.float(), dim=1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 1), np.float32)


def embeddings(paths, force=False):
    f = A.RESULTS / "embeddings" / "dino_embeddings.npz"
    if f.exists() and not force:
        print("DINOv2 embeddings already computed."); return f
    f.parent.mkdir(parents=True, exist_ok=True)
    dev = _dev(); model, proc = _dino(dev)
    pool = real_pool(paths)
    rn = sorted(pool); rp = [pool[b][0] for b in rn]; rl = [pool[b][1] for b in rn]
    data = dict(real_names=np.array(rn), real_labels=np.array(rl), real_emb=_embed(rp, model, proc, dev))
    for key, tag in (("synthetic_selected", "sel"), ("synthetic_all_ntp", "ntp"),
                     ("synthetic_all_baseline", "base")):
        sets = class_sets(paths, key)
        ps = sets["normal"] + sets["cancer"]
        if ps:
            data[f"{tag}_names"] = np.array([os.path.basename(p) for p in ps])
            data[f"{tag}_paths"] = np.array(ps)
            data[f"{tag}_labels"] = np.array(["normal"] * len(sets["normal"]) + ["cancer"] * len(sets["cancer"]))
            data[f"{tag}_emb"] = _embed(ps, model, proc, dev)
    data["real_paths"] = np.array(rp)
    np.savez_compressed(f, **data)
    del model
    print(f"Saved DINOv2 embeddings -> {f}")
    return f


# =============================================================================
# 2. GAN distribution metrics with uncertainty (E8)
# =============================================================================
def _inception_features(paths_list, device, bs=50):
    import torch
    from torchmetrics.image.fid import FrechetInceptionDistance
    fid = FrechetInceptionDistance(feature=2048).to(device)
    feats = []
    for i in range(0, len(paths_list), bs):
        arr = np.stack([np.asarray(Image.open(p).convert("RGB").resize((299, 299), Image.BICUBIC))
                        for p in paths_list[i:i + bs]])
        t = torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous().to(device)
        with torch.no_grad():
            feats.append(fid.inception(t).double().cpu().numpy())
    return np.concatenate(feats)


def _fid(f1, f2):
    """Frechet distance; trace term via eigh(sqrt(S1) S2 sqrt(S1)) in float64."""
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    a = torch.as_tensor(f1, dtype=torch.float64, device=dev); b = torch.as_tensor(f2, dtype=torch.float64, device=dev)
    mu1, mu2 = a.mean(0), b.mean(0)
    s1 = torch.cov(a.T); s2 = torch.cov(b.T)
    w, V = torch.linalg.eigh(s1)
    sq1 = (V * torch.sqrt(torch.clamp(w, min=0))) @ V.T
    m = sq1 @ s2 @ sq1
    tr = torch.sqrt(torch.clamp(torch.linalg.eigvalsh((m + m.T) / 2), min=0)).sum()
    return float(((mu1 - mu2) ** 2).sum() + torch.trace(s1) + torch.trace(s2) - 2 * tr)


def _kid_subsets(fr, fg, n_sub=100, size=100, rng=None, idx_r=None):
    rng = rng or np.random.default_rng(0)
    d = fr.shape[1]; size = min(size, len(fr), len(fg)); vals = []; R = []
    for s in range(n_sub):
        ir = idx_r[s] if idx_r is not None else rng.choice(len(fr), size, replace=False)
        ig = rng.choice(len(fg), size, replace=False)
        x, y = fr[ir], fg[ig]
        kxx = (x @ x.T / d + 1) ** 3; kyy = (y @ y.T / d + 1) ** 3; kxy = (x @ y.T / d + 1) ** 3
        m = size
        mmd = ((kxx.sum() - np.trace(kxx)) / (m * (m - 1)) + (kyy.sum() - np.trace(kyy)) / (m * (m - 1))
               - 2 * kxy.mean())
        vals.append(mmd); R.append(ir)
    return np.array(vals), R


def _prdc(real, fake, k=3):
    from sklearn.metrics import pairwise_distances
    rr = pairwise_distances(real, real); np.fill_diagonal(rr, np.inf)
    ff = pairwise_distances(fake, fake); np.fill_diagonal(ff, np.inf)
    rad_r = np.sort(rr, 1)[:, k - 1]; rad_f = np.sort(ff, 1)[:, k - 1]
    rf = pairwise_distances(real, fake)
    precision = float((rf <= rad_r[:, None]).any(0).mean())
    recall = float((rf.T <= rad_f[:, None]).any(0).mean())
    density = float((rf <= rad_r[:, None]).sum(0).mean() / k)
    coverage = float((rf.min(1) <= rad_r).mean())
    return dict(precision=precision, recall=recall, density=density, coverage=coverage)


def gan_metrics(paths, n_boot=100, force=False):
    out = A.RESULTS / "gan_metrics"
    if _done("gan_metrics") and not force:
        print("GAN metrics already computed."); return
    out.mkdir(parents=True, exist_ok=True)
    dev = _dev()
    real = {c: _list_images(os.path.join(paths["train"], c)) for c in CLASSES}
    sets = {"NTP_selected": class_sets(paths, "synthetic_selected"),
            "NTP_all": class_sets(paths, "synthetic_all_ntp"),
            "Baseline_all": class_sets(paths, "synthetic_all_baseline"),
            "Baseline_selected": class_sets(paths, "baseline_selected_qc")}
    sets = {k: v for k, v in sets.items() if all(len(v[c]) >= 10 for c in CLASSES)}
    if not sets:
        print("No generated sets with >=10 images per class; skipping GAN metrics."); return
    rows, delta_rows = [], []
    feats = {}
    for c in CLASSES:
        fr = _inception_features(real[c], dev)
        feats[("real", c)] = fr
        for name, s in sets.items():
            feats[(name, c)] = _inception_features(s[c], dev)
    rng = np.random.default_rng(2026)
    for c in CLASSES:
        fr = feats[("real", c)]
        boot_idx_r = [rng.integers(0, len(fr), len(fr)) for _ in range(n_boot)]
        kid_idx = [rng.choice(len(fr), min(100, len(fr)), replace=False) for _ in range(100)]
        per_set = {}
        for name in sets:
            fg = feats[(name, c)]
            fid = _fid(fr, fg)
            bidx_g = [np.random.default_rng(1000 + b).integers(0, len(fg), len(fg)) for b in range(n_boot)]
            boots = np.array([_fid(fr[boot_idx_r[b]], fg[bidx_g[b]]) for b in range(n_boot)])
            kid, _ = _kid_subsets(fr, fg, rng=np.random.default_rng(7), idx_r=kid_idx)
            pr = _prdc(fr, fg if len(fg) <= 1500 else fg[:1500])
            per_set[name] = dict(boots=boots, kid=kid)
            rows.append(dict(cls=c, set=name, n_real=len(fr), n_gen=len(fg), FID=fid,
                             FID_CI_low=np.percentile(boots, 2.5), FID_CI_high=np.percentile(boots, 97.5),
                             KID_mean=kid.mean(), KID_std=kid.std(), **pr))
        if "NTP_all" in per_set and "Baseline_all" in per_set:
            dF = per_set["NTP_all"]["boots"] - per_set["Baseline_all"]["boots"]
            dK = per_set["NTP_all"]["kid"] - per_set["Baseline_all"]["kid"]
            delta_rows.append(dict(cls=c, comparison="NTP_all - Baseline_all",
                                   dFID_mean=dF.mean(), dFID_CI_low=np.percentile(dF, 2.5),
                                   dFID_CI_high=np.percentile(dF, 97.5),
                                   dKID_mean=dK.mean(), dKID_sd=dK.std(),
                                   frac_subsets_NTP_better_KID=float((dK < 0).mean())))
    df = pd.DataFrame(rows); df.to_csv(out / "gan_metrics_bootstrap.csv", index=False)
    if delta_rows:
        pd.DataFrame(delta_rows).to_csv(out / "ntp_vs_baseline_paired.csv", index=False)
    show = df.copy()
    show["FID [95% CI]"] = show.apply(lambda r: f"{r.FID:.2f} [{r.FID_CI_low:.2f}, {r.FID_CI_high:.2f}]", axis=1)
    show["KID"] = show.apply(lambda r: f"{r.KID_mean:.4f} $\\pm$ {r.KID_std:.4f}", axis=1)
    show = show[["cls", "set", "n_real", "n_gen", "FID [95% CI]", "KID", "precision", "recall", "density", "coverage"]]
    _df_to_latex(show.round(3), out / "table_gan_metrics.tex",
                 "Distributional quality of generated sets (Inception-v3 features; FID with 95\\% bootstrap CI, "
                 "KID mean $\\pm$ s.d. over 100 subsets; precision/recall/density/coverage, $k=3$).",
                 "tab:gan_ci")
    print(df.round(4).to_string(index=False))
    _mark("gan_metrics")


# =============================================================================
# 3. Memorisation / nearest-neighbour analysis + NN galleries (E8)
# =============================================================================
def memorisation(paths, force=False):
    out = A.RESULTS / "memorisation"
    if _done("memorisation") and not force:
        print("Memorisation analysis already done."); return
    f = A.RESULTS / "embeddings" / "dino_embeddings.npz"
    if not f.exists():
        print("Embeddings missing; run embeddings() first."); return
    C, K = A.import_pipeline(paths)
    E = dict(np.load(f, allow_pickle=True))
    out.mkdir(parents=True, exist_ok=True)
    train_names = {os.path.basename(p) for c in CLASSES for p in _list_images(os.path.join(paths["train"], c))}
    rn = E["real_names"]; rl = E["real_labels"]; R = E["real_emb"]; rp = E["real_paths"]
    in_train = np.array([n in train_names for n in rn])
    regions = np.array([K.region_group(n) for n in rn])
    rows, summary = [], []
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ci, c in enumerate(CLASSES):
        rmask = (rl == c) & in_train
        Rc, Rnames, Rreg, Rpaths = R[rmask], rn[rmask], regions[rmask], rp[rmask]
        sim = Rc @ Rc.T
        same = Rreg[:, None] == Rreg[None, :]
        sim[same] = -np.inf                     # exclude same-region near-duplicate frames
        rr = sim.max(1)
        ax = axes[ci]
        ax.hist(rr, bins=30, alpha=0.5, density=True, label="real $\\rightarrow$ real (other region)")
        q95 = float(np.percentile(rr, 95))
        for tag, lab in (("sel", "QC-selected synthetic"), ("ntp", "all NTP synthetic"), ("base", "all baseline synthetic")):
            if f"{tag}_emb" not in E:
                continue
            m = E[f"{tag}_labels"] == c
            S = E[f"{tag}_emb"][m]; Sn = E[f"{tag}_names"][m]; Sp = E[f"{tag}_paths"][m]
            if len(S) == 0:
                continue
            ss = S @ Rc.T; nn = ss.argmax(1); smax = ss.max(1)
            ax.hist(smax, bins=30, alpha=0.5, density=True, label=f"{lab} $\\rightarrow$ real")
            summary.append(dict(cls=c, set=lab, n=len(S), nn_sim_median=float(np.median(smax)),
                                nn_sim_max=float(smax.max()), real_real_q95=q95,
                                frac_above_real_q95=float((smax > q95).mean()),
                                frac_ge_0995_copy_threshold=float((smax >= 0.995).mean())))
            for i in range(len(S)):
                rows.append(dict(cls=c, set=tag, synthetic=Sn[i], nearest_real=Rnames[nn[i]],
                                 cosine=float(smax[i]), synthetic_path=Sp[i], real_path=Rpaths[nn[i]]))
        ax.axvline(q95, ls="--", c="k", lw=1)
        ax.set_title(f"{c}: nearest-neighbour similarity (DINOv2)"); ax.set_xlabel("cosine similarity")
        ax.legend(fontsize=7)
    _savefig(fig, out / "fig_nn_similarity_hist")
    nn_df = pd.DataFrame(rows); nn_df.to_csv(out / "nearest_neighbours.csv", index=False)
    pd.DataFrame(summary).to_csv(out / "memorisation_summary.csv", index=False)
    print(pd.DataFrame(summary).round(4).to_string(index=False))
    # ---- gallery: selected synthetic next to its nearest real cell (with zoom)
    gal = out / "gallery_selected_vs_nearest_real"
    for c in CLASSES:
        d = nn_df[(nn_df.cls == c) & (nn_df.set == "sel")].sort_values("cosine").reset_index(drop=True)
        if d.empty:
            continue
        pick = np.unique(np.linspace(0, len(d) - 1, min(N_GALLERY, len(d))).round().astype(int))
        items = []
        for k, i in enumerate(pick, 1):
            r = d.iloc[i]
            real_im = Image.open(r.real_path).convert("RGB"); syn_im = Image.open(r.synthetic_path).convert("RGB")
            _save_img(real_im, gal / c / f"{k:02d}_real_{r.nearest_real}")
            _save_img(syn_im, gal / c / f"{k:02d}_synthetic_{r.synthetic}")
            _save_img(_zoom(real_im), gal / c / f"{k:02d}_real_zoom.png")
            _save_img(_zoom(syn_im), gal / c / f"{k:02d}_synthetic_zoom.png")
            items.append((real_im, syn_im, _zoom(real_im), _zoom(syn_im), f"#{k} cos={r.cosine:.3f}"))
        _pair_sheet(items, ["Real (nearest)", "Synthetic (QC-selected)", "Real zoom", "Synthetic zoom"],
                    gal / f"sheet_{c}")
    _mark("memorisation")


def _zoom(im, box=72, out=256):
    a = np.asarray(im.convert("L"), dtype=np.float32)
    from scipy.ndimage import uniform_filter
    s = uniform_filter(a, size=box // 2)
    h, w = a.shape
    y, x = np.unravel_index(np.argmax(s), s.shape)
    y0 = int(np.clip(y - box // 2, 0, max(0, h - box))); x0 = int(np.clip(x - box // 2, 0, max(0, w - box)))
    return im.crop((x0, y0, x0 + box, y0 + box)).resize((out, out), Image.BICUBIC)


def _pair_sheet(items, col_titles, path_noext):
    n = len(items)
    if n == 0:
        return
    ncol = len(items[0]) - 1
    fig, axes = plt.subplots(n, ncol, figsize=(2.6 * ncol, 2.6 * n))
    axes = np.atleast_2d(axes)
    for r, it in enumerate(items):
        for cidx in range(ncol):
            ax = axes[r, cidx]; ax.imshow(it[cidx]); ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(col_titles[cidx], fontsize=9)
        axes[r, 0].set_ylabel(it[-1], fontsize=8)
    _savefig(fig, path_noext, dpi=SHEET_DPI)


def _grid_sheet(paths_by_col, col_titles, path_noext, n=N_GALLERY):
    cols = [p[:n] for p in paths_by_col]
    rows = max(len(c) for c in cols) if cols else 0
    if rows == 0:
        return
    fig, axes = plt.subplots(rows, len(cols), figsize=(2.4 * len(cols), 2.4 * rows))
    axes = np.atleast_2d(axes)
    if axes.shape != (rows, len(cols)):
        axes = axes.reshape(rows, len(cols))
    for j, col in enumerate(cols):
        for i in range(rows):
            ax = axes[i, j]; ax.set_xticks([]); ax.set_yticks([])
            if i < len(col):
                ax.imshow(Image.open(col[i]).convert("RGB"))
                ax.set_xlabel(f"#{i + 1:02d} {os.path.basename(col[i])[:28]}", fontsize=5)
            else:
                ax.axis("off")
            if i == 0:
                ax.set_title(col_titles[j], fontsize=9)
    _savefig(fig, path_noext, dpi=SHEET_DPI)


# =============================================================================
# 4. Texture fidelity: radial power spectrum + uniform LBP (E8, attribution)
# =============================================================================
def _gray(p):
    return np.asarray(Image.open(p).convert("RGB"), dtype=np.float64) @ np.array([0.299, 0.587, 0.114]) / 255.0


def _radial_psd(g, nbins=64):
    h, w = g.shape
    win = np.outer(np.hanning(h), np.hanning(w))
    F = np.fft.fftshift(np.abs(np.fft.fft2((g - g.mean()) * win)) ** 2)
    yy, xx = np.indices((h, w)); r = np.hypot(yy - h / 2, xx - w / 2) / max(h, w)   # cycles/px
    edges = np.linspace(1.0 / max(h, w), 0.5, nbins + 1)
    idx = np.digitize(r.ravel(), edges) - 1
    ok = (idx >= 0) & (idx < nbins)
    prof = np.bincount(idx[ok], weights=F.ravel()[ok], minlength=nbins) / np.maximum(np.bincount(idx[ok], minlength=nbins), 1)
    prof = prof / max(prof.sum(), 1e-12)
    return 0.5 * (edges[:-1] + edges[1:]), prof


def _lbp_hist(g):
    from skimage.feature import local_binary_pattern
    lbp = local_binary_pattern((g * 255).astype(np.uint8), P=8, R=1, method="uniform")
    h = np.bincount(lbp.astype(int).ravel(), minlength=10)[:10].astype(np.float64)
    return h / h.sum()


def texture_fidelity(paths, force=False):
    out = A.RESULTS / "texture"
    if _done("texture") and not force:
        print("Texture analysis already done."); return
    out.mkdir(parents=True, exist_ok=True)
    groups = {"Real": {c: _list_images(os.path.join(paths["train"], c)) for c in CLASSES},
              "NTP selected": class_sets(paths, "synthetic_selected"),
              "NTP all": class_sets(paths, "synthetic_all_ntp"),
              "Baseline all": class_sets(paths, "synthetic_all_baseline"),
              "Baseline selected": class_sets(paths, "baseline_selected_qc")}
    groups = {k: v for k, v in groups.items() if all(len(v[c]) > 0 for c in CLASSES)}
    rows = []
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ci, c in enumerate(CLASSES):
        stats = {}
        for name, s in groups.items():
            ps = s[c][:1000]
            P = []; L = []
            for p in ps:
                g = _gray(p); f, pr = _radial_psd(g); P.append(pr); L.append(_lbp_hist(g))
            stats[name] = (f, np.array(P), np.array(L), len(ps))
        f, Pr, Lr, nr = stats["Real"]
        for name, (f2, P, L, n) in stats.items():
            m, sd = np.log10(P + 1e-12).mean(0), np.log10(P + 1e-12).std(0)
            axes[0, ci].plot(f2, m, label=f"{name} (n={n})", lw=1.6 if name == "Real" else 1.2,
                             ls="-" if name == "Real" else "--")
            if name in ("Real", "NTP selected"):
                axes[0, ci].fill_between(f2, m - sd, m + sd, alpha=0.15)
            hm = L.mean(0); hr = Lr.mean(0)
            if name != "Real":
                rows.append(dict(cls=c, set=name, n=n,
                                 spectral_MAE_log10=float(np.abs(np.log10(P.mean(0) + 1e-12) - np.log10(Pr.mean(0) + 1e-12)).mean()),
                                 LBP_correlation=float(np.corrcoef(hm, hr)[0, 1]),
                                 LBP_intersection=float(np.minimum(hm, hr).sum())))
        axes[0, ci].set_xscale("log"); axes[0, ci].set_title(f"{c}: radially averaged power spectrum")
        axes[0, ci].set_xlabel("spatial frequency (cycles px$^{-1}$)"); axes[0, ci].set_ylabel("log$_{10}$ normalised power")
        axes[0, ci].legend(fontsize=7)
        names = list(stats); wbar = 0.8 / len(names)
        for k, name in enumerate(names):
            L = stats[name][2]
            axes[1, ci].bar(np.arange(10) + k * wbar, L.mean(0), wbar, yerr=L.std(0), capsize=1.5,
                            label=name, error_kw=dict(lw=0.6))
        axes[1, ci].set_xticks(np.arange(10) + 0.4 - wbar / 2); axes[1, ci].set_xticklabels(range(10))
        axes[1, ci].set_title(f"{c}: uniform LBP (P=8, R=1)"); axes[1, ci].set_xlabel("LBP code")
        axes[1, ci].legend(fontsize=7)
    _savefig(fig, out / "fig_texture_fidelity_all_sets")
    df = pd.DataFrame(rows); df.to_csv(out / "texture_fidelity_metrics.csv", index=False)
    _df_to_latex(df.round(4), out / "table_texture_fidelity.tex",
                 "Texture fidelity of each synthetic set relative to the real training images "
                 "(lower spectral MAE and higher LBP correlation/intersection are better).", "tab:texture_sets")
    print(df.round(4).to_string(index=False))
    _mark("texture")


# =============================================================================
# 5. Image galleries (>= 10 candidates per comparison)
# =============================================================================
def galleries(paths, force=False):
    out = A.RESULTS / "galleries"
    if _done("galleries") and not force:
        print("Galleries already done."); return
    C, K = A.import_pipeline(paths)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(2026)
    # (a) preprocessing: original | Otsu foreground crop | network input (224 px)
    crop = C.ForegroundCrop()
    for c in CLASSES:
        ps = _list_images(os.path.join(paths["train"], c))
        pick = [ps[i] for i in sorted(rng.choice(len(ps), min(N_GALLERY, len(ps)), replace=False))]
        items = []
        for k, p in enumerate(pick, 1):
            im = Image.open(p).convert("RGB"); cr = crop(im); net = cr.resize((224, 224), Image.BICUBIC)
            _save_img(im, out / "preprocessing" / c / f"{k:02d}_original_{os.path.basename(p)}")
            _save_img(net, out / "preprocessing" / c / f"{k:02d}_network_input.png")
            items.append((im, cr, net, f"#{k}"))
        _pair_sheet(items, ["Original crop (256 px)", "Otsu foreground crop", "Network input (224 px)"],
                    out / "preprocessing" / f"sheet_{c}")
    # (b) real vs QC-selected vs rejected vs baseline, per class
    sel = class_sets(paths, "synthetic_selected")
    allntp = class_sets(paths, "synthetic_all_ntp")
    base = class_sets(paths, "synthetic_all_baseline")
    basesel = class_sets(paths, "baseline_selected_qc")
    for c in CLASSES:
        real = _list_images(os.path.join(paths["train"], c))
        cols, titles = [], []
        def _pick(lst):
            return [lst[i] for i in sorted(rng.choice(len(lst), min(N_GALLERY, len(lst)), replace=False))] if lst else []
        cols.append(_pick(real)); titles.append("Real")
        cols.append(_pick(sel[c])); titles.append("NTP, QC-selected")
        if allntp[c]:
            selected_names = {os.path.basename(p) for p in sel[c]}
            rejected = [p for p in allntp[c] if os.path.basename(p) not in selected_names]
            cols.append(_pick(rejected)); titles.append("NTP, rejected by QC")
        if base[c]:
            cols.append(_pick(base[c])); titles.append("Baseline (no NTP)")
        if basesel[c]:
            cols.append(_pick(basesel[c])); titles.append("Baseline, QC-selected")
        for col, t in zip(cols, titles):
            for k, p in enumerate(col, 1):
                tdir = out / "synthetic_comparison" / c / t.replace(" ", "_").replace(",", "")
                _save_img(Image.open(p).convert("RGB"), tdir / f"{k:02d}_{os.path.basename(p)}")
        _grid_sheet(cols, titles, out / "synthetic_comparison" / f"sheet_{c}")
    _mark("galleries")


# =============================================================================
# 6. Optional: baseline (no-NTP) generator -> generation + QC (NO training)
# =============================================================================
def prepare_baseline(paths):
    """Generate 1000/class from an uploaded baseline checkpoint and run the
    official QC script on them. Skipped when the folders already exist."""
    qc = A.RESULTS / "baseline_qc"
    if "synthetic_selected_baseline" in paths or (qc / "selected" / "normal").exists():
        print("Baseline-selected set available."); return True
    gen_root = paths.get("synthetic_all_baseline")
    if not gen_root and "baseline_ckpt" in paths:
        out = qc / "generated"
        if not (out / "final_generated" / "normal").exists():
            code = ("import sys,torch; torch.manual_seed(2026); sys.path.insert(0,%r); "
                    "sys.argv=['gan_training.py','--mode','generate','--weights',%r,'--per-class','1000',"
                    "'--output-path',%r]; import gan_training as G; G.main()"
                    % (str(A.REPO_DIR), paths["baseline_ckpt"], str(out)))
            r = subprocess.run([sys.executable, "-c", code], cwd=str(A.WORK))
            if r.returncode != 0:
                print("Baseline generation failed (checkpoint/architecture mismatch?) -> baseline arms skipped.")
                return False
        gen_root = str(out / "final_generated")
    if not gen_root:
        print("No baseline images or checkpoint uploaded -> baseline arms skipped.")
        return False
    code = ("import sys; sys.path.insert(0,%r); import gan_quality_selection as Q; "
            "Q.REAL_FOLDER=%r; Q.GAN_FOLDER=%r; Q.OUTPUT_FOLDER=%r; Q.DINO_AUTO_SELECT=False; "
            "Q._validate_pipeline(); Q.main()" % (str(A.REPO_DIR), paths["train"], gen_root, str(qc / "selected")))
    r = subprocess.run([sys.executable, "-c", code], cwd=str(A.WORK))
    ok = r.returncode == 0 and (qc / "selected" / "normal").exists()
    print("Baseline QC selection", "finished." if ok else "FAILED -> baseline-selected arm skipped.")
    return ok


# =============================================================================
# 7. Grad-CAM++ quantification + galleries (E7)
# =============================================================================
def load_ckpt(C, path, device):
    import torch
    ck = torch.load(path, map_location=device, weights_only=False)
    arch = ck["architecture"]
    m = C.build_model(arch["backbone"], len(ck["class_names"]), device, allow_random_init=True,
                      pretrained=False, embedding_dim=arch.get("feature_dim", 256),
                      embedding_dropout=arch.get("embedding_dropout", 0.10))
    m.load_state_dict(ck["model_state_dict"], strict=True)
    m.eval()
    return m, ck


def _seed_ckpts(exp):
    d = A.RUNS / "E1_kfold" / "AFiSNet" / exp / f"seed_{A.SEEDS[0]}"
    return {f: d / f"fold_{f}" / "best_model.pt" for f in range(A.N_FOLDS)
            if (d / f"fold_{f}" / "best_model.pt").exists()}


def _otsu_mask(base01):
    sig = (base01.max(2) * 255).astype(np.uint8)
    hist = np.bincount(sig.ravel(), minlength=256).astype(np.float64)
    w = np.cumsum(hist); mu = np.cumsum(hist * np.arange(256)); tot = w[-1]
    between = np.zeros(256); v = (w > 0) & (w < tot)
    between[v] = (mu[v] * tot - mu[-1] * w[v]) ** 2 / (w[v] * (tot - w[v]))
    return sig > max(6, int(np.argmax(between)))


def _cam_one(C, model, cam_layer, path, eval_tf, crop, device, T, thr, class_idx=None):
    import torch
    pil = Image.open(path).convert("RGB")
    x = eval_tf(pil).unsqueeze(0).to(device)
    base = np.asarray(crop(pil).resize((224, 224), Image.BICUBIC), dtype=np.float32) / 255.0
    with torch.no_grad():
        p = torch.softmax(model(x).float() / max(T, 1e-6), 1)[0].cpu().numpy()
    pred = int(p[1] >= thr)
    cls = pred if class_idx is None else class_idx
    cammer = C.GradCAMpp(model, cam_layer)
    try:
        with torch.enable_grad():
            cam, _ = cammer(x, class_idx=cls)
    finally:
        cammer.remove()
    up = C._upsample_cam(cam, 224, 224)
    return x, base, up, p, pred, cls


def _trapz(y, x):
    y = np.asarray(y, float); x = np.asarray(x, float)
    return float(np.sum((y[1:] + y[:-1]) * np.diff(x) / 2.0))


def _del_ins(model, x, up, cls, T, rng, steps=10):
    import torch
    H, W = up.shape
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
    black = ((torch.zeros_like(x) - mean) / std)
    fr = np.linspace(0, 1, steps + 1)
    res = {}
    for name, order in (("cam", np.argsort(-up.ravel())), ("random", rng.permutation(H * W))):
        masks = []
        for f in fr:
            m = np.zeros(H * W, np.float32); m[order[:int(round(f * H * W))]] = 1
            masks.append(m.reshape(1, H, W))
        M = torch.from_numpy(np.stack(masks)).to(x.device)            # (S,1,H,W)
        dele = x * (1 - M) + black * M
        ins = black * (1 - M) + x * M
        with torch.no_grad():
            pd_ = torch.softmax(model(dele).float() / T, 1)[:, cls].cpu().numpy()
            pi_ = torch.softmax(model(ins).float() / T, 1)[:, cls].cpu().numpy()
        res[f"deletion_auc_{name}"] = _trapz(pd_, fr)
        res[f"insertion_auc_{name}"] = _trapz(pi_, fr)
    return res


def _overlay(base, up):
    heat = matplotlib.colormaps["viridis"](np.clip(up, 0, 1))[..., :3]
    return np.clip(0.55 * base + 0.45 * heat, 0, 1)


def _cellfocused(base, up, mask):
    g = base @ np.array([0.299, 0.587, 0.114]); gray = np.repeat(g[..., None], 3, 2)
    heat = matplotlib.colormaps["viridis"](np.clip(up, 0, 1))[..., :3]
    out = gray.copy(); out[mask] = np.clip(0.45 * gray[mask] + 0.55 * heat[mask], 0, 1)
    return out


def _wil(a, b, alternative="two-sided"):
    from scipy.stats import wilcoxon
    a = np.asarray(a, float); b = np.asarray(b, float)
    if len(a) < 2 or np.allclose(a, b):
        return float("nan")
    try:
        return float(wilcoxon(a, b, alternative=alternative).pvalue)
    except ValueError:
        return float("nan")


def gradcam_quant(paths, force=False):
    import torch
    out = A.RESULTS / "gradcam"
    if _done("gradcam") and not force:
        print("Grad-CAM quantification already done."); return
    C, K = A.import_pipeline(paths)
    arms = {"Raw": _seed_ckpts("Raw"), "Raw+Synthetic": _seed_ckpts(A.SYN_EXP)}
    if len(arms["Raw+Synthetic"]) < A.N_FOLDS:
        print("AFiS-Net seed-42 checkpoints not complete yet -> Grad-CAM runs in a later session."); return
    if len(arms["Raw"]) < A.N_FOLDS:
        print("Raw-arm checkpoints incomplete -> Raw-vs-Synthetic CAM comparison skipped.")
        arms.pop("Raw")
    out.mkdir(parents=True, exist_ok=True)
    dev = _dev(); pool = real_pool(paths)
    cfg = C.Config(); _, eval_tf = C.build_transforms(224, cfg); crop = C.ForegroundCrop()
    rng = np.random.default_rng(0)
    rows = []; cams = {}
    for arm, ck in arms.items():
        for f, path in sorted(ck.items()):
            model, ckd = load_ckpt(C, path, dev)
            T = float(ckd["decision"]["temperature"]); thr = float(ckd["decision"]["cancer_threshold"])
            layer = C.cam_target_layer(model, model._cam)
            man = json.loads((path.parent / "split_manifest.json").read_text())
            for b in man["test"]:
                p, c, _ = pool[b]
                x, base, up, prob, pred, cls = _cam_one(C, model, layer, p, eval_tf, crop, dev, T, thr)
                mask_repo = C.tissue_mask(base, cfg.cell_thresh)          # repo's cell mask
                mask_otsu = _otsu_mask(base)
                ok = bool(np.isfinite(up).all() and up.max() > 1e-6)     # all-zero CAM = undefined, not "0"
                nan = float("nan")
                r = dict(arm=arm, fold=f, image=b, true=c, pred=CLASSES[pred], p_cancer=float(prob[1]),
                         correct=CLASSES[pred] == c, cam_class=CLASSES[cls], cam_valid=ok,
                         cell_attention_ratio=float((up * mask_repo).sum() / up.sum()) if ok else nan,
                         cell_area_fraction=float(mask_repo.mean()),
                         otsu_attention_ratio=float((up * mask_otsu).sum() / up.sum()) if ok else nan,
                         otsu_area_fraction=float(mask_otsu.mean()),
                         pointing_game_hit=float(mask_repo.reshape(-1)[int(np.argmax(up))]) if ok else nan)
                if ok:
                    r.update(_del_ins(model, x, up, cls, T, rng))
                else:
                    r.update({f"{a}_auc_{o}": nan for a in ("deletion", "insertion") for o in ("cam", "random")})
                rows.append(r)
                cams[(arm, b)] = (up.astype(np.float16), base.astype(np.float16), mask_repo, float(prob[1]), CLASSES[pred])
            del model
            if dev.type == "cuda":
                torch.cuda.empty_cache()
    df = pd.DataFrame(rows); df.to_csv(out / "gradcam_per_image.csv", index=False)
    # ---- summary + paired tests (CAM vs random order)
    from scipy.stats import wilcoxon
    summ = []
    for (arm, cl), d in df.groupby(["arm", "true"]):
        s = dict(arm=arm, cls=cl, n=len(d), n_degenerate_cam=int((~d.cam_valid).sum()))
        d = d[d.cam_valid]
        for k in ("cell_attention_ratio", "cell_area_fraction", "otsu_attention_ratio", "otsu_area_fraction",
                  "deletion_auc_cam", "deletion_auc_random", "insertion_auc_cam", "insertion_auc_random"):
            s[k] = f"{d[k].mean():.3f} ± {d[k].std():.3f}"
        s["pointing_game_hit_rate"] = round(float(d.pointing_game_hit.mean()), 3) if len(d) else float("nan")
        s["p_deletion_cam<random"] = _wil(d.deletion_auc_cam, d.deletion_auc_random, "less")
        s["p_insertion_cam>random"] = _wil(d.insertion_auc_cam, d.insertion_auc_random, "greater")
        s["p_attention>area"] = _wil(d.cell_attention_ratio, d.cell_area_fraction, "greater")
        summ.append(s)
    sd = pd.DataFrame(summ); sd.to_csv(out / "gradcam_summary.csv", index=False)
    _df_to_latex(sd[["arm", "cls", "n", "n_degenerate_cam", "cell_attention_ratio", "cell_area_fraction", "pointing_game_hit_rate",
                     "deletion_auc_cam", "deletion_auc_random", "insertion_auc_cam", "insertion_auc_random"]],
                 out / "table_gradcam.tex",
                 "Quantitative Grad-CAM++ evaluation of AFiS-Net over all test images of the region-grouped "
                 "folds (seed 42). Attention ratio = share of CAM energy inside the cell mask; deletion/insertion "
                 "AUC compared with a random pixel order.", "tab:gradcam")
    print(sd.to_string(index=False))
    # ---- galleries
    gdir = out / "gallery"
    syn = df[(df.arm == "Raw+Synthetic") & df.cam_valid]
    def _export(sub, name, n=N_GALLERY):
        items = []
        for k, (_, r) in enumerate(sub.head(n).iterrows(), 1):
            up, base, mask, pc, pr = cams[("Raw+Synthetic", r.image)]
            up = up.astype(np.float32); base = base.astype(np.float32)
            ov = _overlay(base, up); cf = _cellfocused(base, up, mask)
            stem = gdir / name / f"{k:02d}_{os.path.splitext(r.image)[0]}"
            _save_img(base, f"{stem}_input.png"); _save_img(ov, f"{stem}_gradcampp.png"); _save_img(cf, f"{stem}_cellfocused.png")
            items.append((base, ov, cf, f"#{k} GT {r.true} / pred {r.pred} (P={r.p_cancer:.2f})"))
        _pair_sheet(items, ["Input (network view)", "Grad-CAM++", "Cell-focused"], gdir / f"sheet_{name}")
    _export(syn[(syn.true == "cancer") & syn.correct].sort_values("p_cancer", ascending=False), "correct_cancer")
    _export(syn[(syn.true == "normal") & syn.correct].sort_values("p_cancer"), "correct_normal")
    _export(syn[(syn.true == "cancer") & ~syn.correct].sort_values("p_cancer"), "misclassified_cancer")
    _export(syn[(syn.true == "normal") & ~syn.correct].sort_values("p_cancer", ascending=False), "misclassified_normal")
    # Raw vs Raw+Synthetic model on the same cells (both correct)
    if "Raw" not in arms:
        _mark("gradcam"); return
    dv = df[df.cam_valid]
    piv = dv.pivot_table(index="image", columns="arm", values="correct", aggfunc="first").dropna()
    both = piv[(piv["Raw"].astype(bool)) & (piv["Raw+Synthetic"].astype(bool))].index
    for c in CLASSES:
        sub = syn[syn.image.isin(both) & (syn.true == c)].sort_values("p_cancer", ascending=(c == "normal")).head(N_GALLERY)
        items = []
        for k, (_, r) in enumerate(sub.iterrows(), 1):
            u1, base, _, p1, _ = cams[("Raw", r.image)]; u2, _, _, p2, _ = cams[("Raw+Synthetic", r.image)]
            base = base.astype(np.float32)
            o1, o2 = _overlay(base, u1.astype(np.float32)), _overlay(base, u2.astype(np.float32))
            stem = gdir / f"raw_vs_synthetic_{c}" / f"{k:02d}_{os.path.splitext(r.image)[0]}"
            _save_img(base, f"{stem}_input.png"); _save_img(o1, f"{stem}_raw_model.png"); _save_img(o2, f"{stem}_synthetic_model.png")
            items.append((base, o1, o2, f"#{k} P(cancer) raw {p1:.2f} / +syn {p2:.2f}"))
        _pair_sheet(items, ["Input", "AFiS-Net (Raw)", "AFiS-Net (Raw+Synthetic)"], gdir / f"sheet_raw_vs_synthetic_{c}")
    _mark("gradcam")


# =============================================================================
# 8. Screening re-analysis with the Kaggle-trained ensemble
# =============================================================================
def _jt_test(groups_vals, n_perm=20000, seed=0):
    """Jonckheere-Terpstra trend test (increasing order), permutation p-value."""
    def stat(vals):
        t = 0.0
        for i in range(len(vals)):
            for j in range(i + 1, len(vals)):
                a = np.asarray(vals[i])[:, None]; b = np.asarray(vals[j])[None, :]
                t += float((a < b).sum() + 0.5 * (a == b).sum())
        return t
    obs = stat(groups_vals)
    allv = np.concatenate(groups_vals); sizes = np.cumsum([len(v) for v in groups_vals])[:-1]
    rng = np.random.default_rng(seed); ge = 0
    for _ in range(n_perm):
        ge += stat(np.split(rng.permutation(allv), sizes)) >= obs
    return obs, (ge + 1) / (n_perm + 1)


def _screening_stats(df, order, outdir):
    from scipy.stats import kruskal, mannwhitneyu
    res = {}
    for level, d in (("image", df), ("region", df.groupby(["group", "region"], as_index=False).p_cancer.mean())):
        vals = [d[d.group == g].p_cancer.values for g in order]
        ok = [len(v) > 0 for v in vals]
        r = {"n": {g: int(len(v)) for g, v in zip(order, vals)}}
        if sum(ok) >= 2 and all(len(v) > 1 for v in vals if len(v)):
            r["kruskal_p"] = float(kruskal(*[v for v in vals if len(v)]).pvalue)
        main = [d[d.group == g].p_cancer.values for g in ("normal", "smoke", "cancer") if g in order]
        if len(main) == 3 and all(len(v) for v in main):
            jt, pj = _jt_test(main)
            r["jonckheere_normal<smoke<cancer"] = {"J": jt, "p_one_sided": float(pj)}
        pw = []
        for i in range(len(order)):
            for j in range(i + 1, len(order)):
                a, b = vals[i], vals[j]
                if len(a) and len(b):
                    pw.append((f"{order[i]} vs {order[j]}", float(mannwhitneyu(a, b).pvalue)))
        if pw:
            holm = _holm([p for _, p in pw])
            r["pairwise_mannwhitney"] = {k: {"p": p, "p_holm": float(h)} for (k, p), h in zip(pw, holm)}
        res[level] = r
    summ = df.groupby("group").agg(n=("image", "size"), regions=("region", "nunique"),
                                   p_cancer_mean=("p_cancer", "mean"), p_cancer_median=("p_cancer", "median"),
                                   p_cancer_sd=("p_cancer", "std"), positive_rate=("positive", "mean"),
                                   positive_rate_0_5=("positive_at_0_5", "mean"),
                                   n_scored_out_of_fold=("in_cv_pool", "sum")).reindex(order)
    summ.to_csv(outdir / "group_summary_leakfree.csv")
    (outdir / "stats_leakfree.json").write_text(json.dumps(res, indent=1))
    _df_to_latex(summ.reset_index().round(3), outdir / "table_screening_leakfree.tex",
                 "Screening cohort scored without leakage: images that are also part of the cross-validation pool "
                 "are scored only by the fold model that held them out; the others by the 5-fold ensemble "
                 "(AFiS-Net, Raw+Synthetic, seed 42).", "tab:screening")
    fig, ax = plt.subplots(figsize=(5.5, 4))
    rng = np.random.default_rng(0)
    for i, g in enumerate(order):
        v = df[df.group == g].p_cancer.values
        if len(v):
            ax.boxplot([v], positions=[i], widths=0.5, showfliers=False)
            ax.scatter(i + rng.uniform(-0.15, 0.15, len(v)), v, s=14, alpha=0.7)
    ax.axhline(float(df.threshold.median()), ls="--", c="grey", lw=0.8, label="median validation threshold")
    ax.set_xticks(range(len(order))); ax.set_xticklabels(order); ax.set_ylabel("P(cancer)")
    jt = res["image"].get("jonckheere_normal<smoke<cancer")
    ax.set_title("Screening cohort (leak-free)" + (f", JT trend p = {jt['p_one_sided']:.3g}" if jt else ""))
    ax.legend(fontsize=8)
    _savefig(fig, outdir / "fig_screening_leakfree")
    print(summ.round(3).to_string()); print(json.dumps(res, indent=1)[:1500])


def screening(paths, force=False):
    import torch
    out = A.RESULTS / "screening"
    if _done("screening") and not force:
        print("Screening already done."); return
    if "screening" not in paths:
        print("No screening folder uploaded -> skipped."); return
    ck = _seed_ckpts(A.SYN_EXP)
    if len(ck) < A.N_FOLDS:
        print("AFiS-Net Raw+Synthetic seed-42 checkpoints incomplete -> screening runs later."); return
    C, K = A.import_pipeline(paths)
    out.mkdir(parents=True, exist_ok=True)
    ckdir = A.RUNS / "E1_kfold" / "AFiSNet" / A.SYN_EXP / f"seed_{A.SEEDS[0]}"
    dev = _dev()
    thrs = [float(torch.load(p, map_location="cpu", weights_only=False)["decision"]["cancer_threshold"])
            for p in ck.values()]
    for tag, thr in (("threshold_0.5", 0.5), ("threshold_validation_youden", float(np.median(thrs)))):
        args = [sys.executable, str(A.REPO_DIR / "screening_analysis.py"), "--checkpoint", str(ckdir),
                "--data-root", paths["screening"], "--output", str(out / tag), "--threshold", f"{thr:.6f}"]
        if dev.type != "cuda":
            args.append("--cpu")
        r = subprocess.run(args, cwd=str(A.REPO_DIR))
        print(f"screening_analysis.py ({tag}, thr={thr:.4f}) exit code {r.returncode}")
    (out / "thresholds_used.json").write_text(json.dumps({"fold_youden_thresholds": thrs,
                                                         "median": float(np.median(thrs))}, indent=1))
    # ---- leak-free screening (out-of-fold for any image that is also in the CV pool) ----------------
    # The normal/cancer screening folders can contain images of the real dataset (e.g. copies of real/val).
    # Each such image was a TRAINING image for 3 of the 5 fold models, so it is scored only by the one
    # fold model that held it out as a test image. Images outside the pool (e.g. smokers) use all 5.
    pool = real_pool(paths)
    test_fold = {}
    for f, pth in ck.items():
        man = json.loads((pth.parent / "split_manifest.json").read_text())
        for bname in man["test"]:
            test_fold[bname] = f
    cfg = C.Config(); _, eval_tf = C.build_transforms(224, cfg); crop = C.ForegroundCrop()
    folds = sorted(ck)
    members = {f: load_ckpt(C, ck[f], dev) for f in folds}
    groups = {}
    for g in sorted(os.listdir(paths["screening"])):
        d = os.path.join(paths["screening"], g)
        if os.path.isdir(d):
            groups[{"smoker": "smoke"}.get(g.lower(), g.lower())] = _list_images(d)
    order = [g for g in ("normal", "smoke", "cancer") if g in groups] + \
            [g for g in groups if g not in ("normal", "smoke", "cancer")]
    rows = []
    lf = out / "leakfree"
    for g in order:
        items = []
        for k, p in enumerate(groups[g], 1):
            b = os.path.basename(p)
            in_pool = b in pool
            use = [test_fold[b]] if b in test_fold else folds
            ups, pcs, thr_used = [], [], []
            for f in use:
                model, ckd = members[f]
                T = float(ckd["decision"]["temperature"]); thr = float(ckd["decision"]["cancer_threshold"])
                layer = C.cam_target_layer(model, model._cam)
                x, base, up, prob, pred, cls = _cam_one(C, model, layer, p, eval_tf, crop, dev, T, thr, class_idx=1)
                ups.append(up); pcs.append(float(prob[1])); thr_used.append(thr)
            up = np.mean(ups, 0); up = (up - up.min()) / (up.max() - up.min() + 1e-8)
            pc = float(np.mean(pcs)); thr = float(np.median(thr_used))
            rows.append(dict(group=g, image=b, region=K.region_group(b), in_cv_pool=in_pool,
                             pool_split=pool[b][2] if in_pool else "", pool_class=pool[b][1] if in_pool else "",
                             scored_by=("fold %d (held-out)" % use[0]) if len(use) == 1 else "5-fold ensemble",
                             p_cancer=pc, p_cancer_sd=float(np.std(pcs)), threshold=thr,
                             positive=bool(pc >= thr), positive_at_0_5=bool(pc >= 0.5)))
            ov = _overlay(base, up)
            stem = lf / "gallery" / g / f"{k:02d}_{os.path.splitext(b)[0]}"
            _save_img(base, f"{stem}_input.png"); _save_img(ov, f"{stem}_cancer_gradcampp.png")
            items.append((base, ov, f"#{k} P(cancer)={pc:.2f}" + (" (OOF)" if len(use) == 1 else "")))
        _pair_sheet(items, ["Input", "Cancer-class Grad-CAM++ (leak-free)"], lf / "gallery" / f"sheet_{g}")
    del members
    df = pd.DataFrame(rows); df.to_csv(lf / "per_image_leakfree.csv", index=False)
    _screening_stats(df, order, lf)
    n_ov = int(df.in_cv_pool.sum())
    note = (f"{n_ov} of {len(df)} screening images are also in the real CV pool "
            f"({df[df.in_cv_pool].groupby('group').size().to_dict()}). The official script outputs "
            "(threshold_*/) score them with the 5-fold ensemble, i.e. partly with models trained on them; "
            "use leakfree/ for the paper.")
    (out / "README_overlap.txt").write_text(note + "\n"); print(note)
    _mark("screening")


# =============================================================================
# 9. Statistics, tables and figures for every experiment family
# =============================================================================
TABLE_METRICS = ["f1_macro", "roc_auc_macro", "mcc", "Cancer_recall_sensitivity", "specificity_macro",
                 "balanced_accuracy", "Cancer_precision", "ece", "brier_score"]
SHORT = {"f1_macro": "F1$_\\text{macro}$", "roc_auc_macro": "AUC", "mcc": "MCC",
         "Cancer_recall_sensitivity": "Sen$_\\text{c}$", "specificity_macro": "Spec$_\\text{macro}$",
         "balanced_accuracy": "Bal. acc.", "Cancer_precision": "Prec$_\\text{c}$", "ece": "ECE",
         "brier_score": "Brier"}


def load_records():
    rows = []
    for f in A.RUNS.rglob("record.json") if A.RUNS.exists() else []:
        r = json.loads(f.read_text()); r["_dir"] = str(f.parent); rows.append(r)
    return pd.DataFrame(rows)


def load_preds(rec):
    p = pd.read_csv(Path(rec["_dir"]) / "predictions.csv")
    p["y"] = (p.GroundTruth == "cancer").astype(int); p["yhat"] = (p.Prediction == "cancer").astype(int)
    return p


def _delong(y, s1, s2):
    """Paired DeLong test for two correlated ROC AUCs (fast version, Sun & Xu 2014)."""
    from scipy.stats import norm
    y = np.asarray(y); pos, neg = y == 1, y == 0
    m, n = pos.sum(), neg.sum()
    def _mid(x):
        J = np.argsort(x); Z = x[J]; N = len(x); T = np.zeros(N); i = 0
        while i < N:
            j = i
            while j < N and Z[j] == Z[i]:
                j += 1
            T[i:j] = 0.5 * (i + j - 1) + 1; i = j
        T2 = np.empty(N); T2[J] = T; return T2
    aucs, V10, V01 = [], [], []
    for s in (s1, s2):
        s = np.asarray(s, float); X, Yv = s[pos], s[neg]
        tx, ty, tz = _mid(X), _mid(Yv), _mid(np.concatenate([X, Yv]))
        auc = (tz[:m].sum() / m - (m + 1) / 2) / n
        aucs.append(auc); V10.append((tz[:m] - tx) / n); V01.append(1 - (tz[m:] - ty) / m)
    V10, V01 = np.array(V10), np.array(V01)
    S = np.cov(V10) / m + np.cov(V01) / n
    var = S[0, 0] + S[1, 1] - 2 * S[0, 1]
    z = (aucs[0] - aucs[1]) / np.sqrt(var) if var > 0 else 0.0
    return float(aucs[0]), float(aucs[1]), float(2 * norm.sf(abs(z)))


def _mcnemar(y, a, b):
    from scipy.stats import binomtest
    ca, cb = a == y, b == y
    n01, n10 = int((ca & ~cb).sum()), int((~ca & cb).sum())
    return float(binomtest(min(n01, n10), n01 + n10, 0.5).pvalue) if n01 + n10 > 0 else 1.0


def _holm(ps):
    ps = np.asarray(ps, float); order = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * ps[i]); adj[i] = min(1.0, run)
    return adj


def _pooled(df_runs):
    """Pooled out-of-fold predictions per seed for one (family, model, exp)."""
    out = {}
    for s, d in df_runs.groupby("seed"):
        if d.fold.nunique() < A.N_FOLDS:
            continue
        out[int(s)] = pd.concat([load_preds(r) for _, r in d.iterrows()], ignore_index=True)
    return out


def _pooled_metrics(P):
    from sklearn.metrics import roc_auc_score, f1_score, matthews_corrcoef
    return dict(pooled_auc=roc_auc_score(P.y, P.Probability_Cancer),
                pooled_f1=f1_score(P.y, P.yhat, average="macro"), pooled_mcc=matthews_corrcoef(P.y, P.yhat))


def compare(dA, dB, label_a, label_b, n_boot=1000):
    """Paired comparison of two arms evaluated on identical folds (A minus B)."""
    from scipy.stats import wilcoxon
    from sklearn.metrics import roc_auc_score, f1_score
    import afis_core as AC
    C, K = AC.import_pipeline()
    key = ["seed", "fold"]
    m = dA.merge(dB, on=key, suffixes=("_a", "_b"))
    res = dict(A=label_a, B=label_b, n_pairs=len(m))
    for met in ("f1_macro", "roc_auc_macro", "mcc"):
        diff = m[f"{met}_a"] - m[f"{met}_b"]
        res[f"d_{met}_mean"] = float(diff.mean()); res[f"d_{met}_sd"] = float(diff.std())
        try:
            res[f"p_wilcoxon_{met}"] = float(wilcoxon(m[f"{met}_a"], m[f"{met}_b"]).pvalue) if (diff != 0).any() else 1.0
        except ValueError:
            res[f"p_wilcoxon_{met}"] = np.nan
    PA, PB = _pooled(dA), _pooled(dB)
    seeds = sorted(set(PA) & set(PB))
    dl, mc = [], []
    merged = {}
    for s in seeds:
        mm = PA[s].merge(PB[s], on="Filename", suffixes=("_a", "_b"))
        merged[s] = mm
        a1, a2, p = _delong(mm.y_a, mm.Probability_Cancer_a, mm.Probability_Cancer_b)
        dl.append(p); mc.append(_mcnemar(mm.y_a.values, mm.yhat_a.values, mm.yhat_b.values))
        res[f"pooledAUC_a_s{s}"], res[f"pooledAUC_b_s{s}"] = a1, a2
        res[f"p_delong_s{s}"] = p; res[f"p_mcnemar_s{s}"] = mc[-1]
    if seeds:
        res["p_delong_median"] = float(np.median(dl)); res["p_mcnemar_median"] = float(np.median(mc))
        files = np.array(sorted(set.intersection(*[set(merged[s].Filename) for s in seeds])))
        groups = np.array([K.region_group(f) for f in files]); ug = np.unique(groups)
        gidx = {g: np.where(groups == g)[0] for g in ug}
        arr = {}
        for s in seeds:
            mm = merged[s].drop_duplicates("Filename").set_index("Filename").loc[files]
            arr[s] = (mm.y_a.values, mm.Probability_Cancer_a.values, mm.Probability_Cancer_b.values,
                      mm.yhat_a.values, mm.yhat_b.values)
        rng = np.random.default_rng(12345); dA_, dF_ = [], []
        for _ in range(n_boot):
            sel = np.concatenate([gidx[g] for g in rng.choice(ug, len(ug))])
            da, df_ = [], []
            for s in seeds:
                y, pa, pb, ha, hb = (v[sel] for v in arr[s])
                if len(np.unique(y)) < 2:
                    continue
                da.append(roc_auc_score(y, pa) - roc_auc_score(y, pb))
                df_.append(f1_score(y, ha, average="macro") - f1_score(y, hb, average="macro"))
            if da:
                dA_.append(np.mean(da)); dF_.append(np.mean(df_))
        res["boot_dAUC_mean"] = float(np.mean(dA_)); res["boot_dAUC_CI"] = f"[{np.percentile(dA_, 2.5):+.3f}, {np.percentile(dA_, 97.5):+.3f}]"
        res["boot_dF1_mean"] = float(np.mean(dF_)); res["boot_dF1_CI"] = f"[{np.percentile(dF_, 2.5):+.3f}, {np.percentile(dF_, 97.5):+.3f}]"
    return res


def _sel(rec, family, model, exp, seeds=None):
    d = rec[(rec.family == family) & (rec.model == model) & (rec.experiment == exp)]
    if seeds is not None:
        d = d[d.seed.isin(seeds)]
    return d


def _fmt_ms(x):
    return f"{np.mean(x):.3f} $\\pm$ {np.std(x, ddof=1) if len(x) > 1 else 0:.3f}"


def statistics_and_figures(paths):
    out = A.RESULTS / "stats"; out.mkdir(parents=True, exist_ok=True)
    rec = load_records()
    if rec.empty:
        print("No completed runs yet."); return
    A.import_pipeline(paths)
    rec.to_csv(out / "all_runs.csv", index=False)
    # ---------------- summary of every arm ----------------
    rows = []
    for (fam, model, exp), d in rec.groupby(["family", "model", "experiment"]):
        r = dict(family=fam, model=model, experiment=exp, n_runs=len(d), seeds=",".join(map(str, sorted(d.seed.unique()))))
        for m in TABLE_METRICS:
            r[f"{m}_mean"] = d[m].mean(); r[f"{m}_sd"] = d[m].std(ddof=1) if len(d) > 1 else 0.0
        for s, P in _pooled(d).items():
            for k, v in _pooled_metrics(P).items():
                r[f"{k}_s{s}"] = v
        r["minutes_per_fold"] = d.wall_seconds.mean() / 60
        rows.append(r)
    summ = pd.DataFrame(rows); summ.to_csv(out / "summary_all_arms.csv", index=False)
    print(summ[["family", "model", "experiment", "n_runs", "f1_macro_mean", "f1_macro_sd",
                "roc_auc_macro_mean", "minutes_per_fold"]].round(3).to_string(index=False))
    comps = []
    # ---------------- E1: Table 3 replacement + raw-vs-synthetic + model comparisons -------------
    e1 = rec[rec.family == "E1_kfold"]
    if not e1.empty:
        trows = []
        for model in A.KFOLD_MODELS:
            for exp, lab in (("Raw", "Raw"), (A.SYN_EXP, "Raw+Synthetic")):
                d = _sel(rec, "E1_kfold", model, exp)
                if d.empty:
                    continue
                trows.append({"Model": "AFiS-Net" if model == "AFiSNet" else model, "Training": lab,
                              "runs": f"{len(d)} ({d.seed.nunique()} seeds)",
                              **{SHORT[m]: _fmt_ms(d[m]) for m in ("f1_macro", "roc_auc_macro", "mcc",
                                                                   "Cancer_recall_sensitivity", "ece")}})
        t = pd.DataFrame(trows); t.to_csv(out / "table_E1_kfold.csv", index=False)
        _df_to_latex(t, out / "table_E1_kfold.tex",
                     "Region-grouped five-fold cross-validation repeated with three seeds "
                     "(mean $\\pm$ s.d. over fold $\\times$ seed runs).", "tab:kfold")
        for model in A.KFOLD_MODELS:
            dR, dS = _sel(rec, "E1_kfold", model, "Raw"), _sel(rec, "E1_kfold", model, A.SYN_EXP)
            if not dR.empty and not dS.empty:
                comps.append(dict(family="E1", **compare(dS, dR, f"{model} Raw+Syn", f"{model} Raw")))
        for exp in ("Raw", A.SYN_EXP):
            dA_ = _sel(rec, "E1_kfold", "AFiSNet", exp)
            for model in A.KFOLD_MODELS[1:]:
                dB = _sel(rec, "E1_kfold", model, exp)
                if not dA_.empty and not dB.empty:
                    comps.append(dict(family=f"E1_models_{exp}", **compare(dA_, dB, f"AFiSNet {exp}", f"{model} {exp}")))
        _fig_E1(rec, out)
    afs = _sel(rec, "E1_kfold", "AFiSNet", A.SYN_EXP); afr = _sel(rec, "E1_kfold", "AFiSNet", "Raw")
    # ---------------- E3 synthetic source ----------------
    e3 = rec[rec.family == "E3_synthetic_source"]
    for exp in sorted(e3.experiment.unique()) if not e3.empty else []:
        d = _sel(rec, "E3_synthetic_source", "AFiSNet", exp)
        if not afr.empty:
            comps.append(dict(family="E3_vs_Raw", **compare(d, afr, exp, "Raw")))
        if not afs.empty:
            comps.append(dict(family="E3_vs_QCselected", **compare(afs, d, "QC-selected NTP", exp)))
    # ---------------- E6 fold-aware ----------------
    e6 = _sel(rec, "E6_foldaware", "AFiSNet", "Raw_FoldAwareSelected")
    if not e6.empty and not afs.empty:
        comps.append(dict(family="E6", **compare(e6, afs, "Fold-aware selected", "QC-selected (fixed set)")))
        if not afr.empty:
            comps.append(dict(family="E6", **compare(e6, afr, "Fold-aware selected", "Raw")))
        drops = []
        for _, r in e6.iterrows():
            info = json.loads((Path(r["_dir"]) / "split_manifest.json").read_text())["synthetic_info"]
            drops.append(dict(seed=r.seed, fold=r.fold, n_selected=info.get("n_selected"),
                              n_dropped=info.get("n_dropped_near_heldout")))
        pd.DataFrame(drops).to_csv(out / "E6_dropped_per_fold.csv", index=False)
    # ---------------- E4 ablation ----------------
    for model in A.ABLATION_MODELS:
        d = _sel(rec, "E4_ablation", model, A.SYN_EXP)
        if not d.empty and not afs.empty:
            comps.append(dict(family="E4", **compare(afs, d, "AFiS-Net (full)", model)))
    # ---------------- E5 scaling ----------------
    e5 = rec[rec.family == "E5_scaling"]
    for frac in A.SCALING_FRACTIONS:
        tag = f"frac{int(round(frac * 100)):03d}"
        dR, dS = _sel(rec, "E5_scaling", "AFiSNet", f"Raw_{tag}"), _sel(rec, "E5_scaling", "AFiSNet", f"{A.SYN_EXP}_{tag}")
        if not dR.empty and not dS.empty:
            comps.append(dict(family=f"E5_{tag}", **compare(dS, dR, f"Raw+Syn {tag}", f"Raw {tag}")))
    if comps:
        cdf = pd.DataFrame(comps)
        for fam, idx in cdf.groupby("family").groups.items():
            for col in ("p_wilcoxon_f1_macro", "p_delong_median"):
                if col in cdf:
                    cdf.loc[idx, col + "_holm"] = _holm(cdf.loc[idx, col].fillna(1).values)
        cdf.to_csv(out / "paired_comparisons.csv", index=False)
        keep = [c for c in ["family", "A", "B", "n_pairs", "d_f1_macro_mean", "d_f1_macro_sd", "p_wilcoxon_f1_macro",
                            "p_wilcoxon_f1_macro_holm", "d_roc_auc_macro_mean", "p_wilcoxon_roc_auc_macro",
                            "p_delong_median", "boot_dAUC_CI", "boot_dF1_CI", "p_mcnemar_median"] if c in cdf]
        show = cdf[keep].copy()
        _df_to_latex(show.round(4), out / "table_paired_comparisons.tex",
                     "Paired comparisons on identical region-grouped folds: mean difference (A$-$B) over "
                     "fold$\\times$seed pairs with Wilcoxon signed-rank tests (Holm-adjusted within each family), "
                     "paired DeLong tests on pooled out-of-fold predictions (median over seeds), and 95\\% "
                     "region-clustered bootstrap CIs of the pooled AUC and macro-F1 differences.", "tab:paired")
        print(show.round(4).to_string(index=False))
    _fig_families(rec, afs, afr, out)
    _mark_stats_time(out)


def _mark_stats_time(out):
    (out / "generated_at.txt").write_text(time.strftime("%Y-%m-%d %H:%M:%S"))


def _fig_E1(rec, out):
    from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
    e1 = rec[rec.family == "E1_kfold"]
    models = [m for m in A.KFOLD_MODELS if m in set(e1.model)]
    # boxplots of fold-level F1 / AUC
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, met in zip(axes, ("f1_macro", "roc_auc_macro")):
        data, labels, pos = [], [], []
        for i, m in enumerate(models):
            for j, exp in enumerate(("Raw", A.SYN_EXP)):
                d = _sel(rec, "E1_kfold", m, exp)[met].dropna()
                if len(d):
                    data.append(d.values); pos.append(i * 3 + j); labels.append(f"{m}\n{'Raw' if exp == 'Raw' else '+Syn'}")
        if data:
            bp = ax.boxplot(data, positions=pos, widths=0.8, patch_artist=True)
            for k, b in enumerate(bp["boxes"]):
                b.set_facecolor("#9ecae1" if "Raw" in labels[k] and "+Syn" not in labels[k] else "#fdae6b")
            ax.set_xticks(pos); ax.set_xticklabels(labels, fontsize=7)
        ax.set_title(f"{met} per fold (all seeds)")
    _savefig(fig, out / "fig_E1_fold_boxplots")
    # pooled OOF ROC + confusion matrices (paper Fig. 5 replacement, from CV not the fixed split)
    for m in models:
        fig, axes = plt.subplots(1, 3, figsize=(16, 5.0))
        for exp, col in (("Raw", "#3182bd"), (A.SYN_EXP, "#e6550d")):
            P = _pooled(_sel(rec, "E1_kfold", m, exp))
            grid = np.linspace(0, 1, 201); tprs = []
            for s, d in P.items():
                fpr, tpr, _ = roc_curve(d.y, d.Probability_Cancer)
                tprs.append(np.interp(grid, fpr, tpr)); axes[0].plot(fpr, tpr, color=col, alpha=0.25, lw=0.8)
            if tprs:
                aucs = [roc_auc_score(d.y, d.Probability_Cancer) for d in P.values()]
                axes[0].plot(grid, np.mean(tprs, 0), color=col, lw=2,
                             label=f"{'Raw' if exp == 'Raw' else 'Raw+Synthetic'} (pooled AUC {np.mean(aucs):.3f} ± {np.std(aucs):.3f})")
                s0 = sorted(P)[0]; d = P[s0]
                cm = confusion_matrix(d.y, d.yhat, labels=[0, 1])
                ax = axes[1] if exp == "Raw" else axes[2]
                ax.imshow(cm, cmap="Blues")
                for (i, j), v in np.ndenumerate(cm):
                    ax.text(j, i, str(v), ha="center", va="center", color="white" if v > cm.max() / 2 else "black", fontsize=14)
                ax.set_xticks([0, 1]); ax.set_yticks([0, 1]); ax.set_xticklabels(CLASSES); ax.set_yticklabels(CLASSES)
                ax.set_xlabel("Predicted"); ax.set_ylabel("True")
                ax.set_title(f"{'Raw' if exp == 'Raw' else 'Raw+Synthetic'}\npooled out-of-fold, seed {s0}", fontsize=11)
        axes[0].plot([0, 1], [0, 1], "k:", lw=0.8); axes[0].set_xlabel("False positive rate"); axes[0].set_ylabel("True positive rate")
        axes[0].set_title(f"{m}: ROC\npooled out-of-fold predictions", fontsize=11); axes[0].legend(fontsize=8, loc="lower right")
        _savefig(fig, out / f"fig_E1_roc_confusion_{m}")
    # reliability diagrams
    fig, axes = plt.subplots(1, len(models), figsize=(4 * len(models), 4), squeeze=False)
    for ax, m in zip(axes[0], models):
        for exp, col in (("Raw", "#3182bd"), (A.SYN_EXP, "#e6550d")):
            P = _pooled(_sel(rec, "E1_kfold", m, exp))
            if not P:
                continue
            d = pd.concat(P.values()); bins = np.linspace(0, 1, 11); idx = np.clip(np.digitize(d.Probability_Cancer, bins) - 1, 0, 9)
            xs = [d.Probability_Cancer[idx == b].mean() for b in range(10) if (idx == b).any()]
            ys = [d.y[idx == b].mean() for b in range(10) if (idx == b).any()]
            ax.plot(xs, ys, "o-", color=col, label="Raw" if exp == "Raw" else "Raw+Synthetic")
        ax.plot([0, 1], [0, 1], "k:"); ax.set_title(m); ax.set_xlabel("Predicted P(cancer)"); ax.set_ylabel("Observed frequency")
        ax.legend(fontsize=8)
    _savefig(fig, out / "fig_E1_reliability")


def _bar(ax, names, data, title, ref=None):
    means = [np.mean(d) for d in data]; sds = [np.std(d, ddof=1) if len(d) > 1 else 0 for d in data]
    ax.bar(range(len(names)), means, yerr=sds, capsize=3, color="#6baed6", edgecolor="k", lw=0.5)
    for i, d in enumerate(data):
        ax.scatter(np.full(len(d), i) + np.random.default_rng(i).uniform(-0.15, 0.15, len(d)), d, s=6, c="k", alpha=0.5, zorder=3)
    if ref is not None:
        ax.axhline(ref, ls="--", c="grey", lw=0.8)
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=25, ha="right", fontsize=8); ax.set_title(title)
    lo = min(min(d) for d in data if len(d)) if data else 0
    ax.set_ylim(max(0, lo - 0.05), 1.0)


def _fig_families(rec, afs, afr, out):
    # E3 synthetic source
    e3 = rec[rec.family == "E3_synthetic_source"]
    if not e3.empty:
        arms = [("Real only", afr)] + [(e.replace("Raw_", "+ "), _sel(rec, "E3_synthetic_source", "AFiSNet", e))
                                       for e in sorted(e3.experiment.unique())] + [("+ QC-selected (NTP)", afs)]
        arms = [(n, d) for n, d in arms if not d.empty]
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for ax, met in zip(axes, ("f1_macro", "roc_auc_macro")):
            _bar(ax, [n for n, _ in arms], [d[met].values for _, d in arms], f"AFiS-Net: synthetic source ({met})")
        _savefig(fig, out / "fig_E3_synthetic_source")
        t = pd.DataFrame([{"Training data": n, "runs": len(d), **{SHORT[m]: _fmt_ms(d[m]) for m in ("f1_macro", "roc_auc_macro", "mcc", "Cancer_recall_sensitivity", "ece")}} for n, d in arms])
        _df_to_latex(t, out / "table_E3_synthetic_source.tex", "Effect of the synthetic data source on AFiS-Net "
                     "(region-grouped CV, mean $\\pm$ s.d. over fold $\\times$ seed runs).", "tab:synsource")
        t.to_csv(out / "table_E3_synthetic_source.csv", index=False)
    # E4 ablation
    e4 = rec[rec.family == "E4_ablation"]
    if not e4.empty and not afs.empty:
        names = {"AFiSNet_CNNonly": "ConvNeXt V2 branch only", "AFiSNet_SwinOnly": "Swin V2 branch only",
                 "AFiSNet_Concat": "No cross-attention (GAP concat)", "AFiSNet_AvgPool": "Average pooling (no GeM)",
                 "AFiSNet_noSupCon": "No supervised contrastive term"}
        arms = [("AFiS-Net (full)", afs)] + [(names[m], _sel(rec, "E4_ablation", m, A.SYN_EXP)) for m in A.ABLATION_MODELS]
        arms = [(n, d) for n, d in arms if not d.empty]
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for ax, met in zip(axes, ("f1_macro", "roc_auc_macro")):
            _bar(ax, [n for n, _ in arms], [d[met].values for _, d in arms], f"AFiS-Net ablation ({met})",
                 ref=afs[met].mean())
        _savefig(fig, out / "fig_E4_ablation")
        t = pd.DataFrame([{"Variant": n, "runs": len(d), **{SHORT[m]: _fmt_ms(d[m]) for m in ("f1_macro", "roc_auc_macro", "mcc", "Cancer_recall_sensitivity", "ece")}} for n, d in arms])
        _df_to_latex(t, out / "table_E4_ablation.tex", "AFiS-Net component ablation (Raw+Synthetic training, "
                     "region-grouped CV, mean $\\pm$ s.d. over fold $\\times$ seed runs).", "tab:ablation")
        t.to_csv(out / "table_E4_ablation.csv", index=False)
    # E5 scaling
    e5 = rec[rec.family == "E5_scaling"]
    if not e5.empty:
        e5_seeds = sorted(int(x) for x in e5.seed.unique())
        s0 = ",".join(str(x) for x in e5_seeds)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4)); trow = []
        for ax, met in zip(axes, ("f1_macro", "roc_auc_macro")):
            for lab, base, col, dx in (("Raw", "Raw", "#3182bd", -1.0), ("Raw+Synthetic", A.SYN_EXP, "#e6550d", 1.0)):
                xs, ms, ss = [], [], []
                for frac in A.SCALING_FRACTIONS + [1.0]:
                    if frac == 1.0:
                        d = _sel(rec, "E1_kfold", "AFiSNet", base, seeds=e5_seeds)
                    else:
                        d = _sel(rec, "E5_scaling", "AFiSNet", f"{base}_frac{int(round(frac * 100)):03d}")
                    if len(d):
                        xs.append(frac * 100 + dx); ms.append(d[met].mean()); ss.append(d[met].std(ddof=1) if len(d) > 1 else 0)
                        if met == "f1_macro":
                            trow.append(dict(fraction=f"{int(frac * 100)}%", arm=lab, **{SHORT[m]: _fmt_ms(d[m]) for m in ("f1_macro", "roc_auc_macro", "mcc")}))
                if xs:
                    ax.errorbar(xs, ms, yerr=ss, marker="o", capsize=3, color=col, label=lab)
            ax.set_xlabel("% of real training regions used"); ax.set_title(f"Data scaling: {dict(f1_macro='macro F1', roc_auc_macro='AUC')[met]} (seeds {s0})"); ax.legend()
            ax.set_xticks([f * 100 for f in A.SCALING_FRACTIONS + [1.0]])
        _savefig(fig, out / "fig_E5_scaling")
        if trow:
            t = pd.DataFrame(trow); t.to_csv(out / "table_E5_scaling.csv", index=False)
            _df_to_latex(t, out / "table_E5_scaling.tex", "AFiS-Net performance with reduced real training data "
                         "(region-grouped CV, mean $\\pm$ s.d. over fold $\\times$ seed runs; the same QC-selected synthetic set is "
                         "added at every fraction).", "tab:scaling")
    # E6
    e6 = _sel(rec, "E6_foldaware", "AFiSNet", "Raw_FoldAwareSelected")
    if not e6.empty and not afs.empty:
        arms = [(n, d) for n, d in (("Real only", afr), ("QC-selected (fixed)", afs), ("QC-selected, fold-aware", e6)) if not d.empty]
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, met in zip(axes, ("f1_macro", "roc_auc_macro")):
            _bar(ax, [n for n, _ in arms], [d[met].values for _, d in arms], f"Fold-aware synthetic selection ({met})")
        _savefig(fig, out / "fig_E6_foldaware")


# =============================================================================
# 10. SESSION 6+: QC re-run on all NTP images (preparation for E3_qc_scale)
# =============================================================================
_QC_SMOKE_PATCH = r'''
import torch
from transformers import Dinov2Config, Dinov2Model, CLIPConfig, CLIPModel, CLIPImageProcessor
torch.manual_seed(0)
_dm = Dinov2Model(Dinov2Config(hidden_size=32, num_hidden_layers=1, num_attention_heads=2,
                               intermediate_size=64, image_size=224, patch_size=14)).eval()
_cm = CLIPModel(CLIPConfig(text_config=dict(hidden_size=32, num_hidden_layers=1, num_attention_heads=2,
                                            intermediate_size=64),
                           vision_config=dict(hidden_size=32, num_hidden_layers=1, num_attention_heads=2,
                                              intermediate_size=64, image_size=224, patch_size=32),
                           projection_dim=16)).eval()
_pp = CLIPImageProcessor()
Q.ModelRegistry.set("dino", _dm, _pp); Q.ModelRegistry.set("clip", _cm, _pp)
Q.HAS_LPIPS = False
Q.COPY_SIMILARITY_THRESHOLD = 1.01; Q.DIVERSITY_THRESHOLD = 1.01   # random tiny embeddings: everything looks alike
'''


def qc_all_ntp(paths):
    """Re-run the OFFICIAL gan_quality_selection.py on all NTP images with the same scores, copy/
    class-margin/diversity filters, but keeping the top 80 % instead of the 0.5 x real cap.
    This gives a QC-filtered set as large as the unfiltered pool, so QC can be tested at scale."""
    sel = A.qc_all_dir()
    if A.qc_all_ready():
        print("QC-at-scale selection already available:", sel); return True
    if "synthetic_all_ntp" not in paths:
        print("No synthetic_all_ntp folder -> QC-at-scale experiment skipped."); return False
    if sel.exists():
        shutil.rmtree(sel)                                   # remove a partial earlier attempt
    sel.parent.mkdir(parents=True, exist_ok=True)
    # the QC script matches class folders by exact name -> lower-case symlink mirrors
    mir = A.WORK / "qc_inputs"
    if mir.exists():
        shutil.rmtree(mir)
    for tag, root in (("real", paths["train"]), ("ntp", paths["synthetic_all_ntp"])):
        for c in CLASSES:
            src = A.ci_dir(os.path.join(root, c))
            if not src:
                print(f"Missing class folder {c} in {root} -> QC-at-scale skipped."); return False
            (mir / tag).mkdir(parents=True, exist_ok=True)
            os.symlink(src, mir / tag / c, target_is_directory=True)
    code = ("import sys; sys.path.insert(0, %r); import gan_quality_selection as Q\n"
            "Q.REAL_FOLDER = %r; Q.GAN_FOLDER = %r; Q.OUTPUT_FOLDER = %r\n"
            "Q.TOP_PERCENT = %r; Q.MAX_SELECTED_TO_REAL_RATIO = 1e9\n"
            "Q.ENABLE_UMAP = False; Q.ENABLE_TSNE = False\n"
            % (str(A.REPO_DIR), str(mir / "real"), str(mir / "ntp"), str(sel), A.QC_ALL_TOP_PERCENT))
    if A.SMOKE:
        code += _QC_SMOKE_PATCH
    code += "Q._validate_pipeline(); Q.main()\n"
    print(f"Running the official QC script on all NTP images (top {A.QC_ALL_TOP_PERCENT:.0%}, no ratio cap) ...")
    r = subprocess.run([sys.executable, "-c", code], cwd=str(A.WORK))
    ok = r.returncode == 0 and (sel / "summary.csv").exists() and all(
        len(_list_images(str(sel / c))) > 0 for c in CLASSES)
    if ok:
        print(pd.read_csv(sel / "summary.csv")[["Class", "Number_Real", "Number_Generated",
                                                "Number_Selected", "Number_Duplicates_Removed",
                                                "Number_Near_Real_Copies_Rejected",
                                                "Number_Cross_Class_Ambiguous_Rejected"]].to_string(index=False))
    else:
        print("QC-at-scale selection FAILED (see log above) -> E3_qc_scale jobs are skipped this session.")
        if (sel / "summary.csv").exists():
            (sel / "summary.csv").unlink()
    return ok


# =============================================================================
# 11. E8 v2: GAN metrics at MATCHED sample size (fixes v1's size bias and KID subset bug)
# =============================================================================
def gan_metrics_v2(paths, n_rep=100, force=False):
    """FID, KID and PRDC depend on how many generated images are compared, so the 92/123
    QC-selected images cannot be compared with 1,500 unfiltered ones directly. Here every set is
    evaluated on random subsets of the SAME size m (80 % of the smallest set, without replacement)
    against the full real training set of that class."""
    out = A.RESULTS / "gan_metrics_v2"
    if _done("gan_metrics_v2") and not force:
        print("GAN metrics v2 already computed."); return
    out.mkdir(parents=True, exist_ok=True)
    if A.SMOKE:
        n_rep = 5
    dev = _dev()
    real = {c: _list_images(os.path.join(paths["train"], c)) for c in CLASSES}
    sets = {"NTP QC-selected (paper)": class_sets(paths, "synthetic_selected"),
            "NTP all (unfiltered)": class_sets(paths, "synthetic_all_ntp"),
            "Baseline all": class_sets(paths, "synthetic_all_baseline"),
            "Baseline QC-selected": class_sets(paths, "baseline_selected_qc")}
    if A.qc_all_ready():
        sets["NTP QC top-80% of all"] = {c: _list_images(str(A.qc_all_dir() / c)) for c in CLASSES}
    if A.qc_clean_ready():
        sets["NTP QC-clean (hygiene filter)"] = {c: _list_images(str(A.qc_clean_dir() / c)) for c in CLASSES}
    sets = {k: v for k, v in sets.items() if all(len(v[c]) >= 10 for c in CLASSES)}
    if not sets:
        print("No generated sets with >= 10 images per class -> skipped."); return
    rows, drows = [], []
    for c in CLASSES:
        fr = _inception_features(real[c], dev)
        fg_all = {name: _inception_features(s[c], dev) for name, s in sets.items()}
        m = max(10, int(0.8 * min(len(v) for v in fg_all.values())))
        k_kid = min(100, len(fr), m)
        rng = np.random.default_rng(2026)
        kid_r = [rng.choice(len(fr), k_kid, replace=False) for _ in range(100)]
        reps = {}
        for name, fg in fg_all.items():
            r2 = np.random.default_rng(zlib.crc32(f"{c}|{name}".encode()))
            idx = [r2.choice(len(fg), m, replace=False) for _ in range(n_rep)]
            fids = np.array([_fid(fr, fg[i]) for i in idx])
            kid, _ = _kid_subsets(fr, fg, n_sub=100, size=k_kid, rng=np.random.default_rng(7), idx_r=kid_r)
            pr = pd.DataFrame([_prdc(fr, fg[i], k=5) for i in idx[:20]]).mean().to_dict()
            reps[name] = fids
            rows.append(dict(cls=c, set=name, n_real=len(fr), n_gen_total=len(fg), matched_n=m,
                             FID_all_images=_fid(fr, fg), FID_matched_mean=fids.mean(),
                             FID_matched_low=np.percentile(fids, 2.5), FID_matched_high=np.percentile(fids, 97.5),
                             KID_subset_size=k_kid, KID_mean=kid.mean(), KID_sd=kid.std(),
                             **{f"{k}_matched": v for k, v in pr.items()}))
        ref = "NTP QC-selected (paper)"
        if ref in reps:
            for name in reps:
                if name == ref:
                    continue
                d = reps[name] - reps[ref]
                drows.append(dict(cls=c, comparison=f"{name} - {ref}", matched_n=m, dFID_mean=d.mean(),
                                  dFID_low=np.percentile(d, 2.5), dFID_high=np.percentile(d, 97.5),
                                  frac_reps_lower_FID=float((d < 0).mean())))
    df = pd.DataFrame(rows); df.to_csv(out / "gan_metrics_matched_n.csv", index=False)
    if drows:
        pd.DataFrame(drows).to_csv(out / "fid_differences_vs_qc_selected.csv", index=False)
    show = df.copy()
    show["FID (matched n) [95% range]"] = show.apply(
        lambda r: f"{r.FID_matched_mean:.1f} [{r.FID_matched_low:.1f}, {r.FID_matched_high:.1f}]", axis=1)
    show["KID"] = show.apply(lambda r: f"{r.KID_mean:.4f} $\\pm$ {r.KID_sd:.4f}", axis=1)
    show = show[["cls", "set", "n_gen_total", "matched_n", "FID (matched n) [95% range]", "KID",
                 "precision_matched", "recall_matched", "density_matched", "coverage_matched"]]
    _df_to_latex(show.round(3), out / "table_gan_metrics_matched_n.tex",
                 "Distributional quality of the generated sets at matched sample size (Inception-v3 features). "
                 "FID: mean and 2.5--97.5 percentile over random subsets of equal size drawn without replacement; "
                 "KID: mean $\\pm$ s.d. over 100 equal-size subsets; precision/recall/density/coverage ($k=5$) "
                 "averaged over 20 equal-size subsets. Reference: real training images of the class.",
                 "tab:gan_matched")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, c in zip(axes, CLASSES):
        d = df[df.cls == c].reset_index(drop=True)
        ax.bar(range(len(d)), d.FID_matched_mean,
               yerr=[d.FID_matched_mean - d.FID_matched_low, d.FID_matched_high - d.FID_matched_mean],
               capsize=3, color="#6baed6", edgecolor="k", lw=0.5)
        ax.set_xticks(range(len(d))); ax.set_xticklabels(d.set, rotation=20, ha="right", fontsize=8)
        ax.set_ylabel(f"FID at n = {int(d.matched_n.iloc[0])}"); ax.set_title(f"{c}: FID at matched sample size")
    _savefig(fig, out / "fig_fid_matched_n")
    print(show.to_string(index=False))
    _mark("gan_metrics_v2")


# =============================================================================
# 12. E7 v2: explainability that covers the whole network
# =============================================================================
def _occlusion_map(model, x, cls, patch, stride, chunk=64):
    """Drop of the logit margin (target - other class) when a patch is set to black."""
    import torch
    H, W = x.shape[-2:]
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
    black = (torch.zeros_like(x) - mean) / std
    ys = list(range(0, H - patch + 1, stride)); xs = list(range(0, W - patch + 1, stride))
    if ys[-1] != H - patch:
        ys.append(H - patch)
    if xs[-1] != W - patch:
        xs.append(W - patch)
    pos = [(a, b) for a in ys for b in xs]
    use_amp = x.is_cuda

    def margin(batch):
        with torch.no_grad(), torch.autocast("cuda", enabled=use_amp):
            lo = model(batch).float()
        return (lo[:, cls] - lo[:, 1 - cls]).cpu().numpy()
    base = float(margin(x)[0])
    drops = []
    for i in range(0, len(pos), chunk):
        bt = x.repeat(len(pos[i:i + chunk]), 1, 1, 1).clone()
        for j, (a, b) in enumerate(pos[i:i + chunk]):
            bt[j, :, a:a + patch, b:b + patch] = black[0, :, a:a + patch, b:b + patch]
        drops.append(base - margin(bt))
    drops = np.concatenate(drops)
    acc = np.zeros((H, W), np.float32); cnt = np.zeros((H, W), np.float32)
    for (a, b), d in zip(pos, drops):
        acc[a:a + patch, b:b + patch] += d; cnt[a:a + patch, b:b + patch] += 1
    return acc / np.maximum(cnt, 1), base


def _multilevel_cam(C, model, x, cls):
    """Grad-CAM++ on every FPN level of the CNN branch (not only the finest one), each level
    normalised and averaged over the levels that give a non-empty map."""
    layers = [(n, m) for n, m in model.named_modules() if ".fpn.smooth." in f".{n}" and n.split(".")[-1].isdigit()
              and n.split(".")[-2] == "smooth"]
    maps, valid = [], 0
    for _, layer in layers:
        cammer = C.GradCAMpp(model, layer)
        try:
            import torch
            with torch.enable_grad():
                cam, _ = cammer(x, class_idx=cls)
        finally:
            cammer.remove()
        up = C._upsample_cam(cam, x.shape[-2], x.shape[-1])
        if np.isfinite(up).all() and up.max() > 1e-6:
            maps.append(up / up.max()); valid += 1
    if not maps:
        return np.zeros(x.shape[-2:], np.float32), 0, len(layers)
    m = np.mean(maps, 0)
    return (m - m.min()) / (m.max() - m.min() + 1e-8), valid, len(layers)


def explain_v2(paths, force=False, arms=None, tag="explainability_v2", flag="explain_v2"):
    """Occlusion sensitivity (model-agnostic: covers both the ConvNeXt and the Swin branch) and
    multi-level Grad-CAM++ over all FPN levels, quantified against the cell mask on every test image
    of the seed-42 folds, for AFiS-Net trained on real only and on real + QC-selected synthetic."""
    import torch
    out = A.RESULTS / tag
    if _done(flag) and not force:
        print(f"{tag} already done."); return
    arms = arms or {"Raw": _seed_ckpts("Raw"), "Raw+Synthetic": _seed_ckpts(A.SYN_EXP)}
    arms = {k: v for k, v in arms.items() if len(v) == A.N_FOLDS}
    if not arms:
        print("AFiS-Net seed-42 checkpoints not found -> explainability v2 skipped "
              "(attach the previous notebook output so the checkpoints are restored)."); return
    C, K = A.import_pipeline(paths)
    out.mkdir(parents=True, exist_ok=True)
    dev = _dev(); pool = real_pool(paths)
    cfg = C.Config(); _, eval_tf = C.build_transforms(224, cfg); crop = C.ForegroundCrop()
    patch, stride = (112, 56) if A.SMOKE else (32, 16)
    steps = 4 if A.SMOKE else 10
    rng = np.random.default_rng(0)
    rows, store = [], {}
    for arm, ck in arms.items():
        for f, path in sorted(ck.items()):
            model, ckd = load_ckpt(C, path, dev)
            T = float(ckd["decision"]["temperature"]); thr = float(ckd["decision"]["cancer_threshold"])
            man = json.loads((path.parent / "split_manifest.json").read_text())
            test = man["test"][:4] if A.SMOKE else man["test"]
            for b in test:
                p, c, _ = pool[b]
                pil = Image.open(p).convert("RGB")
                x = eval_tf(pil).unsqueeze(0).to(dev)
                base = np.asarray(crop(pil).resize((224, 224), Image.BICUBIC), dtype=np.float32) / 255.0
                with torch.no_grad():
                    pc = float(torch.softmax(model(x).float() / max(T, 1e-6), 1)[0, 1])
                pred = int(pc >= thr); cls = pred
                occ, margin0 = _occlusion_map(model, x, cls, patch, stride)
                occ_pos = np.clip(occ, 0, None)
                occ_ok = bool(occ_pos.max() > 1e-6)
                occ_n = occ_pos / occ_pos.max() if occ_ok else occ_pos
                mlc, n_valid, n_levels = _multilevel_cam(C, model, x, cls)
                mask = C.tissue_mask(base, cfg.cell_thresh)
                r = dict(arm=arm, fold=f, image=b, true=c, pred=CLASSES[pred], p_cancer=pc, correct=CLASSES[pred] == c,
                         logit_margin=margin0, cell_area_fraction=float(mask.mean()),
                         occl_valid=occ_ok, mlcam_valid_levels=n_valid, mlcam_levels=n_levels)
                for tag, mp, ok in (("occl", occ_n, occ_ok), ("mlcam", mlc, n_valid > 0)):
                    if ok:
                        r[f"{tag}_attention_ratio"] = float((mp * mask).sum() / mp.sum())
                        r[f"{tag}_pointing_hit"] = float(mask.reshape(-1)[int(np.argmax(mp))])
                        di = _del_ins(model, x, mp, cls, 1.0, rng, steps=steps)
                        r.update({f"{tag}_{k}": v for k, v in di.items()})
                    else:
                        r[f"{tag}_attention_ratio"] = r[f"{tag}_pointing_hit"] = float("nan")
                        r.update({f"{tag}_{a}_auc_{o}": float("nan") for a in ("deletion", "insertion")
                                  for o in ("cam", "random")})
                rows.append(r)
                store[(arm, b)] = (base.astype(np.float16), occ_n.astype(np.float16), mlc.astype(np.float16),
                                   mask, pc, CLASSES[pred])
            del model
            if dev.type == "cuda":
                torch.cuda.empty_cache()
    df = pd.DataFrame(rows); df.to_csv(out / "per_image_explainability_v2.csv", index=False)
    summ = []
    for (arm, cl), d in df.groupby(["arm", "true"]):
        for tag, name in (("occl", "Occlusion"), ("mlcam", "Multi-level Grad-CAM++")):
            v = d[d[f"{tag}_attention_ratio"].notna()]
            s = dict(arm=arm, cls=cl, method=name, n=len(d), n_valid=len(v))
            for k in ("attention_ratio", "deletion_auc_cam", "deletion_auc_random", "insertion_auc_cam",
                      "insertion_auc_random"):
                s[k] = f"{v[f'{tag}_{k}'].mean():.3f} ± {v[f'{tag}_{k}'].std():.3f}" if len(v) else ""
            s["cell_area_fraction"] = f"{v.cell_area_fraction.mean():.3f}" if len(v) else ""
            s["pointing_game"] = round(float(v[f"{tag}_pointing_hit"].mean()), 3) if len(v) else float("nan")
            s["p_attention>area"] = _wil(v[f"{tag}_attention_ratio"], v.cell_area_fraction, "greater")
            s["p_deletion<random"] = _wil(v[f"{tag}_deletion_auc_cam"], v[f"{tag}_deletion_auc_random"], "less")
            s["p_insertion>random"] = _wil(v[f"{tag}_insertion_auc_cam"], v[f"{tag}_insertion_auc_random"], "greater")
            summ.append(s)
    sd = pd.DataFrame(summ); sd.to_csv(out / "summary_explainability_v2.csv", index=False)
    _df_to_latex(sd[["arm", "cls", "method", "n_valid", "attention_ratio", "cell_area_fraction", "pointing_game",
                     "deletion_auc_cam", "deletion_auc_random", "insertion_auc_cam", "insertion_auc_random"]],
                 out / "table_explainability_v2.tex",
                 "Explanation faithfulness and cell localisation of AFiS-Net over all test images of the five "
                 "seed-42 folds. Occlusion: drop of the logit margin when a 32$\\times$32 patch is blacked out "
                 "(stride 16); multi-level Grad-CAM++: mean of the normalised maps of all FPN levels. "
                 "Attention ratio = share of the map inside the cell mask; deletion/insertion AUC compared "
                 "with a random pixel order.", "tab:explain_v2")
    if len(arms) == 2:                      # same images, Raw vs Raw+Synthetic model
        prs = []
        for tag in ("occl", "mlcam"):
            pv = df.pivot_table(index="image", columns="arm", values=f"{tag}_attention_ratio").dropna()
            if len(pv) > 1:
                prs.append(dict(method=tag, n=len(pv), raw=pv["Raw"].mean(), raw_plus_syn=pv["Raw+Synthetic"].mean(),
                                p_wilcoxon=_wil(pv["Raw+Synthetic"], pv["Raw"])))
        pd.DataFrame(prs).to_csv(out / "raw_vs_synthetic_attention.csv", index=False)
    deg = df.groupby("arm").agg(n=("image", "size"), occlusion_empty=("occl_valid", lambda x: int((~x.astype(bool)).sum())),
                                mlcam_no_valid_level=("mlcam_valid_levels", lambda x: int((x == 0).sum())),
                                mlcam_mean_valid_levels=("mlcam_valid_levels", "mean"))
    deg.to_csv(out / "empty_map_counts.csv")
    print(sd.to_string(index=False)); print(deg.to_string())
    # ---- galleries (15 per category)
    gdir = out / "gallery"
    syn = df[(df.arm == ("Raw+Synthetic" if "Raw+Synthetic" in arms else "Raw")) & df.occl_valid]
    arm0 = syn.arm.iloc[0] if len(syn) else None

    def _export(sub, name):
        items = []
        for k, (_, r) in enumerate(sub.head(N_GALLERY).iterrows(), 1):
            base, occ, mlc, mask, pc, pr = store[(arm0, r.image)]
            base = base.astype(np.float32); occ = occ.astype(np.float32); mlc = mlc.astype(np.float32)
            o1, o2, o3 = _overlay(base, occ), _overlay(base, mlc), _cellfocused(base, occ, mask)
            stem = gdir / name / f"{k:02d}_{os.path.splitext(r.image)[0]}"
            _save_img(base, f"{stem}_input.png"); _save_img(o1, f"{stem}_occlusion.png")
            _save_img(o2, f"{stem}_multilevel_gradcampp.png"); _save_img(o3, f"{stem}_occlusion_cellfocused.png")
            items.append((base, o1, o2, o3, f"#{k} GT {r.true} / pred {r.pred} (P={r.p_cancer:.2f})"))
        _pair_sheet(items, ["Input", "Occlusion", "Multi-level Grad-CAM++", "Occlusion (cell-focused)"],
                    gdir / f"sheet_{name}")
    if arm0 is not None:
        _export(syn[(syn.true == "cancer") & syn.correct].sort_values("p_cancer", ascending=False), "correct_cancer")
        _export(syn[(syn.true == "normal") & syn.correct].sort_values("p_cancer"), "correct_normal")
        _export(syn[(syn.true == "cancer") & ~syn.correct].sort_values("p_cancer"), "misclassified_cancer")
        _export(syn[(syn.true == "normal") & ~syn.correct].sort_values("p_cancer", ascending=False), "misclassified_normal")
    if len(arms) == 2:
        okd = df[df.occl_valid]
        pv = okd.pivot_table(index="image", columns="arm", values="correct", aggfunc="first").dropna()
        both = pv[pv["Raw"].astype(bool) & pv["Raw+Synthetic"].astype(bool)].index
        for c in CLASSES:
            sub = syn[syn.image.isin(both) & (syn.true == c)].sort_values("p_cancer", ascending=(c == "normal"))
            items = []
            for k, (_, r) in enumerate(sub.head(N_GALLERY).iterrows(), 1):
                b1, o1, _, _, p1, _ = store[("Raw", r.image)]; _, o2, _, _, p2, _ = store[("Raw+Synthetic", r.image)]
                b1 = b1.astype(np.float32)
                v1, v2 = _overlay(b1, o1.astype(np.float32)), _overlay(b1, o2.astype(np.float32))
                stem = gdir / f"raw_vs_synthetic_{c}" / f"{k:02d}_{os.path.splitext(r.image)[0]}"
                _save_img(b1, f"{stem}_input.png"); _save_img(v1, f"{stem}_occlusion_raw_model.png")
                _save_img(v2, f"{stem}_occlusion_synthetic_model.png")
                items.append((b1, v1, v2, f"#{k} P(cancer) raw {p1:.2f} / +syn {p2:.2f}"))
            _pair_sheet(items, ["Input", "Occlusion: AFiS-Net (Raw)", "Occlusion: AFiS-Net (Raw+Synthetic)"],
                        gdir / f"sheet_raw_vs_synthetic_{c}")
    _mark(flag)


# =============================================================================
# 13. Statistics v2: threshold-free metrics, GAN-unseen subset, calibration, new experiments
# =============================================================================
def _run_metrics(rec_row, orig_split):
    from sklearn.metrics import roc_auc_score, f1_score, matthews_corrcoef
    p = load_preds(rec_row)
    p["y05"] = (p.Probability_Cancer >= 0.5).astype(int)
    p["orig"] = p.Filename.map(orig_split)
    out = {}
    for tag, d in (("all", p), ("unseen", p[p.orig.isin(["val", "test"])])):
        if len(d) and d.y.nunique() == 2:
            out[f"auc_{tag}"] = roc_auc_score(d.y, d.Probability_Cancer)
            out[f"f1_05_{tag}"] = f1_score(d.y, d.y05, average="macro")
            out[f"mcc_05_{tag}"] = matthews_corrcoef(d.y, d.y05)
        else:
            out[f"auc_{tag}"] = out[f"f1_05_{tag}"] = out[f"mcc_05_{tag}"] = float("nan")
    out["n_test_unseen"] = int(p.orig.isin(["val", "test"]).sum())
    out["temperature"] = float(p.Temperature.iloc[0]) if "Temperature" in p else float("nan")
    out["cancer_threshold"] = float(p.CancerThreshold.iloc[0]) if "CancerThreshold" in p else float("nan")
    return out, p


def statistics_v2(paths):
    from scipy.stats import wilcoxon
    out = A.RESULTS / "stats_v2"; out.mkdir(parents=True, exist_ok=True)
    rec = load_records()
    if rec.empty:
        print("No finished runs yet."); return
    pool = real_pool(paths)
    orig = {b: ("train" if v[2] == "train" else v[2]) for b, v in pool.items()}
    mets, preds = [], {}
    for _, r in rec.iterrows():
        m, p = _run_metrics(r, orig)
        key = (r.family, r.model, r.experiment, int(r.seed), int(r.fold))
        preds[key] = p
        mets.append(dict(family=r.family, model=r.model, experiment=r.experiment, seed=int(r.seed), fold=int(r.fold),
                         n_synthetic=int(r.get("n_synthetic", 0)) if pd.notna(r.get("n_synthetic", 0)) else 0,
                         f1_val_threshold=r.f1_macro, ece=r.ece, brier=r.brier_score, **m))
    M = pd.DataFrame(mets); M.to_csv(out / "per_run_metrics_v2.csv", index=False)
    cols = ["auc_all", "f1_05_all", "mcc_05_all", "f1_val_threshold", "ece", "brier", "auc_unseen", "f1_05_unseen"]
    summ = M.groupby(["family", "model", "experiment"]).agg(
        runs=("fold", "size"), seeds=("seed", lambda x: ",".join(map(str, sorted(set(x))))),
        n_synthetic=("n_synthetic", "median"),
        **{f"{c}_mean": (c, "mean") for c in cols}, **{f"{c}_sd": (c, "std") for c in cols},
        temperature_at_floor=("temperature", lambda x: int((x <= 0.0501).sum())),
        threshold_min=("cancer_threshold", "min"), threshold_max=("cancer_threshold", "max")).reset_index()
    summ.to_csv(out / "summary_v2.csv", index=False)

    def arm(fam, model, exp, seeds=None):
        d = M[(M.family == fam) & (M.model == model) & (M.experiment == exp)]
        return d[d.seed.isin(seeds)] if seeds is not None else d

    def pooled_auc(fam, model, exp, s):
        ks = [k for k in preds if k[:4] == (fam, model, exp, s)]
        if len(ks) < A.N_FOLDS:
            return None
        return pd.concat([preds[k] for k in ks], ignore_index=True)
    comps = []

    def cmp(group, la, a, lb, b):
        if a.empty or b.empty:
            return
        j = a.merge(b, on=["seed", "fold"], suffixes=("_a", "_b"))
        if len(j) < 2:
            return
        row = dict(group=group, A=la, B=lb, n_pairs=len(j), seeds=",".join(map(str, sorted(j.seed.unique()))))
        for met in ("auc_all", "f1_05_all", "mcc_05_all", "f1_val_threshold", "ece", "brier", "auc_unseen", "f1_05_unseen"):
            d = (j[f"{met}_a"] - j[f"{met}_b"]).dropna()
            row[f"d_{met}"] = d.mean() if len(d) else float("nan")
            row[f"p_{met}"] = float(wilcoxon(d).pvalue) if len(d) > 1 and (d != 0).any() else float("nan")
        dl = []
        for s in sorted(j.seed.unique()):
            ka = a.iloc[0]; kb = b.iloc[0]
            pa = pooled_auc(ka.family, ka.model, ka.experiment, int(s)); pb = pooled_auc(kb.family, kb.model, kb.experiment, int(s))
            if pa is None or pb is None:
                continue
            mm = pa[["Filename", "y", "Probability_Cancer"]].merge(pb[["Filename", "Probability_Cancer"]], on="Filename")
            if mm.y.nunique() == 2:
                dl.append(_delong(mm.y.values, mm.Probability_Cancer_x.values, mm.Probability_Cancer_y.values)[2])
        row["p_delong_pooled_median"] = float(np.median(dl)) if dl else float("nan")
        comps.append(row)
    S = A.SEEDS; s0 = [S[0]]
    for m in A.KFOLD_MODELS:
        cmp("E1 synthetic effect", f"{m} +QC-selected", arm("E1_kfold", m, A.SYN_EXP), f"{m} real only", arm("E1_kfold", m, "Raw"))
    for m in A.KFOLD_MODELS[1:]:
        cmp("E1 AFiS-Net vs baselines (real only)", "AFiS-Net", arm("E1_kfold", "AFiSNet", "Raw"), m, arm("E1_kfold", m, "Raw"))
    e3 = "E3_synthetic_source"
    cmp("E3 synthetic source", "+all 3,000 unfiltered", arm(e3, "AFiSNet", "Raw_AllUnfiltered"), "real only", arm("E1_kfold", "AFiSNet", "Raw"))
    cmp("E3 synthetic source", "+all 3,000 unfiltered", arm(e3, "AFiSNet", "Raw_AllUnfiltered"), "+215 QC-selected", arm("E1_kfold", "AFiSNet", A.SYN_EXP))
    cmp("E3 synthetic source", "+215 random", arm(e3, "AFiSNet", "Raw_RandomUnfiltered"), "real only", arm("E1_kfold", "AFiSNet", "Raw"))
    cmp("E3 synthetic source", "+215 QC-selected", arm("E1_kfold", "AFiSNet", A.SYN_EXP), "+215 random", arm(e3, "AFiSNet", "Raw_RandomUnfiltered"))
    q = "E3_qc_scale"
    cmp("QC at scale (seed 42)", "+QC top-80% of all", arm(q, "AFiSNet", "Raw_QCfilteredAll"), "+random, same count", arm(q, "AFiSNet", "Raw_RandomMatchedQCAll"))
    cmp("QC at scale (seed 42)", "+QC top-80% of all", arm(q, "AFiSNet", "Raw_QCfilteredAll"), "+all 3,000 unfiltered", arm(e3, "AFiSNet", "Raw_AllUnfiltered", s0))
    cmp("QC at scale (seed 42)", "+random, same count", arm(q, "AFiSNet", "Raw_RandomMatchedQCAll"), "+all 3,000 unfiltered", arm(e3, "AFiSNet", "Raw_AllUnfiltered", s0))
    cmp("QC at scale (seed 42)", "+QC top-80% of all", arm(q, "AFiSNet", "Raw_QCfilteredAll"), "real only", arm("E1_kfold", "AFiSNet", "Raw", s0))
    for n in A.DOSE_SIZES:
        cmp("Dose-response (seed 42)", f"+{n} random", arm("E3_dose", "AFiSNet", f"Raw_Random{n:04d}"), "real only", arm("E1_kfold", "AFiSNet", "Raw", s0))
    for m in ("NextViT_Small", "ResNet50"):
        g = arm("E3_generality", m, "Raw_AllUnfiltered")
        cmp("Generality (seed 42)", f"{m} +all 3,000", g, f"{m} real only", arm("E1_kfold", m, "Raw", s0))
        cmp("Generality (seed 42)", f"{m} +all 3,000", g, f"{m} +QC-selected", arm("E1_kfold", m, A.SYN_EXP, s0))
    for frac in A.SCALING_FRACTIONS:
        tag = f"frac{int(round(frac * 100)):03d}"
        cmp("E5 data scaling", f"+QC-selected @{int(frac * 100)}%", arm("E5_scaling", "AFiSNet", f"{A.SYN_EXP}_{tag}"),
            f"real only @{int(frac * 100)}%", arm("E5_scaling", "AFiSNet", f"Raw_{tag}"))
    cmp("E6 fold-aware", "fold-aware selected", arm("E6_foldaware", "AFiSNet", "Raw_FoldAwareSelected"), "+215 QC-selected", arm("E1_kfold", "AFiSNet", A.SYN_EXP))
    for m in A.ABLATION_MODELS:
        cmp("E4 ablation", "AFiS-Net (full)", arm("E1_kfold", "AFiSNet", A.SYN_EXP), m, arm("E4_ablation", m, A.SYN_EXP))
    Cdf = pd.DataFrame(comps)
    if Cdf.empty:
        print("No comparisons available yet."); return
    for met in ("auc_all", "f1_05_all"):
        Cdf[f"p_{met}_holm"] = np.nan
        for g, idx in Cdf.groupby("group").groups.items():
            ps = Cdf.loc[idx, f"p_{met}"]
            ok = ps.notna()
            if ok.any():
                Cdf.loc[ps[ok].index, f"p_{met}_holm"] = _holm(ps[ok].values)
    Cdf.to_csv(out / "comparisons_v2.csv", index=False)
    show = Cdf[["group", "A", "B", "n_pairs", "d_auc_all", "p_auc_all_holm", "d_f1_05_all", "p_f1_05_all_holm",
                "p_delong_pooled_median", "d_auc_unseen", "p_auc_unseen", "d_f1_05_unseen", "p_f1_05_unseen", "d_ece", "p_ece"]].copy()
    show.columns = ["Group", "A", "B", "pairs", "$\\Delta$AUC", "p (Holm)", "$\\Delta$F1@0.5", "p (Holm)", "DeLong p",
                    "$\\Delta$AUC unseen", "p", "$\\Delta$F1 unseen", "p", "$\\Delta$ECE", "p"]
    _df_to_latex(show.round(4), out / "table_comparisons_v2.tex",
                 "Paired comparisons (A $-$ B) over matching fold $\\times$ seed runs: Wilcoxon signed-rank tests, "
                 "Holm-corrected within each group; DeLong test on pooled out-of-fold predictions (median over seeds). "
                 "F1@0.5 does not depend on the temperature or the validation threshold. "
                 "`Unseen' = only the val/test images of the original split, which the GAN never saw.",
                 "tab:comparisons_v2")
    # ---- dose-response figure (seed 42): 0, 215 random, 500, 1000, 2000, 3000 synthetic images
    pts = [(0, arm("E1_kfold", "AFiSNet", "Raw", s0)), (215, arm(e3, "AFiSNet", "Raw_RandomUnfiltered", s0))] + \
          [(n, arm("E3_dose", "AFiSNet", f"Raw_Random{n:04d}")) for n in A.DOSE_SIZES] + \
          [(3000, arm(e3, "AFiSNet", "Raw_AllUnfiltered", s0))]
    pts = [(n, d) for n, d in pts if len(d) == A.N_FOLDS]
    if len(pts) >= 3:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for ax, met, lab in zip(axes, ("auc_all", "f1_05_all"), ("AUC", "macro F1 (threshold 0.5)")):
            xs = [float(d.n_synthetic.median()) for _, d in pts]; ms = [d[met].mean() for _, d in pts]; ss = [d[met].std(ddof=1) for _, d in pts]
            ax.errorbar(xs, ms, yerr=ss, marker="o", capsize=3, color="#e6550d", label="random NTP images")
            qc = arm("E1_kfold", "AFiSNet", A.SYN_EXP, s0)
            if len(qc) == A.N_FOLDS:
                ax.errorbar([float(qc.n_synthetic.median())], [qc[met].mean()], yerr=[qc[met].std(ddof=1)], marker="s", capsize=3, color="#3182bd",
                            ls="none", label="215 QC-selected (paper)")
            qa = arm(q, "AFiSNet", "Raw_QCfilteredAll")
            if len(qa) == A.N_FOLDS:
                ax.errorbar([qa.n_synthetic.median()], [qa[met].mean()], yerr=[qa[met].std(ddof=1)], marker="D", capsize=3,
                            color="#31a354", ls="none", label="QC top-80% of all")
            ax.set_xscale("symlog", linthresh=200); ax.set_xlabel("number of synthetic training images")
            ax.set_title(f"AFiS-Net: {lab} vs synthetic quantity (seed 42)"); ax.legend(fontsize=8)
        _savefig(fig, out / "fig_dose_response")
    # ---- calibration diagnostics (temperature at its lower bound, threshold spread)
    diag = summ[["family", "model", "experiment", "runs", "temperature_at_floor", "threshold_min", "threshold_max",
                 "ece_mean", "brier_mean"]]
    diag.to_csv(out / "calibration_threshold_diagnostics.csv", index=False)
    print(show.round(4).to_string(index=False))
    _mark_stats_time(out)


# =============================================================================
# 14. SESSION 8: extended screening (more smokers), leakage-free novelty + UMAP,
#     high-resolution occlusion maps, publication texture figure
# =============================================================================
_PUB_RC = {"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
           "legend.fontsize": 6.5, "figure.titlesize": 9, "lines.linewidth": 1.2, "axes.linewidth": 0.8,
           "figure.autolayout": False}


def _pub_rc():
    """Publication font sizes (the official modules change matplotlib defaults when imported)."""
    plt.rcParams.update(_PUB_RC)


def _md5(path):
    import hashlib
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _signature(files):
    import hashlib
    return hashlib.md5("\n".join(sorted(os.path.basename(f) + str(os.path.getsize(f)) for f in files)).encode()).hexdigest()


def _screen_groups(root):
    groups = {}
    for g in sorted(os.listdir(root)):
        dd = os.path.join(root, g)
        if os.path.isdir(dd):
            groups[{"smoker": "smoke", "smokers": "smoke", "suspicious": "smoke"}.get(g.lower(), g.lower())] = _list_images(dd)
    return groups


def _ckpts(family, exp, model="AFiSNet"):
    d = A.RUNS / family / model / exp / f"seed_{A.SEEDS[0]}"
    return {f: d / f"fold_{f}" / "best_model.pt" for f in range(A.N_FOLDS)
            if (d / f"fold_{f}" / "best_model.pt").exists()}


def _oof_reference(exp=None, family="E1_kfold", model="AFiSNet"):
    """Novelty and probability of every real image from the fold model that held it out (FeatureBank/test)."""
    exp = exp or A.SYN_EXP
    base = A.RUNS / family / model / exp / f"seed_{A.SEEDS[0]}"
    frames = []
    for f in range(A.N_FOLDS):
        p = base / f"fold_{f}" / "FeatureBank" / "test" / "filenames.csv"
        if p.exists():
            d = pd.read_csv(p); d["fold"] = f; frames.append(d)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _score_member(model, ckd, x):
    """Cancer probability (temperature-scaled), novelty (smallest diagonal Mahalanobis distance to the
    class centroids of the checkpoint's real-training reference) and the validation threshold."""
    import torch
    names = [str(n) for n in ckd.get("class_names", ["normal", "cancer"])]
    cidx = next((i for i, n in enumerate(names) if n.lower() == "cancer"), len(names) - 1)
    with torch.no_grad():
        feats = model.extract_features(x).float()
        T = float(ckd.get("decision", {}).get("temperature", 1.0)) or 1.0
        pc = float(torch.softmax(model.classifier(feats).float() / max(T, 1e-6), 1)[0, cidx])
    ref = ckd["embedding_reference"]["classes"]
    fe = feats.cpu().numpy()[0]
    dist = []
    for cname in names:
        cen = ref[cname]["centroid"]; var = ref[cname]["variance"]
        cen = cen.cpu().numpy() if hasattr(cen, "cpu") else np.asarray(cen)
        var = var.cpu().numpy() if hasattr(var, "cpu") else np.asarray(var)
        dist.append(float(((fe - cen) ** 2 / (var + 1e-8)).sum()))
    return pc, min(dist), float(ckd["decision"]["cancer_threshold"])


_STD_LUMA = np.array([16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69, 56,
                      14, 17, 22, 29, 51, 87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,
                      49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99], float)


def _jpeg_quality(path):
    """Approximate libjpeg quality of a JPEG file from its luminance quantisation table (order-independent)."""
    try:
        q = Image.open(path).quantization
        lum = np.sort(np.asarray(q[0], float))
    except Exception:
        return None
    best, err = None, 1e18
    for qual in range(1, 101):
        sc = 5000.0 / qual if qual < 50 else 200.0 - 2 * qual
        t = np.sort(np.clip(np.floor((_STD_LUMA * sc + 50) / 100), 1, 255))
        e = np.abs(t - lum).sum()
        if e < err:
            best, err = qual, e
    return best


def _jpeg_roundtrip(pil, quality):
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=int(quality))
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def _jpeg_control(members, test_fold, pool, eval_tf, dev, quality, out):
    """File-format control: every cross-validation image is scored by the fold model that held it out, once as the
    original lossless file and once after JPEG re-encoding at the quality of the smoker files. A large rise of
    novelty after re-encoding would mean the smokers' novelty can be explained by the file format alone."""
    from scipy.stats import wilcoxon
    rows = []
    for b, f in sorted(test_fold.items()):
        if b not in pool or f not in members:
            continue
        pil = Image.open(pool[b][0]).convert("RGB")
        r = dict(image=b, cls=pool[b][1], fold=f)
        for tag, im in (("png", pil), ("jpeg", _jpeg_roundtrip(pil, quality))):
            pc, ood, _ = _score_member(members[f][0], members[f][1], eval_tf(im).unsqueeze(0).to(dev))
            r[f"p_cancer_{tag}"] = pc; r[f"novelty_{tag}"] = ood
        rows.append(r)
    d = pd.DataFrame(rows)
    if d.empty:
        return None
    d.to_csv(out / "jpeg_control_per_image.csv", index=False)
    ratio = d.novelty_jpeg / d.novelty_png
    res = {"jpeg_quality": int(quality), "n": int(len(d)),
           "novelty_median_png": float(d.novelty_png.median()), "novelty_median_jpeg": float(d.novelty_jpeg.median()),
           "novelty_ratio_median": float(ratio.median()), "novelty_ratio_iqr": [float(ratio.quantile(.25)), float(ratio.quantile(.75))],
           "abs_delta_p_cancer_median": float((d.p_cancer_jpeg - d.p_cancer_png).abs().median()),
           "prediction_flips_at_0_5": int(((d.p_cancer_png >= 0.5) != (d.p_cancer_jpeg >= 0.5)).sum())}
    try:
        res["wilcoxon_p_novelty"] = float(wilcoxon(d.novelty_jpeg, d.novelty_png).pvalue)
        res["wilcoxon_p_p_cancer"] = float(wilcoxon(d.p_cancer_jpeg, d.p_cancer_png).pvalue)
    except Exception:
        pass
    (out / "jpeg_control_summary.json").write_text(json.dumps(res, indent=1))
    _pub_rc()
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.8), gridspec_kw={"wspace": 0.4})
    for ax, (col, lab, lg) in zip(axes, (("novelty", "Novelty", True), ("p_cancer", "Cancer probability", False))):
        for c, colr in (("normal", "#4a5568"), ("cancer", "#c53030")):
            m = d.cls == c
            ax.scatter(d[m][f"{col}_png"], d[m][f"{col}_jpeg"], s=6, color=colr, alpha=0.6, lw=0, label=c.capitalize())
        lo = min(d[f"{col}_png"].min(), d[f"{col}_jpeg"].min()); hi = max(d[f"{col}_png"].max(), d[f"{col}_jpeg"].max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.7)
        if lg:
            ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel(f"{lab}, original (lossless)"); ax.set_ylabel(f"{lab}, JPEG q = {int(quality)}")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].legend(frameon=False, fontsize=7)
    _savefig(fig, out / "fig_jpeg_control", dpi=600)
    print("JPEG control:", json.dumps(res, indent=1))
    return res


def screening_v8(paths, force=False, family="E1_kfold", exp=None, tag="screening_v8", model="AFiSNet"):
    """Leakage-free screening with any number of smoker images.
    * every screening image is matched to the cross-validation pool by file name AND by MD5 content hash;
      pool images are scored only by the fold model that held them out, all others by the 5 fold models;
    * cancer probability (temperature-scaled), novelty (nearest-class diagonal Mahalanobis distance to the
      real-training reference stored in each checkpoint) and the embedding of the fold-1 model are recorded;
    * novelty is compared with the out-of-fold novelty of all 608 real images (FeatureBank/test)."""
    import torch
    exp = exp or A.SYN_EXP
    out = A.RESULTS / tag
    if "screening" not in paths:
        print("No screening folder -> skipped."); return
    groups = _screen_groups(paths["screening"])
    sig = _signature([p for v in groups.values() for p in v])
    flag = A.RESULTS / "_done" / tag
    if flag.exists() and not force and sig in flag.read_text():
        print(f"{tag} already done for this exact set of screening images."); return
    ck = _ckpts(family, exp, model)
    if len(ck) < A.N_FOLDS:
        print(f"{family}/{model}/{exp} seed-42 checkpoints missing ({len(ck)} of {A.N_FOLDS}) -> skipped "
              "(attach the previous notebook output; E8 checkpoints are trained in session 8)."); return
    C, K = A.import_pipeline(paths)
    out.mkdir(parents=True, exist_ok=True)
    dev = _dev(); cfg = C.Config(); _, eval_tf = C.build_transforms(224, cfg)
    pool = real_pool(paths)
    pool_hash = {_md5(v[0]): b for b, v in pool.items()}
    test_fold = {}
    for f, pth in ck.items():
        for b in json.loads((pth.parent / "split_manifest.json").read_text())["test"]:
            test_fold[b] = f
    folds = sorted(ck)
    members = {f: load_ckpt(C, ck[f], dev) for f in folds}
    rows, emb_fold0 = [], []
    for g, ps in groups.items():
        for p in ps:
            b = os.path.basename(p)
            match = b if b in pool else pool_hash.get(_md5(p))
            use = [test_fold[match]] if match in test_fold else folds
            x = eval_tf(Image.open(p).convert("RGB")).unsqueeze(0).to(dev)
            pcs, oods, thrs = [], [], []
            for f in use:
                pc, ood, thr_f = _score_member(members[f][0], members[f][1], x)
                pcs.append(pc); oods.append(ood); thrs.append(thr_f)
            if match is None:                                   # never in the CV pool: also keep the fold-1 embedding
                with torch.no_grad():
                    emb_fold0.append((g, b, members[folds[0]][0].extract_features(x).float().cpu().numpy()[0]))
            thr = float(np.median(thrs)); pc = float(np.mean(pcs))
            rows.append(dict(group=g, image=b, file_format=os.path.splitext(b)[1].lower().lstrip("."),
                             region=K.region_group(b), in_cv_pool=match is not None,
                             matched_by=("name" if b in pool else "content hash") if match else "",
                             pool_split=pool[match][2] if match else "",
                             scored_by=f"fold {use[0]} (held-out)" if len(use) == 1 else f"{len(use)}-fold mean",
                             p_cancer=pc, p_cancer_sd=float(np.std(pcs)), novelty=float(np.mean(oods)),
                             threshold=thr, positive=bool(pc >= thr), positive_at_0_5=bool(pc >= 0.5)))
    # ---- file-format control (smoker images were JPEG, everything else lossless)
    qs = [q for q in (_jpeg_quality(p) for p in groups.get("smoke", []) if p.lower().endswith((".jpg", ".jpeg"))) if q]
    q_ctl = int(np.median(qs)) if qs else 90
    print(f"JPEG control at quality {q_ctl} ({'estimated from ' + str(len(qs)) + ' smoker JPEG files' if qs else 'default'})")
    try:
        _jpeg_control(members, test_fold, pool, eval_tf, dev, q_ctl, out)
    except Exception:
        import traceback; traceback.print_exc()
    del members
    df = pd.DataFrame(rows); df.to_csv(out / "per_image_screening_v8.csv", index=False)
    order = [g for g in ("normal", "smoke", "cancer") if g in groups] + [g for g in groups if g not in ("normal", "smoke", "cancer")]
    _screening_stats(df, order, out)                           # probabilities: JT / KW / MW (+ region level)
    # ---- novelty statistics
    from scipy.stats import mannwhitneyu, kruskal
    ref = _oof_reference(exp, family, model)
    nov = {}
    if not ref.empty:
        r = ref.mahalanobis_ood_score.values
        q95 = float(np.quantile(r, 0.95))
        nov["reference"] = {"n": int(len(r)), "median": float(np.median(r)), "q95": q95}
        for g in order:
            v = df[df.group == g].novelty.values
            if len(v):
                nov[g] = {"n": int(len(v)), "median": float(np.median(v)), "iqr": [float(np.quantile(v, .25)), float(np.quantile(v, .75))],
                          "frac_above_ref_q95": float((v > q95).mean()),
                          "p_greater_than_reference": float(mannwhitneyu(v, r, alternative="greater").pvalue)}
        vs = [df[df.group == g].novelty.values for g in order if (df.group == g).any()]
        if len(vs) >= 2:
            nov["kruskal_p"] = float(kruskal(*vs).pvalue)
        if "smoke" in order:
            for g in ("normal", "cancer"):
                if g in order:
                    nov[f"smoke>{g}_p"] = float(mannwhitneyu(df[df.group == "smoke"].novelty, df[df.group == g].novelty, alternative="greater").pvalue)
    (out / "novelty_stats.json").write_text(json.dumps(nov, indent=1))
    print(json.dumps(nov, indent=1))
    # ---- UMAP of the fold-1 model: real training (faint), real held-out test, never-seen screening images
    fb = A.RUNS / family / model / exp / f"seed_{A.SEEDS[0]}" / f"fold_{folds[0]}" / "FeatureBank"
    um = None
    if (fb / "train_retrieval" / "embeddings.npy").exists():
        etr = np.load(fb / "train_retrieval" / "embeddings.npy"); ltr = np.load(fb / "train_retrieval" / "labels.npy")
        ete = np.load(fb / "test" / "embeddings.npy"); lte = np.load(fb / "test" / "labels.npy")
        es = np.stack([e for _, _, e in emb_fold0]) if emb_fold0 else np.zeros((0, etr.shape[1]))
        gs = [g for g, _, _ in emb_fold0]
        try:
            import umap
            red = umap.UMAP(n_neighbors=15, min_dist=0.3, metric="cosine", random_state=42).fit(etr)
            ztr, zte = red.embedding_, red.transform(ete); zs = red.transform(es) if len(es) else np.zeros((0, 2)); how = "UMAP"
        except Exception:
            from sklearn.decomposition import PCA
            red = PCA(2, random_state=0).fit(etr); ztr, zte = red.transform(etr), red.transform(ete)
            zs = red.transform(es) if len(es) else np.zeros((0, 2)); how = "PCA"
        um = dict(ztr=ztr, ltr=ltr, zte=zte, lte=lte, zs=zs, gs=np.array(gs), how=how)
        np.savez(out / "embedding_projection.npz", **um)
    # ---- figure (publication style)
    _fig_screening_v8(df, ref, um, order, out)
    flag.parent.mkdir(parents=True, exist_ok=True); flag.write_text(sig + "\n" + time.strftime("%Y-%m-%d %H:%M:%S"))


def _fig_screening_v8(df, ref, um, order, out):
    _pub_rc()
    cols = {"normal": "#4a5568", "smoke": "#dd6b20", "cancer": "#c53030"}
    names = {"normal": "Normal", "smoke": "Smokers", "cancer": "Cancer"}
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.8), gridspec_kw={"width_ratios": [1, 1.15, 1.05], "wspace": 0.45})
    rng = np.random.default_rng(1)
    for k, (col, ttl, tf) in enumerate((("p_cancer", "A  Cancer probability", lambda v: v),
                                        ("novelty", "B  Out-of-distribution novelty", np.log10))):
        ax = axes[k]
        grp = ([("ref", f"CV test\n({len(ref)})", ref.mahalanobis_ood_score.values, "#a0aec0")] if (k == 1 and not ref.empty) else []) + \
              [(g, f"{names.get(g, g)}\n({(df.group == g).sum()})", df[df.group == g][col].values, cols.get(g, "k")) for g in order]
        for i, (g, lab, v, c) in enumerate(grp):
            v = tf(np.asarray(v, float))
            bp = ax.boxplot([v], positions=[i], widths=0.5, showfliers=False, patch_artist=True)
            for bb in bp["boxes"]:
                bb.set(facecolor=c, alpha=0.2, edgecolor=c)
            for kk in ("medians", "whiskers", "caps"):
                for e in bp[kk]:
                    e.set(color=c)
            if g != "ref":
                ax.scatter(i + rng.uniform(-0.14, 0.14, len(v)), v, s=9, color=c, alpha=0.85, lw=0, zorder=3)
        ax.set_xticks(range(len(grp))); ax.set_xticklabels([x[1] for x in grp], fontsize=6.5)
        ax.set_title(ttl, loc="left", fontsize=9)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].axhline(float(df.threshold.median()), ls="--", c="k", lw=0.7); axes[0].set_ylabel("Predicted cancer probability")
    if not ref.empty:
        axes[1].axhline(np.log10(np.quantile(ref.mahalanobis_ood_score.values, 0.95)), ls=":", c="k", lw=0.7)
    axes[1].set_ylabel("Novelty, log$_{10}$ Mahalanobis distance")
    ax = axes[2]
    if um is not None:
        for c, colr, lab in ((0, cols["normal"], "Normal"), (1, cols["cancer"], "Cancer")):
            ax.scatter(um["ztr"][um["ltr"] == c, 0], um["ztr"][um["ltr"] == c, 1], s=4, color=colr, alpha=0.12, lw=0,
                       label=f"{lab}, training")
            ax.scatter(um["zte"][um["lte"] == c, 0], um["zte"][um["lte"] == c, 1], s=10, color=colr, alpha=0.85, lw=0.3, edgecolor="white", label=f"{lab}, held-out")
        for g in sorted(set(um["gs"].tolist())):
            m = um["gs"] == g
            ax.scatter(um["zs"][m, 0], um["zs"][m, 1], s=22, marker="D", color=cols.get(g, "k"), edgecolor="black", lw=0.4,
                       label=f"{names.get(g, g)} (never seen)")
        ax.legend(frameon=False, fontsize=5.8, loc="best", markerscale=1.0, handletextpad=0.2)
        ax.set_xlabel(f"{um['how']} 1"); ax.set_ylabel(f"{um['how']} 2")
    ax.set_xticks([]); ax.set_yticks([]); ax.set_title("C  Learned embedding (fold 1)", loc="left", fontsize=9)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    _savefig(fig, out / "fig_screening", dpi=600)


def explain_v3(paths, force=False, patch=16, stride=4, src_tag="explainability_v2", ck=None,
               tag="explainability_v3", flag="explain_v3"):
    """High-resolution occlusion maps (16 px patches, stride 4) for the gallery candidates of explainability v2,
    saved as raw arrays plus a publication figure. Picks can be changed with the AFIS_EXPLAIN_PICKS setting,
    e.g. 'cancer:1,6,14;normal:5,6,13' (ranks within the correct-cancer / correct-normal galleries)."""
    import torch
    out = A.RESULTS / tag
    if _done(flag) and not force:
        print(f"{tag} already done."); return
    per = A.RESULTS / src_tag / "per_image_explainability_v2.csv"
    ck = ck or _seed_ckpts(A.SYN_EXP)
    if not per.exists() or len(ck) < A.N_FOLDS:
        print("Explainability v2 results or checkpoints missing -> skipped."); return
    if A.SMOKE:
        patch, stride = 56, 28
    C, K = A.import_pipeline(paths)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(per); df = df[(df.arm == "Raw+Synthetic") & df.occl_valid.astype(bool)]
    cats = {"correct_cancer": df[(df.true == "cancer") & df.correct].sort_values("p_cancer", ascending=False),
            "correct_normal": df[(df.true == "normal") & df.correct].sort_values("p_cancer"),
            "misclassified_cancer": df[(df.true == "cancer") & ~df.correct].sort_values("p_cancer"),
            "misclassified_normal": df[(df.true == "normal") & ~df.correct].sort_values("p_cancer", ascending=False)}
    n = 3 if A.SMOKE else N_GALLERY
    pool = real_pool(paths); dev = _dev(); cfg = C.Config(); _, eval_tf = C.build_transforms(224, cfg); crop = C.ForegroundCrop()
    maps = {}
    for f, pth in sorted(ck.items()):
        todo = [(cat, k, r) for cat, sub in cats.items() for k, (_, r) in enumerate(sub.head(n).iterrows(), 1) if int(r.fold) == f]
        if not todo:
            continue
        model, ckd = load_ckpt(C, pth, dev)
        for cat, k, r in todo:
            pil = Image.open(pool[r.image][0]).convert("RGB")
            x = eval_tf(pil).unsqueeze(0).to(dev)
            cls = 1 if r.pred == "cancer" else 0
            occ, _ = _occlusion_map(model, x, cls, patch, stride, chunk=128)
            base = np.asarray(crop(pil).resize((224, 224), Image.BICUBIC), dtype=np.float32) / 255.0
            maps[(cat, k)] = (r.image, base.astype(np.float16), occ.astype(np.float32), float(r.p_cancer))
        del model
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    np.savez_compressed(out / "occlusion_maps_highres.npz",
                        **{f"{c}_{k:02d}": v[2] for (c, k), v in maps.items()},
                        **{f"{c}_{k:02d}_input": v[1] for (c, k), v in maps.items()})
    pd.DataFrame([dict(category=c, rank=k, image=v[0], p_cancer=v[3]) for (c, k), v in sorted(maps.items())]).to_csv(out / "index.csv", index=False)
    _fig_explain_v3(maps, out)
    _mark(flag)


def _fig_explain_v3(maps, out):
    from matplotlib import colormaps
    from scipy.ndimage import gaussian_filter, binary_fill_holes, binary_opening, label as cc_label
    from skimage import filters, measure

    def stretch(b):
        lo, hi = np.percentile(b, 1), np.percentile(b, 99.7)
        return np.clip((b - lo) / (hi - lo + 1e-6), 0, 1)

    def render(base, occ):
        b = stretch(base.astype(np.float32)); s = np.clip(occ, 0, None); s = gaussian_filter(s, 2)
        s = (s - s.min()) / (s.max() - s.min() + 1e-8)
        heat = colormaps["inferno"](s)[..., :3]; a = np.clip((s - 0.25) / 0.75, 0, 1)[..., None] * 0.8
        g = gaussian_filter(b.max(2), 2); m = binary_fill_holes(binary_opening(g > filters.threshold_otsu(g), iterations=2))
        lab, nl = cc_label(m)
        if nl > 1:
            m = lab == (1 + np.argmax(np.bincount(lab.ravel())[1:]))
        return b, np.clip(b * (1 - a) + heat * a, 0, 1), m

    # contact sheets of every candidate (to choose cells)
    for cat in sorted({c for c, _ in maps}):
        ks = sorted(k for c, k in maps if c == cat)
        fig, axes = plt.subplots(2, len(ks), figsize=(1.6 * len(ks), 3.4), squeeze=False)
        for j, k in enumerate(ks):
            img, base, occ, pc = maps[(cat, k)]
            b, ov, m = render(base, occ)
            axes[0, j].imshow(b); axes[1, j].imshow(ov); axes[0, j].set_title(f"#{k}  P={pc:.2f}", fontsize=7)
            for cc in measure.find_contours(m.astype(float), 0.5):
                if len(cc) > 40:
                    axes[1, j].plot(cc[:, 1], cc[:, 0], c="#7fdbff", lw=0.6)
        for ax in axes.ravel():
            ax.axis("off")
        _savefig(fig, out / f"candidates_{cat}", dpi=SHEET_DPI)
    # publication figure
    _pub_rc()
    spec = os.environ.get("AFIS_EXPLAIN_PICKS", "cancer:1,6,14;normal:5,6,13")
    picks = []
    for part in spec.split(";"):
        cl, ids = part.split(":")
        picks += [(f"correct_{cl.strip()}", int(i)) for i in ids.split(",")]
    picks = [p for p in picks if p in maps][:6]
    if not picks:
        return
    rows = sorted({c for c, _ in picks}); ncol = max(sum(1 for c, _ in picks if c == r) for r in rows)
    fig, axes = plt.subplots(len(rows), 2 * ncol, figsize=(1.2 * 2 * ncol, 1.38 * len(rows)), squeeze=False,
                             gridspec_kw={"wspace": 0.04, "hspace": 0.12})
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([]); [sp.set_visible(False) for sp in ax.spines.values()]
    for ri, rr in enumerate(rows):
        for j, (c, k) in enumerate([p for p in picks if p[0] == rr]):
            img, base, occ, pc = maps[(c, k)]
            b, ov, m = render(base, occ)
            axes[ri, 2 * j].imshow(b); axes[ri, 2 * j + 1].imshow(ov)
            for cc in measure.find_contours(m.astype(float), 0.5):
                if len(cc) > 40:
                    axes[ri, 2 * j + 1].plot(cc[:, 1], cc[:, 0], c="#7fdbff", lw=0.6)
            if ri == 0:
                axes[ri, 2 * j].set_title("Input", fontsize=7.5); axes[ri, 2 * j + 1].set_title("Occlusion map", fontsize=7.5)
        axes[ri, 0].set_ylabel(rr.replace("correct_", "").capitalize(), fontsize=8.5)
    sm = plt.cm.ScalarMappable(cmap="inferno"); cb = fig.colorbar(sm, ax=axes, fraction=0.015, pad=0.01)
    cb.set_ticks([0, 1]); cb.set_ticklabels(["low", "high"]); cb.ax.tick_params(labelsize=6.5)
    _savefig(fig, out / "fig_explainability", dpi=600)


def texture_v2(paths, force=False, n_max=1000):
    """Texture fidelity re-run with the arrays saved and a publication-style figure."""
    out = A.RESULTS / "texture_v2"
    if _done("texture_v2") and not force:
        print("Texture v2 already done."); return
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(2026)
    sets = {"Real (training)": {c: _list_images(os.path.join(paths["train"], c)) for c in CLASSES},
            "QC-selected": class_sets(paths, "synthetic_selected"),
            "All generated": class_sets(paths, "synthetic_all_ntp"),
            "Baseline (no NTP)": class_sets(paths, "synthetic_all_baseline")}
    if A.qc_clean_ready():
        sets["QC-clean (hygiene)"] = {c: _list_images(str(A.qc_clean_dir() / c)) for c in CLASSES}
    sets = {k: v for k, v in sets.items() if all(len(v[c]) > 0 for c in CLASSES)}
    arrays, rows = {}, []
    for c in CLASSES:
        for name, s in sets.items():
            ps = s[c]
            if len(ps) > n_max:
                ps = [ps[i] for i in sorted(rng.choice(len(ps), n_max, replace=False))]
            P, L = [], []
            for p in ps:
                g = _gray(p); f, pr = _radial_psd(g); P.append(pr); L.append(_lbp_hist(g))
            arrays[(c, name)] = (f, np.array(P), np.array(L))
        f, Pr, Lr = arrays[(c, "Real (training)")]
        for name in sets:
            if name == "Real (training)":
                continue
            _, P, L = arrays[(c, name)]
            hm, hr = L.mean(0), Lr.mean(0)
            rows.append(dict(cls=c, set=name, n=len(P), n_real=len(Pr),
                             spectral_MAE_log10=float(np.abs(np.log10(P.mean(0) + 1e-12) - np.log10(Pr.mean(0) + 1e-12)).mean()),
                             LBP_correlation=float(np.corrcoef(hm, hr)[0, 1]), LBP_intersection=float(np.minimum(hm, hr).sum())))
    pd.DataFrame(rows).to_csv(out / "texture_metrics.csv", index=False)
    np.savez_compressed(out / "texture_arrays.npz", **{f"{c}|{n}|{k}": v for (c, n), arr in arrays.items()
                                                       for k, v in zip(("freq", "psd", "lbp"), arr)})
    colours = {"Real (training)": "#1a202c", "QC-selected": "#2b6cb0", "All generated": "#dd6b20", "Baseline (no NTP)": "#6b46c1",
               "QC-clean (hygiene)": "#2f855a"}
    _pub_rc()
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0), gridspec_kw={"hspace": 0.45, "wspace": 0.28})
    for ci, c in enumerate(CLASSES):
        ax = axes[0, ci]
        for name in sets:
            f, P, _ = arrays[(c, name)]
            lp = np.log10(P + 1e-12); m, sd = lp.mean(0), lp.std(0)
            ax.plot(f, m, color=colours[name], lw=1.5 if name.startswith("Real") else 1.1,
                    ls="-" if name.startswith("Real") else "--", label=f"{name} (n = {len(P)})")
            if name.startswith("Real") or name == "QC-selected":
                ax.fill_between(f, m - sd, m + sd, color=colours[name], alpha=0.12, lw=0)
        ax.set_xscale("log"); ax.set_xlabel("Spatial frequency (cycles px$^{-1}$)"); ax.set_ylabel("log$_{10}$ normalised power")
        ax.set_title(f"{'AB'[ci]}  {c.capitalize()}: radially averaged power spectrum", loc="left", fontsize=8.5)
        ax.legend(frameon=False, fontsize=6.5)
        ax = axes[1, ci]; names = list(sets); w = 0.8 / len(names)
        for k, name in enumerate(names):
            L = arrays[(c, name)][2]
            ax.bar(np.arange(10) + (k - (len(names) - 1) / 2) * w, L.mean(0), w, yerr=L.std(0), color=colours[name],
                   label=name, error_kw=dict(lw=0.5, capsize=1))
        ax.set_xticks(range(10)); ax.set_xlabel("Uniform LBP code (P = 8, R = 1)"); ax.set_ylabel("Fraction of pixels")
        ax.set_title(f"{'CD'[ci]}  {c.capitalize()}: local binary patterns", loc="left", fontsize=8.5)
        ax.legend(frameon=False, fontsize=6.5)
        for a in axes[:, ci]:
            for sp in ("top", "right"):
                a.spines[sp].set_visible(False)
    _savefig(fig, out / "fig_texture_fidelity", dpi=600)
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    _mark("texture_v2")


def gan_metrics_baseline_refresh(paths):
    """If baseline (no-NTP) generator images are now available, recompute the matched-n GAN metrics with them."""
    has_base = all(len(v) >= 10 for v in class_sets(paths, "synthetic_all_baseline").values())
    f = A.RESULTS / "gan_metrics_v2" / "gan_metrics_matched_n.csv"
    if has_base and (not f.exists() or "Baseline" not in f.read_text()):
        print("Baseline generator images found -> recomputing matched-n GAN metrics with them.")
        gan_metrics_v2(paths, force=True)
    else:
        print("No new baseline generator images -> GAN metrics unchanged.")


# =============================================================================
# 16. SESSION 9/10: QC as a hygiene filter, final-model choice, main results with the final model
# =============================================================================
def _fg_stats(p):
    from skimage.filters import threshold_otsu
    a = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32).max(2)
    t = threshold_otsu(a) if a.max() > a.min() else a.max() + 1
    return float((a > t).mean()), float(a.max())


def _topk_mean(S, k=10):
    k = max(1, min(k, S.shape[1]))
    return np.sort(S, axis=1)[:, -k:].mean(1)


def qc_clean_ntp(paths, force=False):
    """QC as a hygiene filter instead of a ranking. Every generated image is KEPT unless it fails a check that
    almost no real training image of its class would fail AND it lies in the extreme tail of the generated pool
    (real-image references are leave-one-region-out, so frames of the same imaging region do not inflate them):
      1. technical defect   - foreground fraction below min(real 0.5th pct, generated 2.5th pct), above
                              max(real 99.5th pct, generated 97.5th pct), or peak intensity below the lower bound;
      2. near-copy          - DINOv2 cosine >= 0.995 to ANY of the 608 real images (memorisation / leakage safeguard);
      3. unrealistic        - similarity to the nearest real training cell below min(real 1st pct, generated 5th pct);
      4. class-ambiguous    - class margin (mean top-10 similarity to own class minus other class) below
                              min(real 5th pct, generated 5th pct);
      5. duplicate          - DINOv2 cosine >= 0.98 to an already kept generated image of the same class.
    No ranking, no top-k: the aim is to remove bad, copied and duplicated images while keeping the diversity that
    drives the classification gain."""
    root = A.RESULTS / "qc_clean"
    if A.qc_clean_ready() and not force:
        print("QC-clean selection already available:", A.qc_clean_dir()); return True
    if "synthetic_all_ntp" not in paths:
        print("No synthetic_all_ntp -> QC-clean skipped."); return False
    ef = embeddings(paths)
    E = np.load(ef, allow_pickle=True)
    if "ntp_emb" not in E.files:
        print("Embeddings of the NTP pool missing -> recomputing."); E = np.load(embeddings(paths, force=True), allow_pickle=True)
    norm = lambda X: X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    pool = real_pool(paths)
    C, K = A.import_pipeline(paths)
    rn = list(E["real_names"]); R = norm(E["real_emb"].astype(np.float32)); rl = np.array(E["real_labels"])
    rp = [pool[b][0] if b in pool else str(x) for b, x in zip(rn, E["real_paths"])]   # current file locations
    is_train = np.array([pool.get(b, (None, None, ""))[2] == "train" for b in rn])
    gn = list(E["ntp_names"]); G = norm(E["ntp_emb"].astype(np.float32)); gl = np.array(E["ntp_labels"])
    cur = {os.path.basename(x): x for c in CLASSES for x in class_sets(paths, "synthetic_all_ntp")[c]}
    gp = [cur.get(n, str(x)) for n, x in zip(gn, E["ntp_paths"])]
    if root.exists():
        shutil.rmtree(root)
    (root / "selected").mkdir(parents=True, exist_ok=True)
    rows, thr = [], {}
    for c in CLASSES:
        o = [x for x in CLASSES if x != c][0]
        Rc = R[is_train & (rl == c)]; Ro = R[is_train & (rl == o)]
        rpc = [p for p, t, l in zip(rp, is_train, rl) if t and l == c]
        rnc = [b for b, t, l in zip(rn, is_train, rl) if t and l == c]
        reg = np.array([K.region_group(b) for b in rnc])
        # ---- real-image reference (leave-one-REGION-out)
        S = Rc @ Rc.T
        S[reg[:, None] == reg[None, :]] = -np.inf
        real_margin = _topk_mean(S) - _topk_mean(Rc @ Ro.T)
        real_nn = S.max(1)
        ok = np.isfinite(real_nn)
        real_margin, real_nn = real_margin[ok], real_nn[ok]
        fg = np.array([_fg_stats(p) for p in rpc])
        # ---- generated images of class c
        idx = [i for i in range(len(gn)) if gl[i] == c]
        Gc = G[idx]
        own = Gc @ Rc.T
        margin = _topk_mean(own) - _topk_mean(Gc @ Ro.T)
        nn_train = own.max(1)
        nn_any = (Gc @ R.T).max(1)
        gfg = np.array([_fg_stats(gp[i]) for i in idx])
        t = dict(margin=float(min(np.quantile(real_margin, 0.05), np.quantile(margin, 0.05))),
                 realism=float(min(np.quantile(real_nn, 0.01), np.quantile(nn_train, 0.05))),
                 fg_lo=float(min(np.quantile(fg[:, 0], 0.005), np.quantile(gfg[:, 0], 0.025))),
                 fg_hi=float(max(np.quantile(fg[:, 0], 0.995), np.quantile(gfg[:, 0], 0.975))),
                 peak_lo=float(min(np.quantile(fg[:, 1], 0.005), np.quantile(gfg[:, 1], 0.025))),
                 near_copy=1.01 if A.SMOKE else 0.995, duplicate=1.01 if A.SMOKE else 0.98, n_real_train=int(len(Rc)),
                 real_margin_q05=float(np.quantile(real_margin, 0.05)), real_nn_q01=float(np.quantile(real_nn, 0.01)))
        thr[c] = t
        kept_emb = np.zeros((len(idx), Gc.shape[1]), np.float32); n_kept = 0
        order = np.argsort([gn[i] for i in idx])                 # neutral order (file name) for the duplicate check
        for j in order:
            i = idx[j]
            f, pk = gfg[j]
            reason = ""
            if not (t["fg_lo"] <= f <= t["fg_hi"]) or pk < t["peak_lo"]:
                reason = "technical defect"
            elif nn_any[j] >= t["near_copy"]:
                reason = "near-copy of a real image"
            elif nn_train[j] < t["realism"]:
                reason = "unrealistic (unlike any real cell)"
            elif margin[j] < t["margin"]:
                reason = "class-ambiguous"
            elif n_kept and float(np.max(kept_emb[:n_kept] @ Gc[j])) >= t["duplicate"]:
                reason = "duplicate of a kept image"
            if not reason:
                kept_emb[n_kept] = Gc[j]; n_kept += 1
                (root / "selected" / c).mkdir(parents=True, exist_ok=True)
                shutil.copy2(gp[i], root / "selected" / c / gn[i])
            rows.append(dict(image=gn[i], cls=c, kept=not reason, reason=reason, fg_fraction=f, peak=pk,
                             class_margin=float(margin[j]), nn_sim_train=float(nn_train[j]), nn_sim_any_real=float(nn_any[j])))
        pd.DataFrame(dict(cls=c, real_margin=real_margin, real_nn=real_nn)).to_csv(root / f"real_reference_{c}.csv", index=False)
    df = pd.DataFrame(rows); df.to_csv(root / "per_image.csv", index=False)
    counts = df.groupby(["cls", "reason"]).size().unstack(fill_value=0)
    summ = {"thresholds": thr, "n_generated": df.groupby("cls").size().to_dict(),
            "n_kept": df[df.kept].groupby("cls").size().to_dict(),
            "rejected_by_reason": {c: {k: int(v) for k, v in counts.loc[c].items() if k} for c in counts.index}}
    print(json.dumps(summ, indent=1))
    # ---- figures: margins and realism (real vs generated, with thresholds) + galleries of rejected images
    _pub_rc()
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.6), gridspec_kw={"hspace": 0.55, "wspace": 0.3})
    for ci, c in enumerate(CLASSES):
        ref = pd.read_csv(root / f"real_reference_{c}.csv"); d = df[df.cls == c]
        for ri, (col, rcol, key, lab) in enumerate((("class_margin", "real_margin", "margin", "Class margin (DINOv2)"),
                                                    ("nn_sim_train", "real_nn", "realism", "Similarity to nearest real cell"))):
            ax = axes[ri, ci]
            bins = np.linspace(min(d[col].min(), ref[rcol].min()), max(d[col].max(), ref[rcol].max()), 40)
            ax.hist(ref[rcol], bins=bins, density=True, alpha=0.5, color="#1a202c", label=f"Real (n = {len(ref)})")
            ax.hist(d[col], bins=bins, density=True, alpha=0.5, color="#dd6b20", label=f"Generated (n = {len(d)})")
            ax.axvline(thr[c][key], color="k", ls="--", lw=0.8)
            ax.set_xlabel(lab); ax.set_ylabel("Density")
            ax.set_title(f"{'AB'[ci] if ri == 0 else 'CD'[ci]}  {c.capitalize()}", loc="left")
            ax.legend(frameon=False)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
    _savefig(fig, root / "fig_qc_clean_thresholds", dpi=600)
    for reason in sorted(df[~df.kept].reason.unique()):
        d = df[df.reason == reason].head(15)
        if d.empty:
            continue
        n = len(d); fig, axes = plt.subplots(1, n, figsize=(1.3 * n, 1.6), squeeze=False)
        for ax, (_, r) in zip(axes[0], d.iterrows()):
            p = [x for x in gp if os.path.basename(x) == r.image][0]
            ax.imshow(Image.open(p).convert("RGB")); ax.axis("off"); ax.set_title(f"{r.cls[:1].upper()} {r.class_margin:+.3f}", fontsize=6)
        _savefig(fig, root / "rejected" / reason.split(" ")[0], dpi=SHEET_DPI)
    (root / "summary.json").write_text(json.dumps(summ, indent=1))
    return True


def gan_texture_refresh_qcclean(paths):
    """Re-run the matched-n GAN metrics and the texture figure once the QC-clean set exists."""
    if not A.qc_clean_ready():
        return
    f = A.RESULTS / "gan_metrics_v2" / "gan_metrics_matched_n.csv"
    if not f.exists() or "QC-clean" not in f.read_text():
        gan_metrics_v2(paths, force=True)
    t = A.RESULTS / "texture_v2" / "texture_metrics.csv"
    if not t.exists() or "QC-clean" not in t.read_text():
        texture_v2(paths, force=True)


def _holm(ps):
    ps = np.asarray(ps, float); o = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0.0
    for k, i in enumerate(o):
        run = max(run, min(1.0, (m - k) * ps[i])); adj[i] = run
    return adj


def stats_final(paths):
    """Tables and paired tests for the final model (records = metrics of run_one; F1@0.5 / unseen from predictions)."""
    from scipy.stats import wilcoxon, t as tdist
    model = A.choose_final_model()
    if model is None:
        print("Final model not chosen yet (stage A incomplete)."); return
    out = A.RESULTS / "stats_final"; out.mkdir(parents=True, exist_ok=True)
    sel = json.loads(A.final_model_file().read_text())
    pd.DataFrame(sel["candidates"]).to_csv(out / "final_model_selection.csv", index=False)
    rec = load_records()
    pool = real_pool(paths)
    orig = {b: v[2] for b, v in pool.items()}
    rows, preds = [], {}
    for _, r in rec.iterrows():
        m, p = _run_metrics(r, orig)
        k = (r.family, r.model, r.experiment, int(r.seed), int(r.fold)); preds[k] = p
        rows.append(dict(family=r.family, model=r.model, experiment=r.experiment, seed=int(r.seed), fold=int(r.fold),
                         auc=r.roc_auc_macro, f1=r.f1_macro, mcc=r.mcc, sen=r.Cancer_recall_sensitivity, ece=r.ece,
                         brier=r.brier_score, n_synthetic=r.get("n_synthetic", np.nan), **m))
    M = pd.DataFrame(rows)

    def arm(exp, mdl=None, seeds=None):
        fam, mdl = A.final_arm(mdl or model, exp)
        d = M[(M.family == fam) & (M.model == mdl) & (M.experiment == exp)]
        return d[d.seed.isin(seeds)] if seeds else d

    def base(mdl, exp):
        d = M[(M.family == "E1_kfold") & (M.model == mdl) & (M.experiment == exp)]
        return d

    arms = [("Raw", "real only"), (A.SYN_EXP, "+215 QC-selected"), ("Raw_RandomUnfiltered", "+215 random"),
            ("Raw_AllUnfiltered", "+3,000 all generated"), ("Raw_QCclean", "+QC-clean (hygiene filter)"),
            ("Raw_RandomMatchQCclean", "+random, same count as QC-clean (seed 42)")]
    S = []
    for exp, lab in arms:
        d = arm(exp)
        if d.empty:
            continue
        S.append(dict(training=lab, experiment=exp, runs=len(d), n_synthetic=d.n_synthetic.median(),
                      **{f"{k}_mean": d[k].mean() for k in ("auc", "f1", "mcc", "sen", "ece", "brier", "f1_05_all", "auc_unseen")},
                      **{f"{k}_sd": d[k].std() for k in ("auc", "f1", "mcc", "sen")}))
    pd.DataFrame(S).to_csv(out / f"summary_final_{model}.csv", index=False)
    comps = []

    def cmp(group, la, a, lb, b):
        if a.empty or b.empty:
            return
        j = a.merge(b, on=["seed", "fold"], suffixes=("_a", "_b"))
        if len(j) < 2:
            return
        row = dict(group=group, A=la, B=lb, n_pairs=len(j))
        for met in ("auc", "f1", "f1_05_all", "ece", "auc_unseen"):
            d = (j[f"{met}_a"] - j[f"{met}_b"]).dropna()
            row[f"d_{met}"] = d.mean()
            se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else np.nan
            q = tdist.ppf(0.975, len(d) - 1) if len(d) > 1 else np.nan
            row[f"ci95_{met}"] = f"[{d.mean() - q * se:+.4f}, {d.mean() + q * se:+.4f}]"
            row[f"p_{met}"] = float(wilcoxon(d).pvalue) if len(d) > 1 and (d != 0).any() else np.nan
        dl = []
        for s in sorted(j.seed.unique()):
            ka, kb = a.iloc[0], b.iloc[0]
            pa = [preds[k] for k in preds if k[:4] == (ka.family, ka.model, ka.experiment, int(s))]
            pb = [preds[k] for k in preds if k[:4] == (kb.family, kb.model, kb.experiment, int(s))]
            if len(pa) == A.N_FOLDS and len(pb) == A.N_FOLDS:
                mm = pd.concat(pa)[["Filename", "y", "Probability_Cancer"]].merge(pd.concat(pb)[["Filename", "Probability_Cancer"]], on="Filename")
                dl.append(_delong(mm.y.values, mm.Probability_Cancer_x.values, mm.Probability_Cancer_y.values)[2])
        row["p_delong_median"] = float(np.median(dl)) if dl else np.nan
        comps.append(row)
    s0 = [A.SEEDS[0]]
    for exp, lab in arms[1:5]:
        cmp("final: synthetic effect", lab, arm(exp), "real only", arm("Raw"))
    cmp("final: QC-clean vs all", "+QC-clean", arm("Raw_QCclean"), "+3,000 all", arm("Raw_AllUnfiltered"))
    cmp("final: QC-clean vs random (seed 42)", "+QC-clean", arm("Raw_QCclean", seeds=s0), "+random same count", arm("Raw_RandomMatchQCclean"))
    cmp("final: QC-215 vs random-215", "+215 QC-selected", arm(A.SYN_EXP), "+215 random", arm("Raw_RandomUnfiltered"))
    for b in A.KFOLD_MODELS[1:]:
        cmp("final vs baselines (real only)", f"final {model}", arm("Raw"), b, base(b, "Raw"))
        cmp("final vs baselines (+215 QC)", f"final {model}", arm(A.SYN_EXP), b, base(b, A.SYN_EXP))
    if model != "AFiSNet":
        for exp, lab in arms[:4]:
            cmp("final vs original AFiS-Net", f"{model} {lab}", arm(exp), f"AFiSNet {lab}", arm(exp, "AFiSNet"))
    for frac in A.SCALING_FRACTIONS:
        tg = f"frac{int(round(frac * 100)):03d}"
        cmp("final: data scaling", f"+215 QC @{tg}", arm(f"{A.SYN_EXP}_{tg}"), f"real only @{tg}", arm(f"Raw_{tg}"))
    C_ = pd.DataFrame(comps)
    if not C_.empty:
        C_["p_auc_holm"] = np.nan
        for g, idx in C_.groupby("group").groups.items():
            v = C_.loc[idx, "p_auc"]
            ok = v.notna()
            C_.loc[v[ok].index, "p_auc_holm"] = _holm(v[ok].values)
    C_.to_csv(out / f"comparisons_final_{model}.csv", index=False)
    print(f"FINAL MODEL: {model}"); print(pd.DataFrame(S).round(3).to_string(index=False))
    if not C_.empty:
        print(C_[["group", "A", "B", "n_pairs", "d_auc", "ci95_auc", "p_auc", "p_auc_holm", "p_delong_median"]].round(4).to_string(index=False))


def analyses_final(paths):
    """Screening and explainability with the final model trained on the QC-clean set (seed-42 checkpoints)."""
    model = A.choose_final_model()
    if model is None:
        print("Final model not chosen yet."); return
    fam = "E9_final"
    ck_clean = _ckpts(fam, "Raw_QCclean", model)
    raw_fam, _ = A.final_arm(model, "Raw")
    ck_raw = _ckpts(raw_fam, "Raw", model) if model != "AFiSNet" else _seed_ckpts("Raw")
    if len(_ckpts(fam, "Raw_AllUnfiltered", model)) == A.N_FOLDS:      # robustness: same model, all 3,000 images
        screening_v8(paths, family=fam, exp="Raw_AllUnfiltered", model=model, tag="screening_final_all3000")
    if len(ck_clean) == A.N_FOLDS:
        screening_v8(paths, family=fam, exp="Raw_QCclean", model=model, tag="screening_final")
        arms = {"Raw+Synthetic": ck_clean}
        if len(ck_raw) == A.N_FOLDS:
            arms = {"Raw": ck_raw, "Raw+Synthetic": ck_clean}
        explain_v2(paths, arms=arms, tag="explainability_final", flag="explain_final")
        explain_v3(paths, src_tag="explainability_final", ck=ck_clean, tag="explainability_final_highres",
                   flag="explain_final_highres")
    else:
        print(f"Final-model QC-clean checkpoints: {len(ck_clean)} of {A.N_FOLDS} -> screening/explainability wait.")


# =============================================================================
# 15. Packaging for download
# =============================================================================
def package(session):
    """Two downloads: a compact paper zip (no per-run images) and the complete zip (everything but weights)."""
    A.write_marker(session)
    files = [f for f in A.RESULTS.rglob("*")
             if f.is_file() and f.suffix not in (".pt", ".lock") and ".cache" not in f.parts]
    rel = lambda f: f.relative_to(A.RESULTS)
    compact = [f for f in files
               if not (rel(f).parts[0] == "runs" and f.suffix.lower() in (".png", ".pdf", ".jpg", ".svg", ".npz", ".npy"))
               and rel(f).parts[0] != "embeddings"
               and not (rel(f).parts[:2] in (("qc_all", "selected"), ("qc_clean", "selected")) and len(rel(f).parts) == 4
                        and rel(f).parts[2] in CLASSES)]
    out = {}
    for tag, sel in (("paper", compact), ("complete", files)):
        z = A.WORK / f"afis_session{session}_{tag}.zip"
        if z.exists():
            z.unlink()
        with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
            for f in sel:
                zf.write(f, f.relative_to(A.WORK))
        print(f"{tag:9s}: {len(sel):6d} files -> {z.name} ({z.stat().st_size / 1e6:.1f} MB)")
        out[tag] = z
    print(f"Checkpoints stay in {A.RESULTS} for the next session.")
    return out["paper"]
