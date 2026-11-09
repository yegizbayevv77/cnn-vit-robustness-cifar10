"""Aggregate CIFAR-10-C results (results/*.json) into tables and figures for the paper.
Usage: python analyze.py results_dir out_dir
Works with any subset of the 6 runs; missing ones are skipped."""
import json, glob, os, sys
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)

SEEN = ["gaussian_noise", "shot_noise", "impulse_noise",
        "defocus_blur", "glass_blur", "motion_blur", "zoom_blur"]
UNSEEN = ["snow", "frost", "fog", "brightness",
          "contrast", "elastic_transform", "pixelate", "jpeg_compression"]
ALL = SEEN + UNSEEN
MODELS = ["resnet50", "deit_small"]
CONDS = ["baseline", "corrupt_aug", "augmix"]
MNAME = {"resnet50": "ResNet-50", "deit_small": "DeiT-S"}
CNAME = {"baseline": "Baseline", "corrupt_aug": "Corruption aug.", "augmix": "AugMix"}
COLOR = {"resnet50": "#2a78d6", "deit_small": "#eb6834"}
HATCH = {"resnet50": "", "deit_small": "//"}
INK, INK2, GRID = "#1f1f1e", "#5d5c55", "#e4e3dd"

runs = {}
for f in sorted(glob.glob(os.path.join(RES, "*.json"))):
    r = json.load(open(f))
    if r.get("quick_test"):
        continue
    runs.setdefault((r["model"], r["condition"]), []).append(r)
available = [(m, c) for c in CONDS for m in MODELS if (m, c) in runs]
print("runs found:", {f"{m}/{c}": len(runs[(m, c)]) for m, c in available})


def drops(r):
    clean = r["clean_acc"]
    per = {c: (clean - np.mean(r["corruptions"][c])) / clean for c in ALL}
    return {"clean": clean, "seen": np.mean([per[c] for c in SEEN]),
            "unseen": np.mean([per[c] for c in UNSEEN]),
            "all": np.mean([per[c] for c in ALL]), "per": per,
            "sev_seen": np.mean([r["corruptions"][c] for c in SEEN], axis=0),
            "sev_unseen": np.mean([r["corruptions"][c] for c in UNSEEN], axis=0)}

D = {k: [drops(r) for r in sorted(v, key=lambda r: r["seed"])] for k, v in runs.items()}


def vals(m, c, key):
    return np.array([d[key] for d in D[(m, c)]])


def ms(x, pct=True):
    s = 100 if pct else 1
    sd = np.std(x, ddof=1) if len(x) > 1 else 0.0
    return np.mean(x) * s, sd * s

# ---------- Table 1: main results ----------
rows, md = [], ["| Model | Training | n | Clean acc. (%) | Rel. drop seen (%) | Rel. drop unseen (%) | Rel. drop all (%) |",
                "|---|---|---|---|---|---|---|"]
tex = [r"\begin{tabular}{llcccc}", r"\toprule",
       r"Model & Training & Clean acc. (\%) & Seen drop (\%) & Unseen drop (\%) & All drop (\%) \\", r"\midrule"]
for c in CONDS:
    for m in MODELS:
        if (m, c) not in D:
            continue
        cells = [ms(vals(m, c, k)) for k in ["clean", "seen", "unseen", "all"]]
        f = [f"{a:.2f} ± {b:.2f}" for a, b in cells]
        md.append(f"| {MNAME[m]} | {CNAME[c]} | {len(D[(m, c)])} | " + " | ".join(f) + " |")
        tex.append(f"{MNAME[m]} & {CNAME[c]} & " + " & ".join(f"{a:.2f} $\\pm$ {b:.2f}" for a, b in cells) + r" \\")
        rows.append([MNAME[m], CNAME[c], len(D[(m, c)])] + [f"{a:.4f}" for a, _ in cells] + [f"{b:.4f}" for _, b in cells])
tex += [r"\bottomrule", r"\end{tabular}"]

# ---------- Table 2: CNN-ViT gap per training condition ----------
gap_md = ["| Training | Gap seen (pp) | p (Welch) | Ratio seen | Gap unseen (pp) | p (Welch) | Ratio unseen |", "|---|---|---|---|---|---|---|"]
gap_tex = [r"\begin{tabular}{lcccc}", r"\toprule",
           r"& \multicolumn{2}{c}{Seen} & \multicolumn{2}{c}{Unseen} \\", r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
           r"Training & Gap (pp) & Ratio & Gap (pp) & Ratio \\", r"\midrule"]
gaps = {}
for c in CONDS:
    if ("resnet50", c) not in D or ("deit_small", c) not in D:
        continue
    cells, cells_tex = [], []
    for k in ["seen", "unseen"]:
        cnn, vit = vals("resnet50", c, k), vals("deit_small", c, k)
        g = (np.mean(vit) - np.mean(cnn)) * 100
        sd = np.sqrt(np.var(vit, ddof=1) / len(vit) + np.var(cnn, ddof=1) / len(cnn)) * 100
        p = stats.ttest_ind(vit, cnn, equal_var=False).pvalue
        gaps[(c, k)] = (g, sd)
        ratio = np.mean(cnn) / np.mean(vit)
        gaps[(c, k + "_ratio")] = ratio
        cells += [f"{g:+.2f} ± {sd:.2f}", f"{p:.3g}", f"{ratio:.2f}"]
        cells_tex += [f"${g:+.2f} \\pm {sd:.2f}$", f"{ratio:.2f}"]
    gap_md.append(f"| {CNAME[c]} | " + " | ".join(cells) + " |")
    gap_tex.append(f"{CNAME[c]} & " + " & ".join(cells_tex) + r" \\")
gap_tex += [r"\bottomrule", r"\end{tabular}"]

# ---------- Table 3: effect of augmentation vs baseline (per model) ----------
eff_md = ["| Model | Training | Δ clean (pp) | Δ seen drop (pp) | Δ unseen drop (pp) |", "|---|---|---|---|---|"]
for m in MODELS:
    if (m, "baseline") not in D:
        continue
    for c in CONDS[1:]:
        if (m, c) not in D:
            continue
        d = [(np.mean(vals(m, c, k)) - np.mean(vals(m, "baseline", k))) * 100 for k in ["clean", "seen", "unseen"]]
        eff_md.append(f"| {MNAME[m]} | {CNAME[c]} | " + " | ".join(f"{x:+.2f}" for x in d) + " |")

# ---------- Table 4: per-corruption relative drop ----------
pc_md = ["| Corruption | Group | " + " | ".join(f"{MNAME[m]} {CNAME[c]}" for m, c in available) + " |",
         "|---|---|" + "---|" * len(available)]
for cor in ALL:
    cells = [f"{np.mean([d['per'][cor] for d in D[k]]) * 100:.1f}" for k in available]
    pc_md.append(f"| {cor} | {'seen' if cor in SEEN else 'unseen'} | " + " | ".join(cells) + " |")

with open(os.path.join(OUT, "results_summary.md"), "w") as f:
    f.write("## Table 1. Main results (mean ± std over seeds)\n\n" + "\n".join(md))
    f.write("\n\n## Table 2. Robustness gap = ViT drop − CNN drop (pp; ± is SE of difference); Ratio = CNN drop / ViT drop\n\n" + "\n".join(gap_md))
    f.write("\n\n## Table 3. Change relative to baseline (percentage points)\n\n" + "\n".join(eff_md))
    f.write("\n\n## Table 4. Relative drop per corruption (%, mean over seeds and severities)\n\n" + "\n".join(pc_md) + "\n")
with open(os.path.join(OUT, "table_main.tex"), "w") as f:
    f.write("\n".join(tex) + "\n")
with open(os.path.join(OUT, "table_gap.tex"), "w") as f:
    f.write("\n".join(gap_tex) + "\n")
with open(os.path.join(OUT, "table_main.csv"), "w") as f:
    f.write("model,condition,n,clean,seen_drop,unseen_drop,all_drop,clean_sd,seen_sd,unseen_sd,all_sd\n")
    f.writelines(",".join(map(str, r)) + "\n" for r in rows)

# ---------- Figures ----------
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK2,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False, "hatch.linewidth": 0.8})


def style(ax):
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


# Fig 1: relative drop (seen / unseen) per training condition
conds = [c for c in CONDS if any((m, c) in D for m in MODELS)]
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8), sharey=False)
for ax, k, title in zip(axes, ["seen", "unseen"], ["Seen corruptions (noise, blur)", "Unseen corruptions (weather, digital)"]):
    x = np.arange(len(conds))
    w = 0.36
    for i, m in enumerate(MODELS):
        mu, sd, xs = [], [], []
        for j, c in enumerate(conds):
            if (m, c) in D:
                a, b = ms(vals(m, c, k))
                mu.append(a); sd.append(b); xs.append(x[j] + (i - 0.5) * w)
        ax.bar(xs, mu, w * 0.92, yerr=sd, color=COLOR[m], hatch=HATCH[m], edgecolor="white",
               linewidth=0, capsize=2.5, error_kw={"elinewidth": 0.9, "ecolor": INK}, label=MNAME[m])
        for xx, a, b in zip(xs, mu, sd):
            ax.annotate(f"{a:.1f}", (xx, a + b), xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=7, color=INK2)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.12)
    ax.set_xticks(x, [CNAME[c] for c in conds])
    ax.set_title(title, fontsize=9, color=INK)
    ax.set_ylabel("Relative accuracy drop (%)")
    style(ax)
axes[0].legend(frameon=False, fontsize=8, loc="upper right")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_drop_by_condition.pdf"))
fig.savefig(os.path.join(OUT, "fig_drop_by_condition.png"), dpi=200)
plt.close(fig)

# Fig 2: per-corruption relative drop, baseline models
if all((m, "baseline") in D for m in MODELS):
    fig, ax = plt.subplots(figsize=(7.0, 2.9))
    x = np.arange(len(ALL))
    w = 0.38
    for i, m in enumerate(MODELS):
        mu = [np.mean([d["per"][c] for d in D[(m, "baseline")]]) * 100 for c in ALL]
        sd = [np.std([d["per"][c] for d in D[(m, "baseline")]], ddof=1) * 100 for c in ALL]
        ax.bar(x + (i - 0.5) * w, mu, w * 0.92, yerr=sd, color=COLOR[m], hatch=HATCH[m], edgecolor="white",
               linewidth=0, capsize=1.5, error_kw={"elinewidth": 0.7, "ecolor": INK}, label=MNAME[m])
    ax.axvline(len(SEEN) - 0.5, color=INK2, linewidth=0.8, linestyle="--")
    ymax = ax.get_ylim()[1] * 1.08
    ax.set_ylim(0, ymax)
    ax.text(len(SEEN) - 0.7, ymax * 0.98, "seen  ", ha="right", va="top", color=INK2, fontsize=8)
    ax.text(len(SEEN) - 0.3, ymax * 0.98, "  unseen", ha="left", va="top", color=INK2, fontsize=8)
    ax.set_xticks(x, [c.replace("_", " ") for c in ALL], rotation=40, ha="right", fontsize=7.5)
    ax.set_ylabel("Relative accuracy drop (%)")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    style(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_per_corruption_baseline.pdf"))
    fig.savefig(os.path.join(OUT, "fig_per_corruption_baseline.png"), dpi=200)
    plt.close(fig)

# Fig 3: accuracy vs severity
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7), sharey=True)
LS = {"baseline": "-", "corrupt_aug": "--", "augmix": ":"}
MK = {"resnet50": "o", "deit_small": "s"}
for ax, k, title in zip(axes, ["sev_seen", "sev_unseen"], ["Seen corruptions", "Unseen corruptions"]):
    for m, c in available:
        y = np.mean([d[k] for d in D[(m, c)]], axis=0) * 100
        clean = np.mean(vals(m, c, "clean")) * 100
        ax.plot(range(0, 6), [clean] + list(y), color=COLOR[m], linestyle=LS[c], marker=MK[m],
                markersize=4, linewidth=1.6, label=f"{MNAME[m]}, {CNAME[c]}")
    ax.set_xticks(range(0, 6), ["clean", "1", "2", "3", "4", "5"])
    ax.set_xlabel("Severity")
    ax.set_title(title, fontsize=9, color=INK)
    style(ax)
axes[0].set_ylabel("Accuracy (%)")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, frameon=False, fontsize=7.5, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02), handlelength=3.2)
fig.tight_layout(rect=(0, 0.12 if len(available) > 3 else 0.08, 1, 1))
fig.savefig(os.path.join(OUT, "fig_severity.pdf"))
fig.savefig(os.path.join(OUT, "fig_severity.png"), dpi=200)
plt.close(fig)

print(open(os.path.join(OUT, "results_summary.md")).read())
