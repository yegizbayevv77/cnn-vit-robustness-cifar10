"""Second-round analysis (after the instructor's review).
Usage: python analyze_v2.py results_dir out_dir
Adds: 95% CIs for every gap, a second CNN/ViT pair (ConvNeXt-T, Swin-T) with all four CNN-ViT pairings,
the cross-family transfer matrix (train on one family, test on all four), and a clean-accuracy/robustness frontier.
Every part is skipped silently if its runs are missing."""
import json, glob, os, sys
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

RES, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)

FAMILIES = {
    "noise": ["gaussian_noise", "shot_noise", "impulse_noise"],
    "blur": ["defocus_blur", "glass_blur", "motion_blur", "zoom_blur"],
    "weather": ["snow", "frost", "fog", "brightness"],
    "digital": ["contrast", "elastic_transform", "pixelate", "jpeg_compression"],
}
FAMS = list(FAMILIES)
SEEN = FAMILIES["noise"] + FAMILIES["blur"]
UNSEEN = FAMILIES["weather"] + FAMILIES["digital"]
ALL = SEEN + UNSEEN
CNNS, VITS = ["resnet50", "convnext_tiny"], ["deit_small", "swin_tiny"]
MNAME = {"resnet50": "ResNet-50", "deit_small": "DeiT-S", "convnext_tiny": "ConvNeXt-T", "swin_tiny": "Swin-T"}
CNAME = {"baseline": "Baseline", "corrupt_aug": "Noise+blur aug.", "augmix": "AugMix",
         "aug_noise": "Noise aug.", "aug_blur": "Blur aug.", "aug_digital": "Digital aug."}
CNAME.update({"nb_p25": "Noise+blur, $p=0.25$", "nb_p100": "Noise+blur, $p=1$", "nb_weak": "Noise+blur, $S=0.5$", "nb_strong": "Noise+blur, $S=1.5$"})
TRAINED_FAMS = {"baseline": [], "aug_noise": ["noise"], "aug_blur": ["blur"],
                "corrupt_aug": ["noise", "blur"], "aug_digital": ["digital"]}
# categorical slots 1-4 of the validated palette; ViTs also get a hatch so identity is not colour-only
COLOR = {"resnet50": "#2a78d6", "deit_small": "#eb6834", "convnext_tiny": "#1baf7a", "swin_tiny": "#eda100"}
HATCH = {"resnet50": "", "convnext_tiny": "", "deit_small": "//", "swin_tiny": "//"}
INK, INK2, GRID = "#1f1f1e", "#5d5c55", "#e4e3dd"
BLUES = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
         "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK2,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False, "hatch.linewidth": 0.8})

runs = {}
for f in sorted(glob.glob(os.path.join(RES, "*.json"))):
    r = json.load(open(f))
    if r.get("quick_test") or not r.get("pretrained", True):
        continue
    runs.setdefault((r["model"], r["condition"]), {})[r["seed"]] = r
print("runs:", {f"{m}/{c}": len(v) for (m, c), v in sorted(runs.items())})


def summary(r):
    clean = r["clean_acc"]
    per = {c: (clean - np.mean(r["corruptions"][c])) / clean for c in ALL}
    out = {"clean": clean, "per": per,
           "seen": np.mean([per[c] for c in SEEN]), "unseen": np.mean([per[c] for c in UNSEEN]),
           "all": np.mean([per[c] for c in ALL]),
           "corr_acc": np.mean([np.mean(r["corruptions"][c]) for c in ALL])}
    for fam, cs in FAMILIES.items():
        out[fam] = np.mean([per[c] for c in cs])
    return out

S = {k: [summary(v[s]) for s in sorted(v)] for k, v in runs.items()}
has = lambda m, c: (m, c) in S and len(S[(m, c)]) >= 2


def vals(m, c, key):
    return np.array([d[key] for d in S[(m, c)]])


def gap(cnn, vit, cond, key):
    """ViT drop - CNN drop in pp, Welch 95% CI, p-value, and ratio CNN/ViT."""
    a, b = vals(cnn, cond, key), vals(vit, cond, key)
    g = (b.mean() - a.mean()) * 100
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    se = np.sqrt(va + vb)
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1)) if se > 0 else 1
    h = stats.t.ppf(0.975, df) * se * 100
    p = stats.ttest_ind(b, a, equal_var=False).pvalue
    return {"g": g, "lo": g - h, "hi": g + h, "p": p, "ratio": a.mean() / b.mean()}


def fmt_ci(d):
    return f"${d['g']:+.1f}$ [{d['lo']:+.1f}, {d['hi']:+.1f}]"


md = []

# ---------- Table: ResNet-50 vs DeiT-S gap with 95% CI (replaces table_gap.tex) ----------
conds3 = [c for c in ["baseline", "corrupt_aug", "augmix"] if has("resnet50", c) and has("deit_small", c)]
if conds3:
    tex = [r"\begin{tabular}{lcccc}", r"\toprule",
           r"& \multicolumn{2}{c}{Seen (noise, blur)} & \multicolumn{2}{c}{Unseen (weather, digital)} \\",
           r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
           r"Training & Gap, pp [95\% CI] & Ratio & Gap, pp [95\% CI] & Ratio \\", r"\midrule"]
    md += ["## Gap ResNet-50 vs DeiT-S (ViT - CNN, pp, 95% Welch CI), ratio CNN/ViT", "",
           "| Training | seen gap | p | ratio | unseen gap | p | ratio |", "|---|---|---|---|---|---|---|"]
    base = {k: gap("resnet50", "deit_small", "baseline", k) for k in ["seen", "unseen"]} if "baseline" in conds3 else None
    for c in conds3:
        d = {k: gap("resnet50", "deit_small", c, k) for k in ["seen", "unseen"]}
        tex.append(f"{CNAME[c]} & {fmt_ci(d['seen'])} & {d['seen']['ratio']:.2f} & {fmt_ci(d['unseen'])} & {d['unseen']['ratio']:.2f} \\\\")
        red = "" if base is None or c == "baseline" else \
            f" (abs. gap reduced {100 * (1 - d['seen']['g'] / base['seen']['g']):.0f}% seen, {100 * (1 - d['unseen']['g'] / base['unseen']['g']):.0f}% unseen)"
        md.append(f"| {CNAME[c]} | {d['seen']['g']:+.2f} [{d['seen']['lo']:+.2f}, {d['seen']['hi']:+.2f}] | {d['seen']['p']:.2g} | {d['seen']['ratio']:.2f} | "
                  f"{d['unseen']['g']:+.2f} [{d['unseen']['lo']:+.2f}, {d['unseen']['hi']:+.2f}] | {d['unseen']['p']:.2g} | {d['unseen']['ratio']:.2f} |{red}")
    tex += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(OUT, "table_gap.tex"), "w").write("\n".join(tex) + "\n")

# ---------- Second pair: all four models, baseline and noise+blur aug ----------
models4 = [m for m in CNNS + VITS if has(m, "baseline")]
conds2 = [c for c in ["baseline", "corrupt_aug"] if all(has(m, c) for m in models4)]
if len(models4) == 4 and conds2:
    def ms(x):
        return f"{x.mean() * 100:.2f} $\\pm$ {x.std(ddof=1) * 100:.2f}"
    tex = [r"\begin{tabular}{llccc}", r"\toprule",
           r"Training & Model & Clean acc. (\%) & Seen drop (\%) & Unseen drop (\%) \\", r"\midrule"]
    md += ["", "## Four models (mean ± std over seeds)", "", "| Training | Model | clean | seen | unseen |", "|---|---|---|---|---|"]
    for c in conds2:
        for i, m in enumerate(CNNS + VITS):
            tex.append(f"{CNAME[c] if i == 0 else ''} & {MNAME[m]} & " + " & ".join(ms(vals(m, c, k)) for k in ["clean", "seen", "unseen"]) + r" \\")
            md.append(f"| {CNAME[c]} | {MNAME[m]} | " + " | ".join(f"{vals(m, c, k).mean() * 100:.2f} ± {vals(m, c, k).std(ddof=1) * 100:.2f}" for k in ["clean", "seen", "unseen"]) + " |")
        if c != conds2[-1]:
            tex.append(r"\midrule")
    tex += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(OUT, "table_models4.tex"), "w").write("\n".join(tex) + "\n")

    # all four CNN-ViT pairings
    tex = [r"\begin{tabular}{llcccc}", r"\toprule",
           r"& & \multicolumn{2}{c}{Seen} & \multicolumn{2}{c}{Unseen} \\", r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
           r"Training & CNN -- ViT & Gap, pp [95\% CI] & Ratio & Gap, pp [95\% CI] & Ratio \\", r"\midrule"]
    md += ["", "## All CNN-ViT pairings", "", "| Training | pair | seen gap [CI] | ratio | unseen gap [CI] | ratio |", "|---|---|---|---|---|---|"]
    for c in conds2:
        for i, (cn, vt) in enumerate([(a, b) for a in CNNS for b in VITS]):
            d = {k: gap(cn, vt, c, k) for k in ["seen", "unseen"]}
            tex.append(f"{CNAME[c] if i == 0 else ''} & {MNAME[cn]} -- {MNAME[vt]} & {fmt_ci(d['seen'])} & {d['seen']['ratio']:.2f} & {fmt_ci(d['unseen'])} & {d['unseen']['ratio']:.2f} \\\\")
            md.append(f"| {CNAME[c]} | {MNAME[cn]}-{MNAME[vt]} | {d['seen']['g']:+.2f} [{d['seen']['lo']:+.2f}, {d['seen']['hi']:+.2f}] | {d['seen']['ratio']:.2f} | "
                      f"{d['unseen']['g']:+.2f} [{d['unseen']['lo']:+.2f}, {d['unseen']['hi']:+.2f}] | {d['unseen']['ratio']:.2f} |")
        if c != conds2[-1]:
            tex.append(r"\midrule")
    tex += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(OUT, "table_pairs.tex"), "w").write("\n".join(tex) + "\n")

    # within-family differences (model identity) vs between-family differences (architecture family)
    md += ["", "## Within-family vs between-family spread (baseline, pp of relative drop)", ""]
    for k in ["seen", "unseen"]:
        mu = {m: vals(m, "baseline", k).mean() * 100 for m in CNNS + VITS}
        md.append(f"- {k}: CNNs {', '.join(f'{MNAME[m]} {mu[m]:.2f}' for m in CNNS)} | ViTs {', '.join(f'{MNAME[m]} {mu[m]:.2f}' for m in VITS)} | "
                  f"within-CNN diff {abs(mu['resnet50'] - mu['convnext_tiny']):.2f}, within-ViT diff {abs(mu['deit_small'] - mu['swin_tiny']):.2f}, "
                  f"worst CNN - best ViT {max(mu[m] for m in CNNS) - min(mu[m] for m in VITS):.2f}, best CNN - worst ViT {min(mu[m] for m in CNNS) - max(mu[m] for m in VITS):.2f}")

    # figure: seen/unseen drop for the four models
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7))
    for ax, k, title in zip(axes, ["seen", "unseen"], ["Seen corruptions (noise, blur)", "Unseen corruptions (weather, digital)"]):
        x = np.arange(len(conds2))
        w = 0.2
        for i, m in enumerate(CNNS + VITS):
            mu = [vals(m, c, k).mean() * 100 for c in conds2]
            sd = [vals(m, c, k).std(ddof=1) * 100 for c in conds2]
            xs = x + (i - 1.5) * w
            ax.bar(xs, mu, w * 0.9, yerr=sd, color=COLOR[m], hatch=HATCH[m], edgecolor="white", linewidth=0,
                   capsize=2, error_kw={"elinewidth": 0.8, "ecolor": INK}, label=MNAME[m])
            for xx, a, b in zip(xs, mu, sd):
                ax.annotate(f"{a:.1f}", (xx, a + b), xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=6.5, color=INK2)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.12)
        ax.set_xticks(x, [CNAME[c] for c in conds2])
        ax.set_title(title, fontsize=9, color=INK)
        ax.set_ylabel("Relative accuracy drop (%)")
        ax.yaxis.grid(True, color=GRID, linewidth=0.8); ax.set_axisbelow(True)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=8, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    for ext in ["pdf", "png"]:
        fig.savefig(os.path.join(OUT, f"fig_four_models.{ext}"), dpi=200)
    plt.close(fig)

# ---------- Transfer matrix: ResNet-50 / DeiT-S trained on one family, tested on all four ----------
tconds = [c for c in ["baseline", "aug_noise", "aug_blur", "corrupt_aug", "aug_digital"]
          if has("resnet50", c) and has("deit_small", c)]
if len(tconds) >= 3:
    mat = {m: np.array([[vals(m, c, f).mean() * 100 for f in FAMS] for c in tconds]) for m in ["resnet50", "deit_small"]}
    ratio = mat["resnet50"] / mat["deit_small"]
    tex = [r"\begin{tabular}{l" + "cc" * len(FAMS) + "}", r"\toprule",
           "Training & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{f.capitalize()}}}" for f in FAMS) + r" \\",
           "".join(rf"\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(FAMS))),
           "& " + " & ".join(["R & D"] * len(FAMS)) + r" \\", r"\midrule"]
    md += ["", "## Transfer matrix: relative drop (%) per test family; R=ResNet-50, D=DeiT-S; * = family used in training", "",
           "| Training | " + " | ".join(f"{f} R | {f} D | {f} ratio" for f in FAMS) + " |", "|---|" + "---|" * (3 * len(FAMS))]
    for i, c in enumerate(tconds):
        cells, mdc = [], []
        for j, f in enumerate(FAMS):
            r_, d_ = mat["resnet50"][i, j], mat["deit_small"][i, j]
            star = f in TRAINED_FAMS[c]
            cells += [rf"\textbf{{{r_:.1f}}}" if star else f"{r_:.1f}", rf"\textbf{{{d_:.1f}}}" if star else f"{d_:.1f}"]
            mdc += [f"{r_:.1f}{'*' if star else ''}", f"{d_:.1f}{'*' if star else ''}", f"{ratio[i, j]:.2f}"]
        tex.append(f"{CNAME[c]} & " + " & ".join(cells) + r" \\")
        md.append(f"| {CNAME[c]} | " + " | ".join(mdc) + " |")
    tex += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(OUT, "table_transfer.tex"), "w").write("\n".join(tex) + "\n")

    # held-out summary: for each single-family augmentation, gap change on the trained family vs the three others
    if "baseline" in tconds:
        b = tconds.index("baseline")
        tex = [r"\begin{tabular}{lcccc}", r"\toprule",
               r"& \multicolumn{2}{c}{Trained family} & \multicolumn{2}{c}{Held-out families} \\",
               r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
               r"Training & Gap red. (\%) & Ratio (base $\to$ aug.) & Gap red. (\%) & Ratio (base $\to$ aug.) \\", r"\midrule"]
        md += ["", "## Held-out analysis (absolute gap reduction vs baseline and ratio, trained family vs mean of the other families)", ""]
        for c in tconds:
            if c == "baseline":
                continue
            tr = [FAMS.index(f) for f in TRAINED_FAMS[c]]
            ho = [j for j in range(len(FAMS)) if j not in tr]
            def agg(idx, row):
                rr, dd = mat["resnet50"][row, idx].mean(), mat["deit_small"][row, idx].mean()
                return dd - rr, rr / dd
            (g0t, r0t), (g1t, r1t) = agg(tr, b), agg(tr, tconds.index(c))
            (g0h, r0h), (g1h, r1h) = agg(ho, b), agg(ho, tconds.index(c))
            tex.append(f"{CNAME[c]} & ${100 * (1 - g1t / g0t):.0f}$ & {r0t:.2f} $\\to$ {r1t:.2f} & ${100 * (1 - g1h / g0h):.0f}$ & {r0h:.2f} $\\to$ {r1h:.2f} \\\\")
            md.append(f"- {CNAME[c]}: trained gap {g0t:+.2f} -> {g1t:+.2f} pp ({100 * (1 - g1t / g0t):.0f}% red.), ratio {r0t:.2f} -> {r1t:.2f}; "
                      f"held-out gap {g0h:+.2f} -> {g1h:+.2f} pp ({100 * (1 - g1h / g0h):.0f}% red.), ratio {r0h:.2f} -> {r1h:.2f}")
        tex += [r"\bottomrule", r"\end{tabular}"]
        open(os.path.join(OUT, "table_heldout.tex"), "w").write("\n".join(tex) + "\n")

    # figure: three heatmaps (ResNet drop, DeiT drop, ratio), trained-family cells outlined
    from matplotlib.colors import ListedColormap, Normalize
    cmap = ListedColormap(BLUES)
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.85), gridspec_kw={"wspace": 0.08})
    vmax = max(mat["resnet50"].max(), mat["deit_small"].max())
    panels = [(mat["resnet50"], "ResNet-50: relative drop (%)", Normalize(0, vmax), "{:.1f}"),
              (mat["deit_small"], "DeiT-S: relative drop (%)", Normalize(0, vmax), "{:.1f}"),
              (ratio, "Ratio ResNet-50 / DeiT-S", Normalize(1, max(3.0, ratio.max())), "{:.2f}")]
    for k, (ax, (M, title, norm, f)) in enumerate(zip(axes, panels)):
        ax.imshow(M, cmap=cmap, norm=norm, aspect="auto")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                light = norm(M[i, j]) < 0.55
                ax.text(j, i, f.format(M[i, j]), ha="center", va="center", fontsize=7.5, color=INK if light else "white")
                if FAMS[j] in TRAINED_FAMS[tconds[i]]:
                    ax.add_patch(Rectangle((j - 0.47, i - 0.47), 0.94, 0.94, fill=False, edgecolor=INK, linewidth=1.4))
        ax.set_xticks(range(len(FAMS)), [f.capitalize() for f in FAMS], fontsize=7.5)
        ax.set_yticks(range(len(tconds)), [CNAME[c] for c in tconds] if k == 0 else [], fontsize=7.5)
        ax.set_title(title, fontsize=8.5, color=INK)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
    fig.text(0.57, 0.02, "Test family (CIFAR-10-C)   ·   outlined cells: family used in training", ha="center", va="bottom", fontsize=7.5, color=INK2)
    fig.subplots_adjust(left=0.15, right=0.99, top=0.9, bottom=0.2)
    for ext in ["pdf", "png"]:
        fig.savefig(os.path.join(OUT, f"fig_transfer.{ext}"), dpi=200)
    plt.close(fig)

# ---------- Clean accuracy vs corrupted accuracy (frontier) ----------
SENS = {"nb_p25": ("p", 0.25), "nb_p100": ("p", 1.0), "nb_weak": ("S", 0.5), "nb_strong": ("S", 1.5)}
pts = [(m, c) for (m, c) in S if len(S[(m, c)]) >= 2 and c not in SENS]   # sensitivity runs get their own figure
if pts:
    MK = {"baseline": "o", "corrupt_aug": "s", "augmix": "D", "aug_noise": "^", "aug_blur": "v", "aug_digital": "P"}
    # two panels with a shared y axis: ResNet-50 sits ~2 pp lower in clean accuracy than the rest,
    # so one x axis would squeeze the other three models into one corner
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(7.0, 3.1), sharey=True, gridspec_kw={"width_ratios": [1, 1.25], "wspace": 0.06})
    for m, c in sorted(pts):
        x, y = vals(m, c, "clean") * 100, vals(m, c, "corr_acc") * 100
        a = axL if m == "resnet50" else axR
        a.errorbar(x.mean(), y.mean(), xerr=x.std(ddof=1), yerr=y.std(ddof=1), fmt=MK[c], color=COLOR[m],
                   markersize=6.5, markeredgecolor="white", markeredgewidth=0.8, elinewidth=0.8, capsize=1.5)
    for a in (axL, axR):
        a.yaxis.grid(True, color=GRID, linewidth=0.8); a.xaxis.grid(True, color=GRID, linewidth=0.8); a.set_axisbelow(True)
    axL.set_title("ResNet-50", fontsize=8.5, color=INK)
    axR.set_title("ConvNeXt-T, DeiT-S, Swin-T", fontsize=8.5, color=INK)
    axR.tick_params(axis="y", length=0)
    axR.spines["left"].set_visible(False)
    from matplotlib.ticker import MaxNLocator
    for a in (axL, axR):
        a.xaxis.set_major_locator(MaxNLocator(4))
        a.margins(x=0.08)
    fig.supxlabel("Clean accuracy (%)", fontsize=9, color=INK, y=0.0)
    axL.set_ylabel("Mean CIFAR-10-C accuracy (%)")
    ax = axR
    from matplotlib.lines import Line2D
    used_m = [m for m in CNNS + VITS if any(p[0] == m for p in pts)]
    used_c = [c for c in MK if any(p[1] == c for p in pts)]
    h1 = [Line2D([], [], marker="o", linestyle="", color=COLOR[m], markersize=6, label=MNAME[m]) for m in used_m]
    h2 = [Line2D([], [], marker=MK[c], linestyle="", color=INK2, markersize=5.5, label=CNAME[c]) for c in used_c]
    leg1 = ax.legend(handles=h1, title="Model (colour)", frameon=False, fontsize=7.5, title_fontsize=7.5, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    ax.add_artist(leg1)
    ax.legend(handles=h2, title="Training (marker)", frameon=False, fontsize=7.5, title_fontsize=7.5, loc="lower left", bbox_to_anchor=(1.01, 0.0))
    fig.subplots_adjust(left=0.09, right=0.78, top=0.91, bottom=0.16)
    for ext in ["pdf", "png"]:
        fig.savefig(os.path.join(OUT, f"fig_frontier.{ext}"), dpi=200)
    plt.close(fig)


# ---------- Sensitivity of noise+blur augmentation to probability p and intensity S (ResNet-50 vs DeiT-S) ----------
SENS_ROWS = [("p", 0.25, "nb_p25"), ("p", 0.5, "corrupt_aug"), ("p", 1.0, "nb_p100"),
             ("S", 0.5, "nb_weak"), ("S", 1.0, "corrupt_aug"), ("S", 1.5, "nb_strong")]
if all(has(m, c) for m in ["resnet50", "deit_small"] for _, _, c in SENS_ROWS) and has("resnet50", "baseline"):
    tex = [r"\begin{tabular}{llcccccc}", r"\toprule",
           r"& & \multicolumn{2}{c}{Clean acc. (\%)} & \multicolumn{2}{c}{Seen} & \multicolumn{2}{c}{Unseen} \\",
           r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}",
           r"Varied & Value & R & D & Gap, pp [95\% CI] & Ratio & Gap, pp [95\% CI] & Ratio \\", r"\midrule"]
    md += ["", "## Sensitivity of noise+blur aug. (p = probability, S = intensity multiplier); R = ResNet-50, D = DeiT-S", "",
           "| varied | value | clean R | clean D | seen R | seen D | unseen R | unseen D | seen gap [CI] | seen ratio | unseen gap [CI] | unseen ratio |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    base = {k: gap("resnet50", "deit_small", "baseline", k) for k in ["seen", "unseen"]}
    for i, (var, val, c) in enumerate(SENS_ROWS):
        d = {k: gap("resnet50", "deit_small", c, k) for k in ["seen", "unseen"]}
        cl = [vals(m, c, "clean").mean() * 100 for m in ["resnet50", "deit_small"]]
        dr = {(m, k): vals(m, c, k).mean() * 100 for m in ["resnet50", "deit_small"] for k in ["seen", "unseen"]}
        label = ("Probability $p$" if var == "p" else "Intensity $S$") if i % 3 == 0 else ""
        star = r"$^\dagger$" if c == "corrupt_aug" else ""
        tex.append(f"{label} & {val:g}{star} & {cl[0]:.2f} & {cl[1]:.2f} & {fmt_ci(d['seen'])} & {d['seen']['ratio']:.2f} & {fmt_ci(d['unseen'])} & {d['unseen']['ratio']:.2f} \\\\")
        if i == 2:
            tex.append(r"\midrule")
        red = {k: 100 * (1 - d[k]["g"] / base[k]["g"]) for k in ["seen", "unseen"]}
        md.append(f"| {var} | {val:g} | {cl[0]:.2f} | {cl[1]:.2f} | {dr[('resnet50','seen')]:.2f} | {dr[('deit_small','seen')]:.2f} | {dr[('resnet50','unseen')]:.2f} | {dr[('deit_small','unseen')]:.2f} | "
                  f"{d['seen']['g']:+.2f} [{d['seen']['lo']:+.2f}, {d['seen']['hi']:+.2f}] ({red['seen']:.0f}% red.) | {d['seen']['ratio']:.2f} | "
                  f"{d['unseen']['g']:+.2f} [{d['unseen']['lo']:+.2f}, {d['unseen']['hi']:+.2f}] ({red['unseen']:.0f}% red.) | {d['unseen']['ratio']:.2f} |")
    tex += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(OUT, "table_sensitivity.tex"), "w").write("\n".join(tex) + "\n")

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7), sharey=True)
    for ax, var, title, xl in zip(axes, ["p", "S"], ["Augmentation probability", "Augmentation intensity"],
                                  ["Probability $p$ (share of corrupted images)", "Intensity multiplier $S$"]):
        rows = [(val, c) for v, val, c in SENS_ROWS if v == var]
        xs = [0.0] + [val for val, _ in rows]
        for m in ["resnet50", "deit_small"]:
            for k, ls, mk in [("seen", "-", "o"), ("unseen", "--", "s")]:
                mu = [vals(m, "baseline", k).mean() * 100] + [vals(m, c, k).mean() * 100 for _, c in rows]
                sd = [vals(m, "baseline", k).std(ddof=1) * 100] + [vals(m, c, k).std(ddof=1) * 100 for _, c in rows]
                ax.errorbar(xs, mu, yerr=sd, color=COLOR[m], linestyle=ls, marker=mk, markersize=4.5, linewidth=1.6,
                            capsize=1.5, elinewidth=0.8, label=f"{MNAME[m]}, {k}")
        ax.set_xticks(xs, ["none"] + [f"{x:g}" for x in xs[1:]])
        ax.set_xlabel(xl)
        ax.set_title(title, fontsize=9, color=INK)
        ax.yaxis.grid(True, color=GRID, linewidth=0.8); ax.set_axisbelow(True)
    axes[0].set_ylabel("Relative accuracy drop (%)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=7.5, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.01), handlelength=3)
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    for ext in ["pdf", "png"]:
        fig.savefig(os.path.join(OUT, f"fig_sensitivity.{ext}"), dpi=200)
    plt.close(fig)

open(os.path.join(OUT, "results_summary_v2.md"), "w").write("\n".join(md) + "\n")
print("\n".join(md))
