# %% [markdown]
# # CNN vs ViT robustness under augmentation (CIFAR-10 -> CIFAR-10-C)
# Shynbergen Yegizbayev, AITU, Computer Vision paper
#
# **How to run on Kaggle**
# 1. Settings (right panel): Accelerator = **GPU T4 x2** (or P100), Internet = **On**.
# 2. Change only the CONFIG cell below (MODEL and CONDITION).
# 3. First time: set `QUICK_TEST = True` and click **Run All** (about 10 minutes). It prints a time estimate.
# 4. Real run: set `QUICK_TEST = False`, then **Save Version -> Save & Run All (Commit)**. It runs in the background,
#    you can close the browser. Results appear in the **Output** tab as `results/*.json`.
#
# The 6 runs: MODEL in {resnet50, deit_small} x CONDITION in {baseline, corrupt_aug, augmix}, each with 3 seeds.

# %%
# ======================= CONFIG (change only this cell) =======================
MODEL = "resnet50"        # "resnet50" or "deit_small"
CONDITION = "baseline"    # "baseline", "corrupt_aug" or "augmix"
SEEDS = [0, 1, 2]
EPOCHS = 6
BATCH_SIZE = 128
LR = 1e-4
WEIGHT_DECAY = 0.05
LABEL_SMOOTHING = 0.1
IMG_SIZE = 224
QUICK_TEST = False        # True = tiny run (2000 train images, 1 epoch, 500 test images) to check everything works
# ==============================================================================

# %%
import subprocess, sys
try:
    import timm
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "timm"], check=True)
    import timm
import timm.optim

import os, glob, json, time, random, math, tarfile, urllib.request
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
import torchvision.transforms.functional as TF

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP = DEVICE == "cuda"
ON_KAGGLE = os.path.exists("/kaggle/working")
OUT_DIR = "/kaggle/working/results" if ON_KAGGLE else "./results"
DATA_DIR = "/tmp/cvdata"            # not saved as notebook output
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# Both models: ImageNet-1k pretraining only, similar size (~22-25M params).
# resnet50.a1_in1k uses the "ResNet strikes back" A1 recipe, which is close to the DeiT recipe.
TIMM_NAMES = {
    "resnet50": "resnet50.a1_in1k",
    "deit_small": "deit_small_patch16_224.fb_in1k",
}

# CIFAR-10-C: 15 standard corruptions, split by family.
# "seen" = noise and blur families (corrupt_aug trains on our own noise/blur implementations, never on CIFAR-10-C images)
SEEN = ["gaussian_noise", "shot_noise", "impulse_noise",
        "defocus_blur", "glass_blur", "motion_blur", "zoom_blur"]
UNSEEN = ["snow", "frost", "fog", "brightness",
          "contrast", "elastic_transform", "pixelate", "jpeg_compression"]
CORRUPTIONS = SEEN + UNSEEN

print("device:", DEVICE, "| torch", torch.__version__, "| timm", timm.__version__)
if DEVICE == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))

# %%
# ---------------- Data ----------------
C10C_URL = "https://zenodo.org/records/2535967/files/CIFAR-10-C.tar"

def find_cifar10c():
    """Use a Kaggle input dataset if one is attached, otherwise download from Zenodo (about 2.9 GB, 5-10 min)."""
    for path in glob.glob("/kaggle/input/**/gaussian_noise.npy", recursive=True):
        d = os.path.dirname(path)
        if os.path.exists(os.path.join(d, "labels.npy")):
            return d
    d = os.path.join(DATA_DIR, "CIFAR-10-C")
    needed = set(c + ".npy" for c in CORRUPTIONS) | {"labels.npy"}
    if os.path.isdir(d) and needed.issubset(os.listdir(d)):
        return d
    os.makedirs(d, exist_ok=True)
    print("Downloading CIFAR-10-C from Zenodo (streamed, only the needed files are kept)...")
    t0 = time.time()
    with urllib.request.urlopen(C10C_URL) as resp, tarfile.open(fileobj=resp, mode="r|") as tar:
        for member in tar:
            name = os.path.basename(member.name)
            if member.isfile() and name in needed:
                with tar.extractfile(member) as src, open(os.path.join(d, name), "wb") as dst:
                    dst.write(src.read())
                print("  got", name)
    missing = needed - set(os.listdir(d))
    assert not missing, f"CIFAR-10-C files missing: {missing}"
    print(f"CIFAR-10-C ready in {time.time() - t0:.0f}s")
    return d


def load_cifar10():
    """Use CIFAR-10 (python version) from an attached Kaggle input if present, otherwise download it."""
    target = os.path.join(DATA_DIR, "cifar-10-batches-py")
    if not os.path.exists(target):
        for path in glob.glob("/kaggle/input/**/data_batch_1", recursive=True):
            src = os.path.dirname(path)
            if os.path.exists(os.path.join(src, "test_batch")):
                os.symlink(src, target)
                print("Using CIFAR-10 from", src)
                break
    try:
        train = torchvision.datasets.CIFAR10(DATA_DIR, train=True, download=False)
        test = torchvision.datasets.CIFAR10(DATA_DIR, train=False, download=False)
    except RuntimeError:
        if os.path.islink(target):
            os.unlink(target)
        print("CIFAR-10 not found in inputs, downloading (can be slow)...")
        train = torchvision.datasets.CIFAR10(DATA_DIR, train=True, download=True)
        test = torchvision.datasets.CIFAR10(DATA_DIR, train=False, download=True)
    return train, test


class TrainSet(torch.utils.data.Dataset):
    """Returns uint8 tensors at 32x32. Resizing to 224 and normalisation happen on the GPU."""
    def __init__(self, base, condition, indices=None):
        self.base = base
        self.indices = list(range(len(base))) if indices is None else list(indices)
        ops = [T.RandomCrop(32, padding=4), T.RandomHorizontalFlip()]
        if condition == "augmix":
            # all_ops=False excludes contrast/color/brightness/sharpness, which overlap with CIFAR-10-C
            ops.append(T.AugMix(all_ops=False))
        ops.append(T.PILToTensor())
        self.tf = T.Compose(ops)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        img, label = self.base[self.indices[i]]
        return self.tf(img), label

# %%
# ---------------- Corruption augmentation (noise + blur families, applied on GPU at 32x32) ----------------
def _gaussian_noise(x):
    sigma = torch.empty(x.size(0), 1, 1, 1, device=x.device).uniform_(0.02, 0.10)
    return x + torch.randn_like(x) * sigma

def _shot_noise(x):
    lam = torch.empty(x.size(0), 1, 1, 1, device=x.device).uniform_(50, 500)
    return torch.poisson(x * lam) / lam

def _impulse_noise(x):
    amount = torch.empty(x.size(0), 1, 1, 1, device=x.device).uniform_(0.01, 0.07)
    r = torch.rand_like(x)
    x = torch.where(r < amount / 2, torch.zeros_like(x), x)
    return torch.where(r > 1 - amount / 2, torch.ones_like(x), x)

def _gaussian_blur(x):
    sigma = random.uniform(0.4, 1.5)
    return TF.gaussian_blur(x, kernel_size=5, sigma=sigma)

def _motion_blur(x):
    k = random.choice([3, 5, 7])
    angle = random.uniform(0, math.pi)
    kernel = torch.zeros(k, k, device=x.device)
    c = k // 2
    for t in torch.linspace(-c, c, steps=4 * k):
        i = int(round(c + float(t) * math.sin(angle)))
        j = int(round(c + float(t) * math.cos(angle)))
        kernel[i, j] = 1.0
    kernel = (kernel / kernel.sum()).expand(x.size(1), 1, k, k).to(x.dtype)
    x = F.pad(x, (c, c, c, c), mode="reflect")
    return F.conv2d(x, kernel, groups=x.size(1))

CORRUPT_OPS = [_gaussian_noise, _shot_noise, _impulse_noise, _gaussian_blur, _motion_blur]

def corrupt_batch(x, p=0.5):
    """x: float in [0,1], shape (B,3,32,32). Each image is corrupted with probability p by one random op."""
    apply = torch.rand(x.size(0), device=x.device) < p
    op_id = torch.randint(0, len(CORRUPT_OPS), (x.size(0),), device=x.device)
    out = x.clone()
    for k, op in enumerate(CORRUPT_OPS):
        idx = torch.nonzero(apply & (op_id == k)).flatten()
        if idx.numel() > 0:
            out[idx] = op(x[idx])
    return out.clamp_(0, 1)

# %%
# ---------------- Model, training, evaluation ----------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(pretrained=True):
    model = timm.create_model(TIMM_NAMES[MODEL], pretrained=pretrained, num_classes=10)
    cfg = model.pretrained_cfg
    mean = torch.tensor(cfg["mean"], device=DEVICE).view(1, 3, 1, 1)
    std = torch.tensor(cfg["std"], device=DEVICE).view(1, 3, 1, 1)
    model = model.to(DEVICE).to(memory_format=torch.channels_last)
    return model, mean, std


def prepare(x_uint8, mean, std, corrupt=False):
    x = x_uint8.to(DEVICE, non_blocking=True).float().div_(255)
    if corrupt:
        x = corrupt_batch(x)
    x = F.interpolate(x, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)
    x = (x - mean) / std
    return x.contiguous(memory_format=torch.channels_last)


@torch.no_grad()
def accuracy_on_array(model, mean, std, images, labels, batch_size=256):
    """images: numpy uint8 (N,32,32,3); labels: numpy int (N,)."""
    model.eval()
    correct = 0
    for s in range(0, len(images), batch_size):
        xb = torch.from_numpy(np.ascontiguousarray(images[s:s + batch_size])).permute(0, 3, 1, 2)
        yb = torch.from_numpy(np.asarray(labels[s:s + batch_size])).long().to(DEVICE)
        with torch.autocast("cuda", dtype=torch.float16, enabled=USE_AMP):
            pred = model(prepare(xb, mean, std)).argmax(1)
        correct += (pred == yb).sum().item()
    return correct / len(images)


def train_model(seed, train_base, pretrained=True):
    set_seed(seed)
    model, mean, std = build_model(pretrained)
    n_train = 2000 if QUICK_TEST else len(train_base)
    epochs = 1 if QUICK_TEST else EPOCHS
    g = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(train_base), generator=g)[:n_train].tolist()
    ds = TrainSet(train_base, CONDITION, indices)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, generator=g,
        num_workers=min(4, os.cpu_count() or 1), pin_memory=(DEVICE == "cuda"),
        persistent_workers=True)

    opt = timm.optim.create_optimizer_v2(model, "adamw", lr=LR, weight_decay=WEIGHT_DECAY)
    total = epochs * len(loader)
    warmup = max(1, int(0.05 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda step: (step + 1) / warmup if step < warmup
        else 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup))))
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)
    crit = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    corrupt = CONDITION == "corrupt_aug"

    t0 = time.time()
    for ep in range(epochs):
        model.train()
        run_loss, seen = 0.0, 0
        for xb, yb in loader:
            x = prepare(xb, mean, std, corrupt=corrupt)
            y = yb.to(DEVICE, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=USE_AMP):
                loss = crit(model(x), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            run_loss += loss.item() * y.size(0)
            seen += y.size(0)
        print(f"  seed {seed} | epoch {ep + 1}/{epochs} | loss {run_loss / seen:.4f} | {time.time() - t0:.0f}s")
    return model, mean, std, time.time() - t0


def evaluate_all(model, mean, std, test_base, c10c_dir):
    n = 500 if QUICK_TEST else 10000
    test_x = test_base.data[:n]
    test_y = np.array(test_base.targets[:n])
    t0 = time.time()
    clean = accuracy_on_array(model, mean, std, test_x, test_y)
    print(f"  clean acc: {clean:.4f}")
    labels = np.load(os.path.join(c10c_dir, "labels.npy"))
    res = {}
    for c in CORRUPTIONS:
        arr = np.load(os.path.join(c10c_dir, c + ".npy"), mmap_mode="r")
        accs = []
        for sev in range(1, 6):
            lo = (sev - 1) * 10000
            accs.append(accuracy_on_array(model, mean, std, arr[lo:lo + n], labels[lo:lo + n]))
        res[c] = accs
        print(f"  {c:18s} " + " ".join(f"{a:.3f}" for a in accs))
    return clean, res, time.time() - t0


def relative_drops(result):
    clean = result["clean_acc"]
    drop = {c: (clean - float(np.mean(a))) / clean for c, a in result["corruptions"].items()}
    return {
        "seen": float(np.mean([drop[c] for c in SEEN])),
        "unseen": float(np.mean([drop[c] for c in UNSEEN])),
        "all": float(np.mean([drop[c] for c in CORRUPTIONS])),
        "per_corruption": drop,
    }

# %%
# ---------------- Run ----------------
def run_all():
    train_base, test_base = load_cifar10()
    c10c_dir = find_cifar10c()
    results = []
    for seed in SEEDS:
        print(f"\n=== {MODEL} | {CONDITION} | seed {seed} ===")
        model, mean, std, train_time = train_model(seed, train_base)
        clean, corr, eval_time = evaluate_all(model, mean, std, test_base, c10c_dir)
        r = {
            "model": MODEL, "timm_name": TIMM_NAMES[MODEL], "condition": CONDITION, "seed": seed,
            "epochs": 1 if QUICK_TEST else EPOCHS, "batch_size": BATCH_SIZE, "lr": LR,
            "quick_test": QUICK_TEST, "clean_acc": clean, "corruptions": corr,
            "train_time_s": train_time, "eval_time_s": eval_time,
        }
        r["relative_drop"] = relative_drops(r)
        tag = "QUICK_" if QUICK_TEST else ""
        with open(os.path.join(OUT_DIR, f"{tag}{MODEL}_{CONDITION}_seed{seed}.json"), "w") as f:
            json.dump(r, f, indent=2)   # saved after every seed, so partial results survive a crash
        results.append(r)
        print(f"  relative drop: seen {r['relative_drop']['seen']:.4f} | unseen {r['relative_drop']['unseen']:.4f}")
        if QUICK_TEST:
            full_train = train_time * (50000 / 2000) * EPOCHS
            full_eval = eval_time * (10000 / 500)
            print(f"\n  ESTIMATE for a full run: about {(full_train + full_eval) / 60:.0f} min per seed, "
                  f"{len(SEEDS) * (full_train + full_eval) / 3600:.1f} h for {len(SEEDS)} seeds")
            break
        del model
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    def ms(key):
        v = [r[key] if key == "clean_acc" else r["relative_drop"][key] for r in results]
        return f"{np.mean(v):.4f} ± {np.std(v, ddof=1) if len(v) > 1 else 0:.4f}"
    print(f"\n===== SUMMARY {MODEL} / {CONDITION} ({len(results)} seeds) =====")
    print("clean accuracy        :", ms("clean_acc"))
    print("relative drop (seen)  :", ms("seen"))
    print("relative drop (unseen):", ms("unseen"))
    print("relative drop (all)   :", ms("all"))

run_all()
