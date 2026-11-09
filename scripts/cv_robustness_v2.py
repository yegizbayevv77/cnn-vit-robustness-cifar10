# CNN vs ViT robustness under augmentation (CIFAR-10 -> CIFAR-10-C), version 2.
# Shynbergen Yegizbayev, AITU, Computer Vision paper.
# Run by the launcher notebook (run_queue.ipynb); one process = one (MODEL, CONDITION) with 3 seeds.
# The training recipe is identical to version 1; only models and conditions were added.

# ======================= CONFIG (set by the launcher through environment variables) =======================
MODEL = __import__("os").environ.get("CV_MODEL", "resnet50")
# "resnet50", "deit_small", "convnext_tiny", "swin_tiny"
CONDITION = __import__("os").environ.get("CV_CONDITION", "baseline")
# "baseline", "corrupt_aug" (noise+blur), "augmix", "aug_noise", "aug_blur", "aug_digital"
QUICK_TEST = __import__("os").environ.get("CV_QUICK", "0") == "1"
PRETRAINED = __import__("os").environ.get("CV_PRETRAINED", "1") == "1"   # 0 only for offline code tests
NUM_WORKERS = int(__import__("os").environ.get("CV_WORKERS", "2"))
SEEDS = [0, 1, 2]
EPOCHS = 6
BATCH_SIZE = 128
LR = 1e-4
WEIGHT_DECAY = 0.05
LABEL_SMOOTHING = 0.1
IMG_SIZE = 224
# ==========================================================================================================

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
OUT_DIR = os.environ.get("CV_OUT_DIR", "/kaggle/working/results" if ON_KAGGLE else "./results")
DATA_DIR = os.environ.get("CV_DATA_DIR", "/tmp/cvdata")   # not saved as notebook output
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# All models: ImageNet-1k pretraining only, 22-29M parameters, 224x224 input.
# resnet50.a1_in1k uses the "ResNet strikes back" A1 recipe, which is close to the DeiT recipe.
# convnext_tiny (CNN) and swin_tiny (hierarchical ViT) are a second, more modern pair, both trained with DeiT-style recipes.
TIMM_NAMES = {
    "resnet50": "resnet50.a1_in1k",
    "deit_small": "deit_small_patch16_224.fb_in1k",
    "convnext_tiny": "convnext_tiny.fb_in1k",
    "swin_tiny": "swin_tiny_patch4_window7_224.ms_in1k",
}
CONDITIONS = ["baseline", "corrupt_aug", "augmix", "aug_noise", "aug_blur", "aug_digital",
              "nb_p25", "nb_p100", "nb_weak", "nb_strong"]
# Sensitivity conditions (round 3): the noise+blur operations of corrupt_aug with a different probability p
# or with every intensity range multiplied by S. corrupt_aug itself is p = 0.5, S = 1.
AUG_SETTINGS = {"nb_p25": (0.25, 1.0), "nb_p100": (1.0, 1.0), "nb_weak": (0.5, 0.5), "nb_strong": (0.5, 1.5)}
AUG_P, AUG_SCALE = AUG_SETTINGS.get(CONDITION, (0.5, 1.0))
assert MODEL in TIMM_NAMES, MODEL
assert CONDITION in CONDITIONS, CONDITION

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
    sigma = torch.empty(x.size(0), 1, 1, 1, device=x.device).uniform_(0.02 * AUG_SCALE, 0.10 * AUG_SCALE)
    return x + torch.randn_like(x) * sigma

def _shot_noise(x):
    lam = torch.empty(x.size(0), 1, 1, 1, device=x.device).uniform_(50 / AUG_SCALE, 500 / AUG_SCALE)  # smaller rate = more noise
    return torch.poisson(x * lam) / lam

def _impulse_noise(x):
    amount = torch.empty(x.size(0), 1, 1, 1, device=x.device).uniform_(0.01 * AUG_SCALE, 0.07 * AUG_SCALE)
    r = torch.rand_like(x)
    x = torch.where(r < amount / 2, torch.zeros_like(x), x)
    return torch.where(r > 1 - amount / 2, torch.ones_like(x), x)

def _gaussian_blur(x):
    sigma = random.uniform(0.4 * AUG_SCALE, 1.5 * AUG_SCALE)
    return TF.gaussian_blur(x, kernel_size=5 if AUG_SCALE <= 1 else 9, sigma=sigma)

def _motion_blur(x):
    k = random.choice([3, 5, 7])
    k = max(3, 2 * round((k * AUG_SCALE - 1) / 2) + 1)   # odd length scaled by S (unchanged for S = 1)
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

# ---- digital family (added in version 2 for the cross-family transfer experiment) ----
def _contrast(x):
    c = torch.empty(x.size(0), 1, 1, 1, device=x.device).uniform_(0.3, 0.8)
    m = x.mean(dim=(1, 2, 3), keepdim=True)
    return (x - m) * c + m

def _pixelate(x):
    s = random.choice([12, 16, 20, 24])
    small = F.interpolate(x, size=(s, s), mode="area")
    return F.interpolate(small, size=x.shape[-2:], mode="nearest")

def _jpeg(x):
    q = random.randint(15, 60)
    u8 = (x.clamp(0, 1) * 255).round().to(torch.uint8).cpu()
    out = [torchvision.io.decode_jpeg(torchvision.io.encode_jpeg(img, quality=q)) for img in u8]
    return torch.stack(out).to(x.device, x.dtype) / 255

def _elastic(x):
    alpha = random.uniform(10.0, 30.0)
    disp = T.ElasticTransform.get_params([alpha, alpha], [3.0, 3.0], [x.size(-2), x.size(-1)]).to(x.device, x.dtype)
    return TF.elastic_transform(x, disp, interpolation=T.InterpolationMode.BILINEAR)

NOISE_OPS = [_gaussian_noise, _shot_noise, _impulse_noise]
BLUR_OPS = [_gaussian_blur, _motion_blur]
DIGITAL_OPS = [_contrast, _pixelate, _jpeg, _elastic]
OPS_BY_CONDITION = {
    "corrupt_aug": NOISE_OPS + BLUR_OPS,   # version 1 condition, unchanged
    "aug_noise": NOISE_OPS,
    "aug_blur": BLUR_OPS,
    "aug_digital": DIGITAL_OPS,
    **{c: NOISE_OPS + BLUR_OPS for c in AUG_SETTINGS},
}

def corrupt_batch(x, ops, p=None):
    """x: float in [0,1], shape (B,3,32,32). Each image is corrupted with probability p by one random op."""
    p = AUG_P if p is None else p
    apply = torch.rand(x.size(0), device=x.device) < p
    op_id = torch.randint(0, len(ops), (x.size(0),), device=x.device)
    out = x.clone()
    for k, op in enumerate(ops):
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


def build_model(pretrained=PRETRAINED):
    model = timm.create_model(TIMM_NAMES[MODEL], pretrained=pretrained, num_classes=10)
    cfg = model.pretrained_cfg
    mean = torch.tensor(cfg["mean"], device=DEVICE).view(1, 3, 1, 1)
    std = torch.tensor(cfg["std"], device=DEVICE).view(1, 3, 1, 1)
    model = model.to(DEVICE).to(memory_format=torch.channels_last)
    return model, mean, std


def prepare(x_uint8, mean, std, ops=None):
    x = x_uint8.to(DEVICE, non_blocking=True).float().div_(255)
    if ops:
        x = corrupt_batch(x, ops)
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


def train_model(seed, train_base):
    set_seed(seed)
    model, mean, std = build_model()
    n_train = 2000 if QUICK_TEST else len(train_base)
    epochs = 1 if QUICK_TEST else EPOCHS
    g = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(train_base), generator=g)[:n_train].tolist()
    ds = TrainSet(train_base, CONDITION, indices)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, generator=g,
        num_workers=NUM_WORKERS, pin_memory=(DEVICE == "cuda"),
        persistent_workers=NUM_WORKERS > 0)

    opt = timm.optim.create_optimizer_v2(model, "adamw", lr=LR, weight_decay=WEIGHT_DECAY)
    total = epochs * len(loader)
    warmup = max(1, int(0.05 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda step: (step + 1) / warmup if step < warmup
        else 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup))))
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)
    crit = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    ops = OPS_BY_CONDITION.get(CONDITION)

    t0 = time.time()
    for ep in range(epochs):
        model.train()
        run_loss, seen = 0.0, 0
        for xb, yb in loader:
            x = prepare(xb, mean, std, ops=ops)
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
            "quick_test": QUICK_TEST, "pretrained": PRETRAINED, "version": 2,
            "aug_ops": [op.__name__.strip("_") for op in OPS_BY_CONDITION.get(CONDITION, [])],
            "aug_p": AUG_P if CONDITION in OPS_BY_CONDITION else None, "aug_scale": AUG_SCALE,
            "clean_acc": clean, "corruptions": corr,
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


def prepare_only(models):
    """Run once before parallel jobs: put datasets and pretrained weights in place, so the jobs do not race."""
    load_cifar10()
    print("CIFAR-10-C at", find_cifar10c())
    for m in models:
        timm.create_model(TIMM_NAMES[m], pretrained=PRETRAINED, num_classes=10)
        print("weights ready:", TIMM_NAMES[m])


if __name__ == "__main__":
    if os.environ.get("CV_PREPARE_MODELS"):
        prepare_only(os.environ["CV_PREPARE_MODELS"].split(","))
    else:
        run_all()
