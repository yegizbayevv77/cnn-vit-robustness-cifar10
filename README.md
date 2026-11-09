# CNN vs ViT corruption robustness under data augmentation (CIFAR-10 → CIFAR-10-C)

Code and raw results for the paper *"How Far Does Data Augmentation Narrow the Corruption-Robustness Gap Between Pretrained CNNs and Vision Transformers? A Fine-Tuning Study with Four ImageNet-1k Models on CIFAR-10-C"* (Shynbergen Yegizbayev, Astana IT University).

## What is here

| Path | Content |
|---|---|
| `notebooks/round1_single_run.ipynb` | First round: one (model, condition) per run, 3 seeds. Used for ResNet-50 and DeiT-S × {baseline, corrupt_aug, augmix}. |
| `notebooks/run_A_new_models.ipynb` | Second round, part A: ConvNeXt-T and Swin-T × {baseline, corrupt_aug}. |
| `notebooks/run_B_transfer.ipynb` | Second round, part B: ResNet-50 and DeiT-S trained with one corruption family each (noise, blur, digital). |
| `notebooks/run_C_probability.ipynb` | Sensitivity, part C: noise+blur augmentation with p = 0.25 and p = 1.0 (ResNet-50, DeiT-S). |
| `notebooks/run_D_intensity.ipynb` | Sensitivity, part D: noise+blur augmentation with all intensity ranges ×0.5 and ×1.5 (ResNet-50, DeiT-S). |
| `scripts/cv_robustness.py` | Round-1 experiment script (same code as the round-1 notebook). |
| `scripts/cv_robustness_v2.py` | Round-2 experiment script (same recipe; adds models and single-family conditions). |
| `scripts/analyze.py` | Main tables and figures for ResNet-50 / DeiT-S. |
| `scripts/analyze_v2.py` | Confidence intervals, four-model comparison, transfer matrix, clean-vs-robust plot. |
| `results/*.json` | 72 files, one per (model, condition, seed): clean accuracy and accuracy for each of the 15 CIFAR-10-C corruptions at 5 severities. |
| `analysis/` | Summary tables (`results_summary.md`, `results_summary_v2.md`) and the figures used in the paper. |

## Setup

- Models (from `timm`, ImageNet-1k only): `resnet50.a1_in1k`, `convnext_tiny.fb_in1k`, `deit_small_patch16_224.fb_in1k`, `swin_tiny_patch4_window7_224.ms_in1k`.
- Fine-tuning (identical for all runs): AdamW, lr 1e-4, weight decay 0.05, batch 128, 6 epochs, 5% warm-up + cosine, label smoothing 0.1, AMP, 32×32 images resized to 224×224 (bilinear). Seeds 0, 1, 2.
- Conditions: `baseline` (crop + flip), `corrupt_aug` (noise + blur ops, p = 0.5), `augmix` (torchvision AugMix, restricted ops, no JSD loss), `aug_noise`, `aug_blur`, `aug_digital` (one family each, p = 0.5); `nb_p25`, `nb_p100`, `nb_weak`, `nb_strong` (noise+blur with p = 0.25 / 1.0, or intensity ×0.5 / ×1.5).
- Hardware: Kaggle, NVIDIA Tesla T4 (round 2 runs two jobs in parallel on T4 ×2).

## Reproduce

1. On Kaggle, create a notebook with **GPU T4 x2** and **Internet on**, and attach the datasets *CIFAR-10 Python* (pankrzysiu/cifar10-python) and *CIFAR10-C* (faigarvn/cifar10-c). Without them the code downloads CIFAR-10 and CIFAR-10-C (Zenodo) itself.
2. Import one of the notebooks and run it (Save & Run All). Results are written to `results/*.json`.
3. Analysis, locally:
   ```bash
   pip install numpy scipy matplotlib
   python scripts/analyze.py results analysis
   python scripts/analyze_v2.py results analysis
   ```

## Data and licences

CIFAR-10 (Krizhevsky, 2009) and CIFAR-10-C (Hendrycks & Dietterich, ICLR 2019) are public datasets and are not redistributed here. Pretrained weights are downloaded from the Hugging Face Hub through `timm`.
