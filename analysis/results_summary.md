## Table 1. Main results (mean ± std over seeds)

| Model | Training | n | Clean acc. (%) | Rel. drop seen (%) | Rel. drop unseen (%) | Rel. drop all (%) |
|---|---|---|---|---|---|---|
| ResNet-50 | Baseline | 3 | 96.46 ± 0.09 | 32.25 ± 1.32 | 11.98 ± 0.04 | 21.44 ± 0.63 |
| DeiT-S | Baseline | 3 | 98.04 ± 0.06 | 11.99 ± 0.52 | 4.94 ± 0.09 | 8.23 ± 0.29 |
| ResNet-50 | Corruption aug. | 3 | 95.74 ± 0.13 | 8.38 ± 0.16 | 8.97 ± 0.14 | 8.70 ± 0.06 |
| DeiT-S | Corruption aug. | 3 | 98.13 ± 0.07 | 4.02 ± 0.12 | 3.52 ± 0.10 | 3.75 ± 0.11 |
| ResNet-50 | AugMix | 3 | 96.21 ± 0.04 | 20.35 ± 0.61 | 9.88 ± 0.03 | 14.76 ± 0.29 |
| DeiT-S | AugMix | 3 | 98.23 ± 0.02 | 7.11 ± 0.51 | 3.90 ± 0.04 | 5.40 ± 0.24 |

## Table 2. Robustness gap = ViT drop − CNN drop (pp; ± is SE of difference); Ratio = CNN drop / ViT drop

| Training | Gap seen (pp) | p (Welch) | Ratio seen | Gap unseen (pp) | p (Welch) | Ratio unseen |
|---|---|---|---|---|---|---|
| Baseline | -20.26 ± 0.82 | 0.000366 | 2.69 | -7.04 ± 0.06 | 6.19e-06 | 2.42 |
| Corruption aug. | -4.36 ± 0.12 | 7.17e-06 | 2.09 | -5.46 ± 0.10 | 1.95e-06 | 2.55 |
| AugMix | -13.24 ± 0.46 | 1.13e-05 | 2.86 | -5.98 ± 0.03 | 2.44e-08 | 2.54 |

## Table 3. Change relative to baseline (percentage points)

| Model | Training | Δ clean (pp) | Δ seen drop (pp) | Δ unseen drop (pp) |
|---|---|---|---|---|
| ResNet-50 | Corruption aug. | -0.72 | -23.87 | -3.01 |
| ResNet-50 | AugMix | -0.25 | -11.91 | -2.10 |
| DeiT-S | Corruption aug. | +0.09 | -7.97 | -1.43 |
| DeiT-S | AugMix | +0.19 | -4.88 | -1.05 |

## Table 4. Relative drop per corruption (%, mean over seeds and severities)

| Corruption | Group | ResNet-50 Baseline | DeiT-S Baseline | ResNet-50 Corruption aug. | DeiT-S Corruption aug. | ResNet-50 AugMix | DeiT-S AugMix |
|---|---|---|---|---|---|---|---|
| gaussian_noise | seen | 56.8 | 19.4 | 9.3 | 4.5 | 40.1 | 13.0 |
| shot_noise | seen | 42.8 | 15.0 | 6.7 | 3.2 | 28.7 | 9.5 |
| impulse_noise | seen | 48.6 | 12.3 | 4.5 | 2.1 | 21.8 | 3.9 |
| defocus_blur | seen | 8.3 | 3.4 | 2.5 | 1.1 | 3.6 | 1.6 |
| glass_blur | seen | 40.6 | 22.6 | 26.9 | 13.9 | 35.2 | 16.7 |
| motion_blur | seen | 17.3 | 7.0 | 5.5 | 2.0 | 8.1 | 3.1 |
| zoom_blur | seen | 11.4 | 4.3 | 3.3 | 1.4 | 4.9 | 2.0 |
| snow | unseen | 6.5 | 3.4 | 5.5 | 2.6 | 6.3 | 2.9 |
| frost | unseen | 8.0 | 3.9 | 5.2 | 2.5 | 7.4 | 3.4 |
| fog | unseen | 4.6 | 3.6 | 3.0 | 2.4 | 3.1 | 2.4 |
| brightness | unseen | 1.4 | 0.7 | 1.2 | 0.6 | 1.2 | 0.6 |
| contrast | unseen | 10.4 | 3.0 | 7.5 | 1.7 | 8.3 | 1.7 |
| elastic_transform | unseen | 11.8 | 5.6 | 7.7 | 3.7 | 9.0 | 4.3 |
| pixelate | unseen | 33.2 | 9.2 | 27.1 | 6.3 | 28.1 | 6.8 |
| jpeg_compression | unseen | 19.9 | 10.2 | 14.6 | 8.4 | 15.6 | 9.0 |
