## Gap ResNet-50 vs DeiT-S (ViT - CNN, pp, 95% Welch CI), ratio CNN/ViT

| Training | seen gap | p | ratio | unseen gap | p | ratio |
|---|---|---|---|---|---|---|
| Baseline | -20.26 [-23.10, -17.42] | 0.00037 | 2.69 | -7.04 [-7.24, -6.83] | 6.2e-06 | 2.42 |
| Noise+blur aug. | -4.36 [-4.70, -4.03] | 7.2e-06 | 2.09 | -5.46 [-5.74, -5.17] | 2e-06 | 2.55 | (abs. gap reduced 78% seen, 22% unseen)
| AugMix | -13.24 [-14.52, -11.95] | 1.1e-05 | 2.86 | -5.98 [-6.07, -5.90] | 2.4e-08 | 2.54 | (abs. gap reduced 35% seen, 15% unseen)

## Four models (mean ± std over seeds)

| Training | Model | clean | seen | unseen |
|---|---|---|---|---|
| Baseline | ResNet-50 | 96.46 ± 0.09 | 32.25 ± 1.32 | 11.98 ± 0.04 |
| Baseline | ConvNeXt-T | 98.16 ± 0.07 | 15.55 ± 0.30 | 7.68 ± 0.09 |
| Baseline | DeiT-S | 98.04 ± 0.06 | 11.99 ± 0.52 | 4.94 ± 0.09 |
| Baseline | Swin-T | 98.13 ± 0.12 | 18.25 ± 0.19 | 7.12 ± 0.08 |
| Noise+blur aug. | ResNet-50 | 95.74 ± 0.13 | 8.38 ± 0.16 | 8.97 ± 0.14 |
| Noise+blur aug. | ConvNeXt-T | 98.12 ± 0.12 | 4.43 ± 0.03 | 4.86 ± 0.19 |
| Noise+blur aug. | DeiT-S | 98.13 ± 0.07 | 4.02 ± 0.12 | 3.52 ± 0.10 |
| Noise+blur aug. | Swin-T | 98.01 ± 0.03 | 4.77 ± 0.08 | 4.89 ± 0.26 |

## All CNN-ViT pairings

| Training | pair | seen gap [CI] | ratio | unseen gap [CI] | ratio |
|---|---|---|---|---|---|
| Baseline | ResNet-50-DeiT-S | -20.26 [-23.10, -17.42] | 2.69 | -7.04 [-7.24, -6.83] | 2.42 |
| Baseline | ResNet-50-Swin-T | -14.00 [-17.19, -10.82] | 1.77 | -4.86 [-5.04, -4.69] | 1.68 |
| Baseline | ConvNeXt-T-DeiT-S | -3.56 [-4.62, -2.49] | 1.30 | -2.73 [-2.95, -2.52] | 1.55 |
| Baseline | ConvNeXt-T-Swin-T | +2.70 [+2.09, +3.31] | 0.85 | -0.56 [-0.76, -0.35] | 1.08 |
| Noise+blur aug. | ResNet-50-DeiT-S | -4.36 [-4.70, -4.03] | 2.09 | -5.46 [-5.74, -5.17] | 2.55 |
| Noise+blur aug. | ResNet-50-Swin-T | -3.61 [-3.95, -3.27] | 1.76 | -4.08 [-4.61, -3.55] | 1.83 |
| Noise+blur aug. | ConvNeXt-T-DeiT-S | -0.41 [-0.69, -0.13] | 1.10 | -1.34 [-1.74, -0.94] | 1.38 |
| Noise+blur aug. | ConvNeXt-T-Swin-T | +0.34 [+0.17, +0.51] | 0.93 | +0.04 [-0.49, +0.57] | 0.99 |

## Within-family vs between-family spread (baseline, pp of relative drop)

- seen: CNNs ResNet-50 32.25, ConvNeXt-T 15.55 | ViTs DeiT-S 11.99, Swin-T 18.25 | within-CNN diff 16.70, within-ViT diff 6.26, worst CNN - best ViT 20.26, best CNN - worst ViT -2.70
- unseen: CNNs ResNet-50 11.98, ConvNeXt-T 7.68 | ViTs DeiT-S 4.94, Swin-T 7.12 | within-CNN diff 4.30, within-ViT diff 2.18, worst CNN - best ViT 7.04, best CNN - worst ViT 0.56

## Transfer matrix: relative drop (%) per test family; R=ResNet-50, D=DeiT-S; * = family used in training

| Training | noise R | noise D | noise ratio | blur R | blur D | blur ratio | weather R | weather D | weather ratio | digital R | digital D | digital ratio |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Baseline | 49.4 | 15.6 | 3.17 | 19.4 | 9.3 | 2.08 | 5.1 | 2.9 | 1.77 | 18.8 | 7.0 | 2.70 |
| Noise aug. | 6.0* | 2.9* | 2.07 | 15.7 | 7.2 | 2.18 | 5.1 | 2.4 | 2.08 | 16.4 | 5.7 | 2.86 |
| Blur aug. | 50.1 | 13.5 | 3.71 | 10.2* | 4.7* | 2.18 | 4.5 | 2.1 | 2.14 | 14.9 | 5.1 | 2.96 |
| Noise+blur aug. | 6.8* | 3.3* | 2.09 | 9.5* | 4.6* | 2.08 | 3.7 | 2.0 | 1.83 | 14.3 | 5.0 | 2.84 |
| Digital aug. | 44.3 | 13.7 | 3.23 | 11.9 | 4.6 | 2.58 | 4.3 | 2.4 | 1.78 | 6.8* | 2.7* | 2.48 |

## Held-out analysis (absolute gap reduction vs baseline and ratio, trained family vs mean of the other families)

- Noise aug.: trained gap -33.83 -> -3.13 pp (91% red.), ratio 3.17 -> 2.07; held-out gap -8.05 -> -7.27 pp (10% red.), ratio 2.26 -> 2.42
- Blur aug.: trained gap -10.08 -> -5.54 pp (45% red.), ratio 2.08 -> 2.18; held-out gap -15.97 -> -16.28 pp (-2% red.), ratio 2.88 -> 3.37
- Noise+blur aug.: trained gap -21.96 -> -4.26 pp (81% red.), ratio 2.76 -> 2.09; held-out gap -7.04 -> -5.46 pp (22% red.), ratio 2.42 -> 2.55
- Digital aug.: trained gap -11.84 -> -4.05 pp (66% red.), ratio 2.70 -> 2.48; held-out gap -15.38 -> -13.24 pp (14% red.), ratio 2.66 -> 2.92

## Sensitivity of noise+blur aug. (p = probability, S = intensity multiplier); R = ResNet-50, D = DeiT-S

| varied | value | clean R | clean D | seen R | seen D | unseen R | unseen D | seen gap [CI] | seen ratio | unseen gap [CI] | unseen ratio |
|---|---|---|---|---|---|---|---|---|---|---|---|
| p | 0.25 | 96.15 | 98.21 | 9.64 | 4.45 | 9.63 | 3.76 | -5.20 [-5.57, -4.82] (74% red.) | 2.17 | -5.86 [-6.14, -5.58] (17% red.) | 2.56 |
| p | 0.5 | 95.74 | 98.13 | 8.38 | 4.02 | 8.97 | 3.52 | -4.36 [-4.70, -4.03] (78% red.) | 2.09 | -5.46 [-5.74, -5.17] (22% red.) | 2.55 |
| p | 1 | 94.70 | 97.96 | 7.28 | 3.44 | 8.95 | 3.44 | -3.84 [-4.05, -3.63] (81% red.) | 2.12 | -5.51 [-5.97, -5.05] (22% red.) | 2.60 |
| S | 0.5 | 96.18 | 98.15 | 10.29 | 4.98 | 9.74 | 4.03 | -5.31 [-6.04, -4.58] (74% red.) | 2.07 | -5.71 [-6.11, -5.32] (19% red.) | 2.42 |
| S | 1 | 95.74 | 98.13 | 8.38 | 4.02 | 8.97 | 3.52 | -4.36 [-4.70, -4.03] (78% red.) | 2.09 | -5.46 [-5.74, -5.17] (22% red.) | 2.55 |
| S | 1.5 | 95.52 | 98.13 | 8.41 | 3.79 | 9.00 | 3.46 | -4.62 [-5.05, -4.20] (77% red.) | 2.22 | -5.54 [-6.02, -5.06] (21% red.) | 2.60 |
