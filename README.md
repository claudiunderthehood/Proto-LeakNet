# Proto-LeakNet

Official codebase for the CVIU paper **"Proto-LeakNet: Towards Signal-Leak Aware Attribution in Synthetic and Deepfake Images"**.

This repository contains a prototype-based attribution pipeline for synthetic/deepfake image forensics, with two main workflows:

1. `pipeline/train_eval.py`: closed-set training and evaluation.
2. `open_set.py`: open-set rejection using KDE-based out-of-distribution scoring from a trained checkpoint.

This README is intentionally focused on those two entry points and the code they actually use.

## Overview

Proto-LeakNet learns class-discriminative embeddings that emphasize generator-specific signal leaks rather than semantic image content. The main training pipeline combines:

- a leak-oriented backbone (`proto_leaknet/model.py`)
- a multi-prototype classifier head (`proto_leaknet/prototypes.py`)
- optional SD2.1 latent-space encoding and diffusion-step perturbation
- post-hoc closed-set scoring with metrics such as diagonal Mahalanobis, Proto-GMM, KDE, tied-covariance LLR, or prototype-distance scoring

The companion open-set script loads a trained checkpoint, extracts embeddings for closed and open data, and performs OOD rejection with **kernel density estimation (KDE)**. It also includes optional ablations against Mahalanobis and energy-based scoring.

## Main Entry Points

### `pipeline/train_eval.py`

This is the main script for the paper pipeline.

It handles:

- dataset discovery and split loading
- closed-set training on known source classes
- prototype learning with optional attention
- optional latent-space processing through the Stable Diffusion 2.1 VAE
- optional multi-step diffusion noising over selected timesteps
- metric fitting and evaluation on validation/test splits
- calibration, diagnostics, drift checks, confusion matrices, and ROC plots

In practice, the training flow is:

1. Build train/val/test closed-set splits from `--closed-root`.
2. Encode images either in image space or SD2.1 latent space.
3. Train the backbone + prototype head with cross-entropy over class scores.
4. Evaluate closed-set attribution on the test split.

### `open_set.py`

This script performs open-set evaluation from a checkpoint produced by `train_eval.py`.

It:

- loads the saved backbone checkpoint
- rebuilds the corresponding feature extractor
- embeds closed-set and open-set samples
- fits a KDE density model on closed-set embeddings
- scores closed and open embeddings for OOD rejection
- reports AUROC, EER, FPR@95, overlap, and optional UMAP visualizations

If `--ablation` is enabled, it also compares KDE with:

- Mahalanobis scoring
- energy-based scoring derived from prototype logits

## Repository Scope

Only a small part of the repository is required for the two workflows above.

Core modules used by `train_eval.py`:

- `data/closed_open_dataset.py`: closed-set split handling and transforms
- `proto_leaknet/model.py`: backbone definitions
- `proto_leaknet/prototypes.py`: multi-prototype head
- `heads/metrics_proto.py`: Mahalanobis / GMM utilities and LSE aggregation
- `sd21/vae.py`: SD2.1 VAE loading and latent encoding
- `sd21/noising.py`: forward noising and sigma normalization
- `utils/metrics_auc.py`: closed-set AUC utilities
- `utils/diag.py`: drift summaries
- `proto_leaknet/compressors/dvae.py`: optional DVAE compression

Core modules used by `open_set.py`:

- `proto_leaknet/model.py`
- `proto_leaknet/prototypes.py`
- `heads/metrics_proto.py`
- `scikit-learn` KDE / ROC utilities

Other exploratory or plotting scripts in the repository are not part of the two main workflows and are intentionally not documented here.

## Installation

Create a Python environment and install the dependencies you need for these workflows.

```bash
pip install -r requirements.txt
```

At minimum, the two main scripts rely on:

- `torch`, `torchvision`
- `numpy`, `scikit-learn`, `scipy`
- `matplotlib`, `seaborn`
- `Pillow`, `tqdm`
- `diffusers`, `accelerate` when `--latent-space true` is used

## Dataset Layout

### Closed-set root

`train_eval.py` expects `--closed-root` to contain one subdirectory per class:

```text
closed_root/
  class_a/
    img_001.png
    img_002.png
  class_b/
    img_001.png
    img_002.png
  ...
```

Splits are resolved in one of two ways:

- If `Split.json` exists under `closed_root`, it is used.
- Otherwise the script builds a default per-class split of roughly 70% train, 15% val, 15% test.

If you use the optional `--train-step` filter, the code expects parent directory names such as `step1`, `step2`, or `step3` somewhere in the sample path.

### Open-set root

`open_set.py` accepts either:

- `--open-root /path/to/open_images`
- or `--image-net`, which uses `datasets/ImageNet`

The open root may be:

- a flat directory of images
- or a directory with class subfolders

For open-set scoring, labels are ignored and all open samples are treated as OOD.

## Closed-Set Training

Basic example:

```bash
python pipeline/train_eval.py \
  --closed-root /path/to/closed_root \
  --backbone resnet18 \
  --pretrained true \
  --embed-dim 128 \
  --protos-per-class 6 \
  --metric maha_diag \
  --epochs 10 \
  --batch-size 16 \
  --run-name protoleak_run \
  --output-dir outputs
```

### Important options

- `--backbone`: `conv`, `resnet18`, `resnet50`, `resnet101`, `efficientnet_b4`, `vit`
- `--metric`: `maha_diag`, `proto_gmm`, `kde`, `llr_tied`, `euclidean`
- `--latent-space true|false`: use SD2.1 VAE latents instead of raw RGB
- `--t-steps 0,100,300,...`: diffusion timesteps used for noising
- `--sigma-normalize true|false`: divide noised inputs by `sigma_t`
- `--use-attention true|false`: enable feature-wise attention in the backbone
- `--compressor none|pca|dvae`: optional embedding compression before metric fitting
- `--calib true|false`: enable score calibration
- `--calibration-mode zscore|perclass_temp|none`
- `--tta-flip true|false`: horizontal-flip test-time averaging

### Latent-space mode

When `--latent-space true`, the script:

- loads the Stable Diffusion 2.1 VAE from Hugging Face
- converts images to SD2.1 latent tensors
- applies forward diffusion noise at the requested timesteps
- optionally normalizes by the noise level

This mode requires access to the SD2.1 VAE weights. If the model is gated in your setup, provide the appropriate Hugging Face token through the local `env.py` expected by the script.

## Closed-Set Outputs

`train_eval.py` writes several artifacts:

- checkpoint: `checkpoints/<checkpoint_name>`
- run summary: `<output-dir>/metrics/<run_name>.txt`
- optional silhouette dump: `<output-dir>/metrics/<run_name>_silhouette.json`
- confusion matrices: `plots/matrixes/`
- one-vs-rest ROC curves: `plots/roc_curves/`

The metrics report includes:

- closed-set macro AUC
- top-1 / top-5 accuracy
- balanced accuracy
- per-class AUC
- calibration statistics
- drift diagnostics
- temporal attention summaries when multiple diffusion steps are used
- embedding statistics and label-shuffle sanity checks

## Open-Set KDE Evaluation

Basic example:

```bash
python open_set.py \
  --checkpoint checkpoints/best_model.pt \
  --closed-root /path/to/closed_root \
  --open-root /path/to/open_root \
  --report-dir reports/open_set \
  --plot
```

### What `open_set.py` does

The default evaluation mode is `dual`:

- closed-set embeddings are extracted with attention enabled
- open-set embeddings are extracted with attention disabled

This behavior follows the intended signal-leak-aware separation logic implemented in the script. You can also compare:

- `both_off`
- `both_on`
- `dual`

via `--compare-modes`.

### Important options

- `--bandwidth`: fixed KDE bandwidth
- `--bw-grid`: candidate bandwidths used when `--bandwidth` is omitted
- `--kfold`: K-fold evaluation for closed-set out-of-fold likelihoods
- `--ablation`: compare KDE vs Mahalanobis vs energy
- `--plot`: save UMAP projections
- `--diag`: save per-channel input diagnostics
- `--alpha-policy keep|zero|drop_usepad`: handling of the fourth channel when the checkpoint expects 4-channel inputs
- `--seed-sweep`: repeat evaluation over multiple seeds
- `--alpha-sweep`: evaluate all alpha policies
- `--label-shuffle`: sanity AUROC under shuffled labels
- `--bw-curve`: plot AUROC as a function of bandwidth

## Open-Set Outputs

`open_set.py` writes to `--report-dir`:

- `summary.json`: aggregated metrics
- `*_embeddings.npz`: closed/open embeddings and score arrays
- `*_kde_scores.png`, `*_kde_cdf.png`: KDE score diagnostics
- optional bandwidth curve plots
- optional UMAP plots
- optional ablation visualizations for Mahalanobis and energy

The main reported metrics are:

- AUROC
- EER
- FPR@95
- overlap coefficient between closed/open score distributions

## Notes on Reproducibility

- Random seeds are explicitly set in both main scripts.
- `open_set.py` uses K-fold closed-set scoring to avoid optimistic in-sample KDE estimates.
- `train_eval.py` supports label-shuffle sanity checks and embedding drift diagnostics.
- If CUDA is unavailable, `train_eval.py` falls back to CPU automatically.

## Citation

If you use this repository, please cite the paper:

```bibtex
@article{giusti2025proto,
  title={Proto-LeakNet: Towards Signal-Leak Aware Attribution in Synthetic Human Face Imagery},
  author={Giusti, Claudio and Guarnera, Luca and Battiato, Sebastiano},
  journal={arXiv preprint arXiv:2511.04260},
  year={2025}
}
```
