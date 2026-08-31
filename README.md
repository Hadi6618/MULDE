# MULDE — Appearance Stream for the PRISM Framework

> This repository contains the **appearance-based** subcomponent of
> [PRISM](https://github.com/Hadi6618/PRISM) (Pose + RGB Integration for
> Scene Monitoring), a two-stream late-fusion framework for Video Anomaly
> Detection.

MULDE learns a **multiscale density model** of normal frame appearance using
Denoising Score Matching (DSM). Frame features are extracted with a
Hiera-L video backbone, and a small MLP is trained to estimate the gradient
of the log-density of the normal-data distribution at multiple noise scales.
At evaluation time, the score network is queried at 16 fixed noise levels,
producing a 16-dimensional log-density signature per frame. A Gaussian
Mixture Model fitted on the *training* signatures turns each test signature
into a scalar negative log-likelihood — the anomaly score.

> **Paper:** *MULDE: Multiscale Log-Density Estimation via Denoising Score
> Matching for Video Anomaly Detection*, CVPR 2024 —
> [PDF](https://openaccess.thecvf.com/content/CVPR2024/papers/Micorek_MULDE_Multiscale_Log-Density_Estimation_via_Denoising_Score_Matching_for_Video_CVPR_2024_paper.pdf) ·
> [original code](https://github.com/jmicorek/mulde)

---

## Role in PRISM

MULDE is the **appearance stream** of the PRISM ensemble. It detects contextual
anomalies — vehicles on sidewalks, bicycles in pedestrian zones, objects that
should not be in the scene — by modelling the distribution of *how the scene
normally looks* and flagging frames that deviate.

| Component | Repository | What It Watches |
| :-- | :-- | :-- |
| STG-NF (pose) | [Hadi6618/STG-NF](https://github.com/Hadi6618/STG-NF) | What people are *doing* |
| **MULDE (appearance)** | **This repo** | **What the scene *looks like*** |
| Fusion pipeline | [Hadi6618/PRISM](https://github.com/Hadi6618/PRISM) | Combines both streams |

Training of the MULDE density model (DSM + GMM fitting) is orchestrated from
the PRISM repository's [`MULDE.ipynb`](https://github.com/Hadi6618/PRISM/blob/main/MULDE.ipynb)
Colab notebook, which clones this repo, extracts features, trains the MLP, and
exports a score pickle for the fusion stage. This repository retains only the
inference-time model code and the end-to-end test script.

---

## Repository Layout

```
MULDE/
├── models.py                          # MLP + ScoreOrLogDensityNetwork (DSM architecture)
├── MULDE_test.py                      # End-to-end inference on a single custom MP4 video
├── Hiera_L_Feature_Extraction.ipynb   # Colab: extract 1152-D Hiera-L features from
│                                      #   ShanghaiTech / Avenue frames
└── README.md
```

### `models.py`

Defines the two model classes used during inference:

- **`MLPs`** — A 2-layer MLP (4096 hidden units, SiLU activation) that maps
  1152-dimensional Hiera-L frame features to the score (gradient of
  log-density) at a given noise scale. Trained via Denoising Score Matching.
- **`ScoreOrLogDensityNetwork`** — Wraps an `MLPs` backbone and evaluates it
  at *L = 16* logarithmically-spaced noise scales
  (σ ∈ [10⁻³, 10⁰]), producing a 16-dimensional log-density signature per
  frame.

### `MULDE_test.py`

A self-contained CLI script that runs the **full MULDE inference pipeline**
on a single custom `.mp4` video file:

1. Loads the pretrained Hiera-L model from PyTorch Hub (head set to Identity).
2. Decodes and preprocesses video frames (falls back to OpenCV if decord fails).
3. Extracts spatiotemporal features (1152-dim) in batches.
4. Standardizes features using training statistics (mean / std).
5. Computes the 16-dimensional multiscale log-density signature via the trained MLP.
6. Scores the signatures using the GMM to produce raw log-likelihood scores.
7. Applies temporal Gaussian smoothing (σ = 15.0).
8. Classifies frames as normal / anomaly via an adaptive threshold.
9. Detects anomaly time segments (frame → seconds using video FPS).
10. Saves a multi-panel dashboard, per-frame CSV, interval table, and JSON summary.

```bash
python MULDE_test.py \
    --video path/to/video.mp4 \
    --mlp_weights path/to/mlp.pth \
    --gmm_model path/to/gmm.joblib \
    --mean_std path/to/mean_std.npz \
    --output_dir results/
```

> **Note:** `MULDE_test.py` imports `visualization.py` from the PRISM
> repository. On Google Colab, ensure PRISM is cloned to `/content/PRISM`
> before running, or set `PYTHONPATH` accordingly.

### `Hiera_L_Feature_Extraction.ipynb`

A Colab notebook that extracts 1152-dimensional Hiera-L features from
ShanghaiTech Campus or Avenue video frames. The extracted features are saved
as `.npz` files and used as input for DSM training (done in
[`MULDE.ipynb`](https://github.com/Hadi6618/PRISM/blob/main/MULDE.ipynb) in the
PRISM repository).

---

## Results (within PRISM)

| Method | Stream | ShanghaiTech (Micro AUC) | Avenue (Micro AUC) |
| :-- | :-- | ---: | ---: |
| **MULDE** | Appearance | **79.7%** | **81.4%** |
| STG-NF | Pose | 83.9% | 57.0% |
| PRISM (fusion) | Both | 89.9% | 83% |

When fused with the pose stream in PRISM, the combined system achieves
**89.9% Micro AUC on ShanghaiTech** (+10.2 pp over MULDE alone) and
**83% on Avenue** (+1.6 pp). The appearance stream is particularly
important on Avenue where the pose stream underperforms.

---

## Citation

```bibtex
@inproceedings{micorek24mulde,
  title     = {{MULDE: Multiscale Log-Density Estimation via Denoising Score
               Matching for Video Anomaly Detection}},
  author    = {Micorek, Jakub and Possegger, Horst and Narnhofer, Dominik
               and Bischof, Horst and Kozi{\'n}ski, Mateusz},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and
               Pattern Recognition (CVPR)},
  month     = {June},
  year      = {2024},
  pages     = {18868-18877}
}

@misc{prism2026,
  title   = {PRISM: Pose + RGB Integration for Scene Monitoring},
  author  = {Hadi},
  year    = {2026},
  note    = {Late fusion of STG-NF and MULDE for video anomaly detection},
  url     = {https://github.com/Hadi6618/PRISM}
}
```

## License

This repository contains experiment code for a graduate research project.
The underlying MULDE method is the work of its respective authors; please
respect their license when reusing those components.
