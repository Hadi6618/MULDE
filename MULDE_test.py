#!/usr/bin/env python
"""
Run MULDE Anomaly Detection on a Custom MP4 Video
-------------------------------------------------
This script runs the entire MULDE inference pipeline on a single custom .mp4 video file:
1. Loads the pretrained Hiera-L model from PyTorch Hub (head set to Identity).
2. Decodes and preprocesses the video frames (falls back to OpenCV if decord fails).
3. Extracts spatiotemporal features (1152-dim) in batches.
4. Standardizes features using training stats (mean/std).
5. Computes the 16-dimensional multiscale log-density signature using the trained MLP.
6. Scores the signatures using the GMM to compute raw log-likelihood scores.
7. Applies temporal Gaussian smoothing (sigma=15.0).
8. Classifies frames as normal/anomaly via an adaptive threshold on smoothed scores.
9. Detects anomaly time segments (frame → seconds using the video FPS).
10. Saves a multi-panel dashboard, per-frame CSV, interval table, and JSON summary.
"""

import os
import sys
import gc
import argparse
import subprocess
import numpy as np
from pathlib import Path
import torch
import cv2
import joblib
from decord import VideoReader, cpu
from scipy.ndimage import gaussian_filter1d

# visualization.py lives in the PRISM repo (sibling on Colab: /content/PRISM).
# Add it to sys.path so the import resolves regardless of where this script runs from.
_PRISM_DIR = Path("/content/PRISM")
if _PRISM_DIR.is_dir() and str(_PRISM_DIR) not in sys.path:
    sys.path.insert(0, str(_PRISM_DIR))

from visualization import (
    build_results_dataframe,
    generate_anomaly_dashboard,
    parse_frame_ranges,
    print_anomaly_report,
    save_anomaly_artifacts,
)

# Inject official MULDE repo path if available to import architectures
OFFICIAL_REPO_URL = "https://github.com/jakubmicorek/MULDE-Multiscale-Log-Density-Estimation-via-Denoising-Score-Matching-for-Video-Anomaly-Detection.git"
OFFICIAL_REPO_DIR = Path("/content/MULDE_official")
# Optional reproducibility pin. Leave as None to use the repository default branch.
OFFICIAL_REPO_COMMIT = None

if not OFFICIAL_REPO_DIR.exists():
    subprocess.run(["git", "clone", OFFICIAL_REPO_URL, str(OFFICIAL_REPO_DIR)], check=True)
else:
    print(f"Repository already exists: {OFFICIAL_REPO_DIR}")
from models import MLPs, ScoreOrLogDensityNetwork


def load_hiera_extractor(device):
    print("Loading Hiera-L model (hiera_large_16x224, checkpoint=mae_k400_ft_k400) from PyTorch Hub...")
    model = torch.hub.load(
        "facebookresearch/hiera",
        model="hiera_large_16x224",
        pretrained=True,
        checkpoint="mae_k400_ft_k400"
    )
    model.head = torch.nn.Identity()  # Replace classifier head with Identity
    model = model.to(device)
    model.eval()
    return model


def preprocess_all_frames_decord(vr, num_frames, target_size=(224, 224), chunk_size=128):
    """Decode and normalize all video frames exactly once using decord."""
    mean = np.array([0.45, 0.45, 0.45], dtype=np.float32).reshape(1, 3, 1, 1)
    std  = np.array([0.225, 0.225, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)
    all_frames = np.empty((num_frames, 3, target_size[0], target_size[1]), dtype=np.float32)

    for start in range(0, num_frames, chunk_size):
        end = min(start + chunk_size, num_frames)
        indices = list(range(start, end))
        frames_np = vr.get_batch(indices).asnumpy()  # [chunk, H, W, C] RGB uint8

        for j, img in enumerate(frames_np):
            img_resized = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
            img_float = img_resized.astype(np.float32) / 255.0
            all_frames[start + j] = img_float.transpose(2, 0, 1)  # HWC -> CHW

    all_frames = (all_frames - mean) / std
    return all_frames


def preprocess_all_frames_opencv(video_path, num_frames, target_size=(224, 224)):
    """Decode and normalize all video frames exactly once using OpenCV."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV failed to open video: {video_path}")

    mean = np.array([0.45, 0.45, 0.45], dtype=np.float32).reshape(1, 3, 1, 1)
    std  = np.array([0.225, 0.225, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)
    all_frames = np.empty((num_frames, 3, target_size[0], target_size[1]), dtype=np.float32)

    for idx in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            print(f"[WARNING] OpenCV read failed at frame {idx}. Truncating video to {idx} frames.")
            all_frames = all_frames[:idx]
            num_frames = idx
            break
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, target_size, interpolation=cv2.INTER_LINEAR)
        img_float = img_resized.astype(np.float32) / 255.0
        all_frames[idx] = img_float.transpose(2, 0, 1)  # HWC -> CHW

    cap.release()
    all_frames = (all_frames - mean) / std
    return all_frames, num_frames


def generate_clip_indices(i, num_frames):
    """Sample 16 frames with stride 4, centered around target frame i."""
    indices = []
    for k in range(16):
        idx = i - 30 + 4 * k
        idx = max(0, min(idx, num_frames - 1))
        indices.append(idx)
    return np.array(indices, dtype=np.int64)


def extract_hiera_features(video_path, model, device, batch_size=8):
    """Run Hiera-L batched inference to extract 1152-dim features."""
    print(f"Decoding video: {video_path}...")
    
    use_decord = True
    try:
        vr = VideoReader(video_path, ctx=cpu(0))
        num_frames = len(vr)
        fps = vr.get_avg_fps()
    except Exception as e:
        print(f"[WARNING] decord failed to open video: {e}. Falling back to OpenCV...")
        use_decord = False

    if use_decord:
        print(f"Pre-caching {num_frames} frames (using decord)...")
        cached_frames = preprocess_all_frames_decord(vr, num_frames)
        del vr
    else:
        # Fallback to OpenCV
        cap_temp = cv2.VideoCapture(video_path)
        if not cap_temp.isOpened():
            raise RuntimeError(f"OpenCV also failed to read video: {video_path}")
        num_frames = int(cap_temp.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap_temp.get(cv2.CAP_PROP_FPS)
        cap_temp.release()
        
        print(f"Pre-caching {num_frames} frames (using OpenCV)...")
        cached_frames, num_frames = preprocess_all_frames_opencv(video_path, num_frames)
        
    gc.collect()

    print("Running Hiera-L batched feature extraction...")
    frame_indices = np.arange(num_frames, dtype=np.int64)
    clip_indices = np.zeros((num_frames, 16), dtype=np.int64)
    for i in range(num_frames):
        clip_indices[i] = generate_clip_indices(i, num_frames)

    features_list = []
    current_batch_size = batch_size
    success = False

    while not success and current_batch_size >= 1:
        try:
            features_list = []
            for batch_start in range(0, num_frames, current_batch_size):
                batch_end = min(batch_start + current_batch_size, num_frames)
                batch_clips = []
                for i in range(batch_start, batch_end):
                    clip_frames = cached_frames[clip_indices[i]]  # [16, 3, 224, 224]
                    clip_tensor = torch.from_numpy(clip_frames.copy()).permute(1, 0, 2, 3)
                    batch_clips.append(clip_tensor)

                stacked = torch.stack(batch_clips, dim=0).to(device)
                with torch.no_grad():
                    with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                        feats = model(stacked)
                    features_list.append(feats.float().cpu().numpy())
                del stacked
            features = np.concatenate(features_list, axis=0)
            success = True
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"[WARNING] OOM with batch={current_batch_size}. Halving and retrying...")
                current_batch_size //= 2
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if current_batch_size < 1:
                    raise e
            else:
                raise e

    del cached_frames
    gc.collect()
    return features, num_frames, fps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True, help="Path to input .mp4 video file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained MULDE neural network (.pt)")
    parser.add_argument("--stats", type=str, required=True, help="Path to training train_feature_stats.npz")
    parser.add_argument("--gmm", type=str, required=True, help="Path to trained GMM model (.joblib)")
    parser.add_argument("--output_dir", type=str, default="output_results", help="Directory to save output files")
    parser.add_argument("--smooth_sigma", type=float, default=15.0, help="Sigma for temporal Gaussian smoothing")
    parser.add_argument(
        "--shading",
        type=str,
        default=None,
        help="Optional known-anomaly frame ranges to highlight in yellow (e.g. '50-200,340-440')",
    )
    parser.add_argument(
        "--threshold_method",
        type=str,
        default="mad",
        choices=["mad", "percentile", "manual"],
        help="How to set the anomaly threshold on smoothed scores (higher = more anomalous)",
    )
    parser.add_argument(
        "--threshold_percentile",
        type=float,
        default=90.0,
        help="Percentile for threshold_method=percentile (frames above this score are anomalous)",
    )
    parser.add_argument(
        "--threshold_mad_k",
        type=float,
        default=3.0,
        help="MAD multiplier for threshold_method=mad",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Fixed anomaly score threshold when threshold_method=manual",
    )
    parser.add_argument(
        "--min_segment_sec",
        type=float,
        default=0.4,
        help="Minimum contiguous anomaly duration to report as a segment",
    )
    parser.add_argument(
        "--merge_gap_sec",
        type=float,
        default=0.25,
        help="Merge anomaly segments separated by gaps shorter than this (seconds)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Hiera Feature Extraction
    hiera_model = load_hiera_extractor(device)
    features, num_frames, fps = extract_hiera_features(args.video, hiera_model, device)
    print(f"Features extracted: {features.shape} at {fps:.2f} FPS")

    # 2. Standardization
    print(f"Loading feature standardization stats from: {args.stats}")
    stats = np.load(args.stats)
    train_mean = stats["mean"].astype(np.float32)
    train_std = stats["std"].astype(np.float32)
    train_std = np.where(train_std < 1e-8, 1.0, train_std)  # Avoid div-by-zero
    features_std = (features - train_mean) / train_std

    # 3. Load MULDE Network & Compute Signatures
    print(f"Loading MULDE network from checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    
    # Recreate the MLP/Log-Density model matching training configuration
    mulde_net = ScoreOrLogDensityNetwork(
        MLPs(
            input_dim=1152 + 1,
            output_dim=1,
            units=[4096, 4096]
        ),
        score_network=False
    ).to(device)
    
    # Load state dict
    if "model_state_dict" in checkpoint:
        mulde_net.load_state_dict(checkpoint["model_state_dict"])
    else:
        mulde_net.load_state_dict(checkpoint)
    mulde_net.eval()

    # Generate multiscale log-density signatures (L=16)
    sigma_levels = np.linspace(1e-3, 1.0, 16, dtype=np.float32)
    signatures = np.empty((num_frames, 16), dtype=np.float32)

    print("Computing multiscale log-density signatures...")
    with torch.no_grad():
        # Process in batches to limit memory
        batch_size = 512
        for start in range(0, num_frames, batch_size):
            end = min(start + batch_size, num_frames)
            x_batch = torch.from_numpy(features_std[start:end]).to(device)
            cols = []
            for sigma_val in sigma_levels:
                sigma_col = torch.full((x_batch.shape[0], 1), float(sigma_val), device=device)
                log_density = mulde_net(torch.cat([x_batch, sigma_col], dim=1)).reshape(-1)
                cols.append(log_density.cpu().numpy())
            signatures[start:end] = np.stack(cols, axis=1)

    # 4. Fit/Score GMM
    print(f"Loading GMM model from: {args.gmm}")
    gmm = joblib.load(args.gmm)
    
    raw_log_likelihood = gmm.score_samples(signatures)
    smoothed_log_likelihood = gaussian_filter1d(raw_log_likelihood, sigma=args.smooth_sigma)

    if fps <= 0:
        print("[WARNING] Invalid FPS reported by decoder; defaulting to 25.0 FPS for timestamps.")
        fps = 25.0

    video_name = os.path.splitext(os.path.basename(args.video))[0]
    manual_ranges = parse_frame_ranges(args.shading)

    df_out, threshold, segments = build_results_dataframe(
        raw_log_likelihood,
        smoothed_log_likelihood,
        fps,
        threshold_method=args.threshold_method,
        threshold_percentile=args.threshold_percentile,
        threshold_mad_k=args.threshold_mad_k,
        manual_threshold=args.threshold,
        min_segment_sec=args.min_segment_sec,
        merge_gap_sec=args.merge_gap_sec,
    )

    dashboard_path = os.path.join(args.output_dir, f"{video_name}_anomaly_dashboard.png")
    generate_anomaly_dashboard(
        df_out,
        segments,
        video_name=video_name,
        fps=fps,
        threshold=threshold,
        output_path=dashboard_path,
        manual_frame_ranges=manual_ranges or None,
        threshold_method=args.threshold_method,
    )

    artifact_paths = save_anomaly_artifacts(
        df_out,
        segments,
        output_dir=args.output_dir,
        video_name=video_name,
        fps=fps,
        threshold=threshold,
        threshold_method=args.threshold_method,
        smooth_sigma=args.smooth_sigma,
        dashboard_path=Path(dashboard_path),
    )

    print_anomaly_report(segments, fps, threshold, args.threshold_method)
    print(f"\n✓ Saved frame scores CSV:      {artifact_paths['scores_csv']}")
    print(f"✓ Saved anomaly intervals CSV: {artifact_paths['intervals_csv']}")
    print(f"✓ Saved summary JSON:          {artifact_paths['summary_json']}")
    print(f"✓ Saved anomaly dashboard:     {artifact_paths['dashboard_png']}")
    print("Inference completed successfully!")


if __name__ == "__main__":
    main()
