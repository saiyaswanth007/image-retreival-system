"""
Color Feature Extractor (Pure NumPy)
=====================================
Extracts intra-channel and inter-channel color statistics from images.

From the assignment (Task 6):
    f_intra = [mu_R, sigma_R, mu_G, sigma_G, mu_B, sigma_B]
    f_inter = [corr(R,G), corr(G,B), corr(R,B)]

Extended features (applied-science additions):
    - Skewness and kurtosis per channel  (shape/tail info histograms miss)
    - HSV statistics                     (perceptual color space)
    - Per-channel histograms             (distribution shape, not just moments)
    - Grid-based spatial color moments   (where colors occur, not just what)

Public API:
    extract_color(images, grid=4, hist_bins=32, use_hsv=True) -> np.ndarray  (N, D)
"""

import numpy as np


# ─── COLOR SPACE CONVERSION ──────────────────────────────────────────────────

def _rgb_to_hsv(images: np.ndarray) -> np.ndarray:
    """
    Convert NCHW RGB [0..255] → NCHW HSV [H:0..360, S:0..1, V:0..1].

    Pure NumPy implementation — no OpenCV dependency.
    """
    r = images[:, 0] / 255.0   # (N, H, W)
    g = images[:, 1] / 255.0
    b = images[:, 2] / 255.0

    cmax = np.maximum(np.maximum(r, g), b)
    cmin = np.minimum(np.minimum(r, g), b)
    delta = cmax - cmin

    # ── Hue ───────────────────────────────────────────────────────────────────
    hue = np.zeros_like(delta)

    mask_r = (cmax == r) & (delta > 0)
    mask_g = (cmax == g) & (delta > 0)
    mask_b = (cmax == b) & (delta > 0)

    hue[mask_r] = 60.0 * (((g[mask_r] - b[mask_r]) / delta[mask_r]) % 6)
    hue[mask_g] = 60.0 * (((b[mask_g] - r[mask_g]) / delta[mask_g]) + 2)
    hue[mask_b] = 60.0 * (((r[mask_b] - g[mask_b]) / delta[mask_b]) + 4)

    # ── Saturation ────────────────────────────────────────────────────────────
    sat = np.where(cmax > 0, delta / cmax, 0.0)

    # ── Value ─────────────────────────────────────────────────────────────────
    val = cmax

    # Stack back to NCHW
    hsv = np.stack([hue, sat, val], axis=1)  # (N, 3, H, W)
    return hsv.astype(np.float32)


# ─── STATISTICAL MOMENTS ─────────────────────────────────────────────────────

def _channel_moments(channel_flat: np.ndarray) -> np.ndarray:
    """
    Compute [mean, std, skewness, kurtosis] for each image's channel.

    Args:
        channel_flat : float32  (N, P)  where P = H*W pixel values

    Returns:
        moments : float32  (N, 4)
    """
    mu  = channel_flat.mean(axis=1)                         # (N,)
    std = channel_flat.std(axis=1)                          # (N,)

    # Guard zero-std
    safe_std = np.where(std < 1e-7, 1.0, std)
    centered = channel_flat - mu[:, None]

    skew = (centered ** 3).mean(axis=1) / (safe_std ** 3)   # (N,)
    kurt = (centered ** 4).mean(axis=1) / (safe_std ** 4) - 3.0  # excess kurtosis

    return np.stack([mu, std, skew, kurt], axis=1)          # (N, 4)


def _intra_channel_stats(images: np.ndarray) -> np.ndarray:
    """
    Per-channel statistics: [mu, sigma, skew, kurtosis] × 3 channels.

    Args:
        images : float32  (N, 3, H, W)  pixel values in [0..255]

    Returns:
        features : float32  (N, 12)  — 4 stats × 3 channels
    """
    N, C, H, W = images.shape
    all_moments = []

    for c in range(C):
        flat = images[:, c].reshape(N, -1)                  # (N, H*W)
        moments = _channel_moments(flat)                    # (N, 4)
        all_moments.append(moments)

    return np.concatenate(all_moments, axis=1)              # (N, 12)


def _inter_channel_corr(images: np.ndarray) -> np.ndarray:
    """
    Pairwise Pearson correlation between channels: corr(R,G), corr(G,B), corr(R,B).

    Args:
        images : float32  (N, 3, H, W)

    Returns:
        features : float32  (N, 3)
    """
    N = images.shape[0]
    flat = images.reshape(N, 3, -1)   # (N, 3, H*W)

    pairs = [(0, 1), (1, 2), (0, 2)]  # (R,G), (G,B), (R,B)
    corrs = np.zeros((N, 3), dtype=np.float32)

    for idx, (ci, cj) in enumerate(pairs):
        a = flat[:, ci]  # (N, P)
        b = flat[:, cj]  # (N, P)

        a_centered = a - a.mean(axis=1, keepdims=True)
        b_centered = b - b.mean(axis=1, keepdims=True)

        num   = (a_centered * b_centered).mean(axis=1)
        denom = a.std(axis=1) * b.std(axis=1)
        denom = np.where(denom < 1e-7, 1.0, denom)

        corrs[:, idx] = num / denom

    return corrs


# ─── CHANNEL HISTOGRAMS ──────────────────────────────────────────────────────

def _channel_histograms(images: np.ndarray, n_bins: int = 32) -> np.ndarray:
    """
    Per-channel intensity histograms (RGB), L1-normalized.

    Args:
        images : float32  (N, 3, H, W)  in [0..255]
        n_bins : histogram bins per channel

    Returns:
        features : float32  (N, 3*n_bins)
    """
    N, C = images.shape[0], images.shape[1]

    # Quantize to bin indices
    clipped = np.clip(images, 0, 255)
    bin_idx = (clipped * (n_bins / 256.0)).astype(np.int32)
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)

    features = np.zeros((N, C * n_bins), dtype=np.float32)

    for c in range(C):
        flat = bin_idx[:, c].reshape(N, -1)  # (N, H*W)
        offset = c * n_bins
        for i in range(N):
            counts = np.bincount(flat[i], minlength=n_bins)[:n_bins]
            features[i, offset : offset + n_bins] = counts.astype(np.float32)

    # L1-normalize per image
    row_sums = features.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums < 1e-7, 1.0, row_sums)
    features /= row_sums

    return features


# ─── GRID-BASED SPATIAL COLOR MOMENTS ────────────────────────────────────────

def _grid_color_moments(images: np.ndarray, grid: int) -> np.ndarray:
    """
    Compute per-channel [mean, std] in each spatial grid cell.
    Captures WHERE colors appear, not just global stats.

    Args:
        images : float32  (N, 3, H, W)  in [0..255]
        grid   : spatial grid divisions

    Returns:
        features : float32  (N, grid*grid*3*2)
    """
    N, C, H, W = images.shape
    cell_h = H // grid
    cell_w = W // grid

    # 2 stats (mean, std) per channel per cell
    D = grid * grid * C * 2
    features = np.zeros((N, D), dtype=np.float32)

    idx = 0
    for gy in range(grid):
        for gx in range(grid):
            y0, y1 = gy * cell_h, (gy + 1) * cell_h
            x0, x1 = gx * cell_w, (gx + 1) * cell_w
            cell = images[:, :, y0:y1, x0:x1]  # (N, C, cell_h, cell_w)

            for c in range(C):
                flat = cell[:, c].reshape(N, -1)       # (N, cell_h*cell_w)
                features[:, idx]     = flat.mean(axis=1)
                features[:, idx + 1] = flat.std(axis=1)
                idx += 2

    return features


# ─── PUBLIC API ───────────────────────────────────────────────────────────────

def extract_color(
    images: np.ndarray,
    grid: int = 4,
    hist_bins: int = 32,
    use_hsv: bool = True,
) -> np.ndarray:
    """
    Extract color features from a batch of RGB images.

    Feature vector composition:
        1. Intra-channel stats   : [mu, std, skew, kurt] × 3 RGB channels    = 12
        2. Inter-channel corr    : corr(R,G), corr(G,B), corr(R,B)           =  3
        3. RGB histograms        : hist_bins × 3 channels                     = 96  (if bins=32)
        4. HSV intra stats       : [mu, std, skew, kurt] × 3 HSV channels    = 12  (if use_hsv)
        5. HSV histograms        : hist_bins × 3 channels                     = 96  (if use_hsv)
        6. Grid color moments    : grid×grid × 3 channels × 2 (mean, std)    = 96  (if grid=4)

    Total default dim: 12 + 3 + 96 + 12 + 96 + 96 = 315

    Args:
        images    : float32  (N, C, H, W)  pixel values in [0..255] range
        grid      : spatial grid for local color moments (default 4)
        hist_bins : bins per channel for histograms (default 32)
        use_hsv   : include HSV color space features (default True)

    Returns:
        features : float32  (N, D)  L2-normalized
    """
    assert images.ndim == 4, f"Expected NCHW, got shape {images.shape}"
    assert images.shape[1] == 3, f"Color features require 3 channels, got {images.shape[1]}"

    parts = []

    # ── 1. Intra-channel statistics (RGB) ─────────────────────────────────────
    parts.append(_intra_channel_stats(images))                 # (N, 12)

    # ── 2. Inter-channel correlations ─────────────────────────────────────────
    parts.append(_inter_channel_corr(images))                  # (N, 3)

    # ── 3. RGB channel histograms ─────────────────────────────────────────────
    parts.append(_channel_histograms(images, n_bins=hist_bins))  # (N, 3*bins)

    # ── 4+5. HSV features ─────────────────────────────────────────────────────
    if use_hsv:
        hsv = _rgb_to_hsv(images)
        # Scale H from [0..360] to [0..255] for histogram compatibility
        hsv_scaled = hsv.copy()
        hsv_scaled[:, 0] = hsv[:, 0] * (255.0 / 360.0)
        hsv_scaled[:, 1] = hsv[:, 1] * 255.0
        hsv_scaled[:, 2] = hsv[:, 2] * 255.0

        parts.append(_intra_channel_stats(hsv))                    # (N, 12)
        parts.append(_channel_histograms(hsv_scaled, n_bins=hist_bins))  # (N, 3*bins)

    # ── 6. Grid-based spatial color moments ───────────────────────────────────
    parts.append(_grid_color_moments(images, grid))            # (N, grid*grid*6)

    # ── Concatenate + L2-normalize ────────────────────────────────────────────
    features = np.concatenate(parts, axis=1)

    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms = np.where(norms < 1e-7, 1.0, norms)
    features /= norms

    return features
