import numpy as np


def make_reference(f, n_samples, fs=250, n_harmonics=5):
    # sin and cos at f and its harmonics, shape (n_samples, 2 * n_harmonics)
    t = np.arange(n_samples) / fs
    cols = []
    for h in range(1, n_harmonics + 1):
        cols.append(np.sin(2 * np.pi * h * f * t))
        cols.append(np.cos(2 * np.pi * h * f * t))
    return np.array(cols).T


def cca_corr(x, y):
    # largest canonical correlation between x (samples, channels)
    # and y (samples, reference waves)
    x = x - x.mean(0)
    y = y - y.mean(0)
    qx, _ = np.linalg.qr(x)   # orthonormal basis of x's columns
    qy, _ = np.linalg.qr(y)
    # singular values of qx.T @ qy are the canonical correlations
    return np.linalg.svd(qx.T @ qy, compute_uv=False)[0]


def classify(window, freqs, fs=250, n_harmonics=5):
    # window: (channels, samples). returns predicted target index
    scores = [cca_corr(window.T, make_reference(f, window.shape[1], fs, n_harmonics))
              for f in freqs]
    return int(np.argmax(scores))