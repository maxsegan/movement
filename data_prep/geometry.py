import numpy as np


def normalize_screen(xy, w, h):
    out = xy.copy()
    out[..., 0] = (out[..., 0] / w) * 2.0 - 1.0
    out[..., 1] = (out[..., 1] / w) * 2.0 - (h / w)
    return out


def fit_similarity_2d(src_xy, dst_xy, eps=1e-8):
    x0 = src_xy - src_xy[0]
    y0 = dst_xy - dst_xy[0]
    A = x0.astype(np.float64)
    B = y0.astype(np.float64)
    M = A.T @ B
    U, S, Vt = np.linalg.svd(M)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = U @ Vt
    denom = (A * A).sum()
    s = (S.sum() / max(denom, eps)) if denom > eps else 1.0
    t = dst_xy[0] - (A[0] @ R) * s
    return s.astype(np.float32), R.astype(np.float32), t.astype(np.float32)


