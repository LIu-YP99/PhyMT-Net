import torch
import numpy as np
from itertools import permutations

def to_3x3(m_vec):
    """
    Convert 1x6 vector (Mxx, Myy, Mzz, Mxy, Mxz, Myz) to 3x3 matrix.
    Assuming NED or similar system where vector is [M11, M22, M33, M12, M13, M23]
    """
    m = m_vec.flatten()
    M = np.array([
        [m[0], m[3], m[4]],
        [m[3], m[1], m[5]],
        [m[4], m[5], m[2]]
    ])
    return M

def calculate_cosine_similarity(y_true, y_pred):
    """
    Calculate mean cosine similarity between true and predicted tensors.
    y_true, y_pred: (N, 6)
    """
    # Normalize vectors
    norms_true = np.linalg.norm(y_true, axis=1, keepdims=True)
    norms_pred = np.linalg.norm(y_pred, axis=1, keepdims=True)
    
    # Avoid division by zero
    norms_true[norms_true < 1e-10] = 1.0
    norms_pred[norms_pred < 1e-10] = 1.0
    
    y_true_norm = y_true / norms_true
    y_pred_norm = y_pred / norms_pred
    
    # Dot product
    cos_sim = np.sum(y_true_norm * y_pred_norm, axis=1)
    # Clamp to [-1, 1] to avoid numerical errors
    cos_sim = np.clip(cos_sim, -1.0, 1.0)
    
    return np.mean(cos_sim)

def calculate_iso_percent(m_vec):
    """
    Approximation of ISO percentage.
    Using definition: %ISO = 100 * (1/3 * tr(M)) / max_eigen_abs ?
    Or energy based. 
    Standard decomposition: 
    M = M_iso + M_dc + M_clvd
    """
    # Eigen decomposition
    M = to_3x3(m_vec)
    eig_vals, _ = np.linalg.eig(M)
    eig_vals = np.sort(eig_vals) # Ascending
    
    # Isotropic part
    iso = np.mean(eig_vals)
    
    # Deviatoric eigenvalues
    dev_eigs = eig_vals - iso
    
    # Scalar moments (approximations for percentage)
    # This varies by definition. Let's use simple squared norm ratio or similar.
    # Common in seismology (Vavrycuk, 2011 or MST):
    # iso_moment = abs(iso)
    # total_moment = max(abs(eig_vals)) or sum of absolute components
    
    m_iso = np.abs(np.trace(M)) / 3.0
    # A simple metric: ISO contribution to total scalar moment
    # Total scalar moment M0 = (sum(|eigs|))/2 if purely DC? 
    # Let's use: ratio of |trace| to sum of |eigenvalues|
    
    sum_abs_eigs = np.sum(np.abs(eig_vals))
    if sum_abs_eigs < 1e-9:
        return 0.0
        
    return (np.abs(np.trace(M)) / sum_abs_eigs) * 100.0

def calculate_iso_error(y_true, y_pred):
    errors = []
    for i in range(len(y_true)):
        iso_true = calculate_iso_percent(y_true[i])
        iso_pred = calculate_iso_percent(y_pred[i])
        errors.append(np.abs(iso_true - iso_pred))
    return np.mean(errors)


# Hudson logic (adapted from evaluate.py/vis_hudson.py)
def ae_uvt_point(TT, KK):
    # Single point version
    TAU = TT * (1 - np.abs(KK))
    
    # 2nd and 4th quadrants
    if (TAU > 0 and KK < 0) or (TAU < 0 and KK > 0):
        return TAU, KK

    # First quadrant, Region A
    if (TAU < 4 * KK) and (TAU >= 0) and (KK >= 0):
        return TAU / (1 - TAU / 2), KK / (1 - TAU / 2)

    # First quadrant, Region B
    if (TAU >= 4 * KK) and (TAU >= 0) and (KK >= 0):
        return TAU / (1 - 2 * KK), KK / (1 - 2 * KK)

    # Third quadrant
    if (TAU >= 4 * KK) and (TAU <= 0) and (KK <= 0):
        return TAU / (1 + TAU / 2), KK / (1 + TAU / 2)

    if (TAU < 4 * KK) and (TAU <= 0) and (KK <= 0):
        return TAU / (1 + 2 * KK), KK / (1 + 2 * KK)
        
    return np.nan, np.nan

def get_hudson_uv(m_vec):
    """
    Calculate (u, v) coordinates for Hudson plot for a single MT.
    Need to calculate T, k parameters first.
    m_vec: [Mxx, Myy, Mzz, Mxy, Mxz, Myz] (NED)
    eigenvalues -> T, k -> u, v
    """
    M = to_3x3(m_vec)
    eigs, _ = np.linalg.eig(M)
    eigs = np.sort(eigs) # e1 <= e2 <= e3
    
    m1, m2, m3 = eigs[0], eigs[1], eigs[2]
    
    # Deviatoric eigenvalues for k
    mean_m = np.mean(eigs)
    m1_dev, m2_dev, m3_dev = m1 - mean_m, m2 - mean_m, m3 - mean_m
    
    # Sort by absolute value for T? No, T calculation:
    # T = 2 * m_iso / |m_max_dev| ??
    # Actually Hudson (1989) definitions:
    # u and v are derived from T and k.
    
    # Let's use pyrocko.hudson.project logic since it's already used in the project?
    # But I want to avoid dependency if possible.
    # Re-implementing Hudson math:
    # m'1 >= m'2 >= m'3  (Deviatoric eigenvalues sorted descending)
    # k = -m'2 / max(|m'1|, |m'3|)  (-1 <= k <= 1) (CLVD measure)
    # T = 2 * m_iso / ( |m'1| + |m'3| + ... ) ?? 
    
    # Let's rely on `hudson.project` from pyrocko if available, 
    # but since I cannot easily guarantee it in this file,
    # I will assume `calculate_test_metrics.py` can import `pyrocko`.
    # I'll output raw vectors and let the caller handle Pyrocko if present.
    
    # Wait, the user specifically asked for "Hudson Distance".
    pass

# For Kagan Angle
def get_eigen_vectors(m_vec):
    M = to_3x3(m_vec)
    vals, vecs = np.linalg.eigh(M) # eigh for symmetric
    # Sort eigenvalues and vectors
    idx = vals.argsort()
    V = vecs[:, idx]
    
    # Ensure Right-Handed System (determinant = +1)
    if np.linalg.det(V) < 0:
        V[:, -1] *= -1 # Flip last vector
        
    return V

def kagan_angle(m1, m2):
    """
    Calculate Kagan angle between two moment tensors.
    Here we calculate the rotation angle between the principal axes frames.
    Assumes strictly ordered eigenvalues (P, B, T correspondence).
    Considers the 4-fold ambiguity of principal axes directions (sign flips).
    Does NOT consider axis permutations (P must match P, T must match T).
    """
    V1 = get_eigen_vectors(m1)
    V2 = get_eigen_vectors(m2)
    
    # Calculate relative rotation matrix in the frame of V1
    # R = V1^T @ V2
    R = V1.T @ V2
    
    max_trace = -10.0
    
    # Allowed sign flips for eigenvectors that preserve RH system:
    # 4 combinations: I, and 180 deg rotations about x, y, z axes
    signs = [
        [1, 1, 1],
        [1, -1, -1],
        [-1, 1, -1],
        [-1, -1, 1]
    ]
    
    for s in signs:
        S = np.diag(s)
        # We try to maximize trace(S @ R) or trace(R @ S) - they are same for diagonal S? 
        # No, R isn't diagonal. But trace(S @ R) means summing component-wise signed diagonals.
        # Equivalent to choosing signs of columns of V2 relative to V1.
        
        tr = np.trace(R @ S) 
        if tr > max_trace:
            max_trace = tr
                
    # trace = 1 + 2 cos(theta)
    # cos(theta) = (trace - 1) / 2
    # Clamp to handle numerical noise
    cos_theta = (max_trace - 1.0) / 2.0
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    
    angle_rad = np.arccos(cos_theta)
    return np.degrees(angle_rad)
    
    # Allowed sign flips: each axis can be flipped. 2^3 = 8.
    # But usually we keep right-handed system? 
    # Kagan angle defines the minimum rotation.
    # Let's iterate all 6 permutations and all 8 sign combinations?
    # Actually for principal axes (non-directional lines), sign doesn't matter for the axis itself,
    # but for the rotation Frame it does.
    # Determinant must be +1 for rotation.
    
    for p in perms:
        R_perm = R_base[:, p]
        
        # Check all 4 sign combinations that preserve Det=1 (Rotation)
        # (1,1,1), (1,-1,-1), (-1,1,-1), (-1,-1,1)
        signs = [
            [1, 1, 1],
            [1, -1, -1],
            [-1, 1, -1],
            [-1, -1, 1]
        ]
        
        for s in signs:
            S = np.diag(s)
    # trace = 1 + 2 cos(theta)
    # cos(theta) = (trace - 1) / 2
    # Clamp to handle numerical noise
    cos_theta = (max_trace - 1.0) / 2.0
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    
    angle_rad = np.arccos(cos_theta)
    return np.degrees(angle_rad)


def is_pure_dc_reference(m_vec, tol=1e-4):
    """判断真实矩张量是否为纯 DC。"""
    eig = np.linalg.eigvalsh(to_3x3(m_vec))
    scale = np.max(np.abs(eig))

    if scale == 0:
        return False

    iso = np.mean(eig)
    dev = eig - iso
    dev_scale = np.max(np.abs(dev))

    return (
        dev_scale > 0
        and abs(iso) / scale < tol
        and abs(dev[1]) / dev_scale < tol
    )


def best_fitting_dc_tensor(m_vec):
    """从全矩张量提取 Frobenius 范数意义下的最佳拟合 DC。"""
    M = to_3x3(m_vec)

    # 去除 ISO 分量
    D = M - np.trace(M) / 3.0 * np.eye(3)

    # 偏量矩张量特征分解
    eig, V = np.linalg.eigh(D)

    # 最佳拟合 DC 的特征值为 [-a, 0, a]
    a = 0.5 * (eig[-1] - eig[0])

    M_dc = a * (
        np.outer(V[:, -1], V[:, -1])
        - np.outer(V[:, 0], V[:, 0])
    )

    return np.array([
        M_dc[0, 0], M_dc[1, 1], M_dc[2, 2],
        M_dc[0, 1], M_dc[0, 2], M_dc[1, 2]
    ])


def dc_orientation_defined(m_vec, tol=1e-3):
    """检查预测矩张量的 DC 主轴是否数值稳定。"""
    eig = np.linalg.eigvalsh(to_3x3(m_vec))
    dev = eig - np.mean(eig)

    full_scale = np.max(np.abs(eig))
    dev_scale = np.max(np.abs(dev))

    return (
        np.all(np.isfinite(eig))
        and full_scale > 0
        and dev_scale > tol * full_scale
        and np.min(np.diff(dev)) > tol * dev_scale
    )


def calculate_mean_kagan_angle_dc_only(y_true, y_pred):
    """仅统计真实震源为纯 DC 且方向有效的测试样本。"""
    angles = []
    n_dc = 0
    n_undefined = 0

    for i in range(len(y_true)):

        if not is_pure_dc_reference(y_true[i]):
            continue

        n_dc += 1

        if not dc_orientation_defined(y_pred[i]):
            n_undefined += 1
            continue

        dc_true = best_fitting_dc_tensor(y_true[i])
        dc_pred = best_fitting_dc_tensor(y_pred[i])

        angles.append(kagan_angle(dc_true, dc_pred))

    if not angles:
        raise ValueError("No valid DC samples found.")

    return (
        float(np.mean(angles)),
        n_dc,
        len(angles),
        n_undefined
    )


def calculate_mean_kagan_angle(y_true, y_pred):
    angles = []
    for i in range(len(y_true)):
        angles.append(kagan_angle(y_true[i], y_pred[i]))
    return np.mean(angles)

def calculate_hudson_distance(y_true, y_pred):
    """
    Calculate mean Euclidean distance in Hudson (u, v) space.
    Requires pyrocko.plot.hudson.
    """
    try:
        import pyrocko.plot.hudson as hudson
    except ImportError:
        print("Warning: pyrocko not found. Skipping Hudson Distance.")
        return np.nan

    dists = []
    # pyrocko hudson projection expects (mnn, mee, mdd, mne, mnd, med) -> (N, E, D, NE, ND, ED)
    # Our input is usually NED: (Mxx, Myy, Mzz, Mxy, Mxz, Myz) -> (N, E, D, NE, ND, ED)
    # They match directly.
    
    for i in range(len(y_true)):
        try:
            u_true, v_true = hudson.project(tuple(y_true[i]))
            u_pred, v_pred = hudson.project(tuple(y_pred[i]))
            
            d = np.sqrt((u_true - u_pred)**2 + (v_true - v_pred)**2)
            dists.append(d)
        except:
            dists.append(np.nan) # Handle projection errors?
            
    return np.nanmean(dists)


