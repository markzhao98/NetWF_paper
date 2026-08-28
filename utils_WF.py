# Tingyu Zhao

import pickle
import numpy as np
import pandas as pd
import networkx as nx

import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, cg, minres
from scipy.sparse import save_npz, load_npz
import scipy.stats as stats
from scipy.stats import pearsonr, rankdata

from sklearn.metrics import auc
from sklearn.utils import resample
from collections import defaultdict
import random
import math

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Palatino']
plt.rcParams['font.size'] = 16
plt.rcParams['text.usetex'] = True
plt.rcParams['lines.linewidth'] = 1.5
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['figure.titlesize'] = 20
plt.rcParams['legend.fontsize'] = 16
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16

from typing import Literal, Optional, Tuple, Union


ProfileSimilarityMode = Literal["mean", "max", "max_abs", "source", "target"]

def calculate_profile_similarity(
    A: np.ndarray,
    threshold: float = -np.inf,
    mode: ProfileSimilarityMode = "mean",
) -> np.ndarray:
    """
    Compute a node-level profile similarity network (PSN) from an adjacency/data matrix.

    The PSN is an n×n symmetric matrix whose (i, j) entry measures how similar node i and node j
    are based on their connectivity profiles in A.

    For each pair of nodes (i, j), we compute:
      - a *source-profile* correlation: Pearson correlation between rows A[i, :] and A[j, :],
        using only indices where both entries are observed (non-NaN).
      - a *target-profile* correlation: Pearson correlation between columns A[:, i] and A[:, j],
        using only indices where both entries are observed (non-NaN).

    The final similarity value is chosen by `mode`:
      - "mean": average of the source and target correlations (ignoring NaNs)
      - "max": source/target correlation with the larger value (ignoring NaNs)
      - "max_abs": source/target correlation with the larger absolute value (ignoring NaNs)
      - "source": source-profile correlation only
      - "target": target-profile correlation only

    Parameters
    ----------
    A : np.ndarray
        (n×n) data/adjacency matrix. NaNs are treated as missing observations.
    threshold : float
        Values below this threshold are set to 0 in the returned similarity matrix.
    mode : {"mean", "max", "max_abs", "source", "target"}
        How to combine/choose source/target profile correlations.

    Returns
    -------
    sim : np.ndarray
        (n×n) symmetric node similarity matrix with ones on the diagonal.

    Notes
    -----
    When A contains no NaNs, all pairwise correlations are computed at once via
    np.corrcoef (a vectorized path that is orders of magnitude faster than the
    pairwise loop and numerically identical to it). The per-pair loop below is
    only used as a fallback when A has missing entries, since each pair then
    needs its own mask of jointly observed indices.
    """
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"A must be square (n×n). Got shape {A.shape}.")

    if mode not in ("source", "target", "mean", "max", "max_abs"):
        raise ValueError("Invalid mode. Choose 'mean', 'max', 'max_abs', 'source', or 'target'.")

    n = A.shape[0]

    if not np.isnan(A).any():
        # Constant rows/columns have zero variance; corrcoef yields NaN there,
        # matching the pairwise convention below (handled by nan_to_num at the end).
        with np.errstate(invalid="ignore", divide="ignore"):
            source_corrs = np.corrcoef(A)    # source profiles: rows
            target_corrs = np.corrcoef(A.T)  # target profiles: columns

        if mode == "source":
            sim = source_corrs
        elif mode == "target":
            sim = target_corrs
        elif mode == "mean":
            with np.errstate(invalid="ignore"):
                sim = np.nanmean(np.stack([source_corrs, target_corrs]), axis=0)
        elif mode == "max":
            with np.errstate(invalid="ignore"):
                sim = np.nanmax(np.stack([source_corrs, target_corrs]), axis=0)
        else:  # max_abs
            sim = np.where(
                np.isnan(target_corrs), source_corrs,
                np.where(
                    np.isnan(source_corrs), target_corrs,
                    np.where(np.abs(source_corrs) >= np.abs(target_corrs), source_corrs, target_corrs),
                ),
            )

        sim = np.array(sim, dtype=float)
        np.fill_diagonal(sim, 1.0)   # match the pairwise convention (diagonal from np.eye)
        sim = 0.5 * (sim + sim.T)    # enforce exact symmetry
        sim[sim < threshold] = 0.0
        return np.nan_to_num(sim, nan=0.0)

    sim = np.eye(n, dtype=float)

    for i in range(n):
        for j in range(i):
            # --- Source-profile similarity: correlate outgoing profiles (rows) ---
            row_i = A[i, :]
            row_j = A[j, :]
            valid_row = ~np.isnan(row_i) & ~np.isnan(row_j)

            if np.any(valid_row):
                row_i_valid = row_i[valid_row]
                row_j_valid = row_j[valid_row]

                # Pearson correlation is undefined if either vector is constant.
                if np.all(row_i_valid == row_i_valid[0]) or np.all(row_j_valid == row_j_valid[0]):
                    source_corr = np.nan
                else:
                    source_corr, _ = pearsonr(row_i_valid, row_j_valid)
            else:
                source_corr = np.nan

            # --- Target-profile similarity: correlate incoming profiles (columns) ---
            col_i = A[:, i]
            col_j = A[:, j]
            valid_col = ~np.isnan(col_i) & ~np.isnan(col_j)

            if np.any(valid_col):
                col_i_valid = col_i[valid_col]
                col_j_valid = col_j[valid_col]

                if np.all(col_i_valid == col_i_valid[0]) or np.all(col_j_valid == col_j_valid[0]):
                    target_corr = np.nan
                else:
                    target_corr, _ = pearsonr(col_i_valid, col_j_valid)
            else:
                target_corr = np.nan

            # --- Combine/choose correlations according to the requested mode ---
            if mode == "mean":
                corr = np.nanmean([source_corr, target_corr])
            elif mode == "max":
                corr = np.nanmax([source_corr, target_corr])
            elif mode == "max_abs":
                if np.isnan(source_corr) and np.isnan(target_corr):
                    corr = np.nan
                elif np.isnan(target_corr):
                    corr = source_corr
                elif np.isnan(source_corr):
                    corr = target_corr
                else:
                    corr = source_corr if abs(source_corr) >= abs(target_corr) else target_corr
            elif mode == "source":
                corr = source_corr
            elif mode == "target":
                corr = target_corr
            else:  # defensive (should be unreachable)
                raise ValueError("Invalid mode. Choose 'mean', 'max', 'max_abs', 'source', or 'target'.")

            sim[i, j] = corr
            sim[j, i] = corr

    # Threshold small/negative similarities and replace any remaining NaNs with 0.
    sim[sim < threshold] = 0.0
    return np.nan_to_num(sim, nan=0.0)


def wiener_filter_direct(
    A: np.ndarray,
    Cn: np.ndarray,
    *,
    epsilon: float = 1e-2,
    directed: bool = True,
    similarity_threshold: float = -np.inf,
    onlypos: bool = False,
    noSL: bool = False,
    renorm: bool = False,
) -> np.ndarray:
    """
    Direct (dense) Wiener filtering in edge space.

    Model
    -----
    Observed matrix A ∈ R^{n×n}:
        A = U + N,
    where U is the latent (signal) matrix and N is additive noise. Define the demeaned edge-space vector
        a = vec(A - A_mu),
    where A_mu is the (NaN-ignoring) mean of A and vec(·) uses NumPy's row-major flattening.

    The Wiener estimate is
        u_hat = Cu (Cu + Cn + epsilon I)^{-1} a,
    and the returned matrix estimate is
        U_hat = unvec(u_hat) + A_mu.

    Covariances
    -----------
    - Signal covariance Cu is built from profile similarity networks (PSNs):
        * directed=True:
            Cu = A_var * (S_source ⊗ S_target)
          where S_source is the PSN of rows (outgoing profiles) and S_target is the PSN of columns
          (incoming profiles).
        * directed=False:
            Cu = A_var * 0.5*(K + K_swapped),   K = S ⊗ S,
          where K_swapped corresponds to swapping the endpoints of one edge (tensor transpose (0,1,3,2)
          in the (A,B,C,D) indexing).

    - Noise covariance Cn may be fully dense in edge space, or provided in diagonal shorthand:
        * (n², n²): full edge-space covariance
        * (n, n): per-entry noise variances -> diag(vec(Cn))
        * (n²,): diagonal of Cn

    Missing values
    --------------
    - NaNs in A are filled with A_mu before constructing PSNs and forming a.
    - If Cn is supplied as variances/diagonal and contains NaNs, they are imputed with a conservative
      fallback variance:
          fallback = max(A_var, 0) + max(mean_noise_var, 0),
      where mean_noise_var is computed from the provided noise variances (ignoring NaNs).
      For a dense (n²×n²) Cn, NaNs on the diagonal are filled with the same fallback and NaNs off-diagonal
      are set to 0.

    Notes
    -----
    This routine explicitly forms Cu (an n²×n² matrix) and solves a dense linear system. It is intended
    for small n or validation; for larger networks, use `wiener_filter_iterative`.

    Parameters
    ----------
    A : np.ndarray
        (n×n) observed data matrix. NaNs are treated as missing observations.
    Cn : np.ndarray
        Noise covariance in edge space (or a diagonal shorthand; see above).
    epsilon : float
        Regularization strength (adds epsilon I in edge space).
    directed : bool
        If True, use directed Cu construction; otherwise use the undirected swap symmetrization.
    similarity_threshold : float
        Similarities below this value are set to 0 in PSNs.
    onlypos : bool
        If True, clip negative entries of U_hat to 0.
    noSL : bool
        If True, set the diagonal of U_hat to 0.
    renorm : bool
        If True, rescale U_hat so that mean(U_hat) == mean(A_filled).

    Returns
    -------
    U_hat : np.ndarray
        (n×n) Wiener estimate of the latent signal matrix U.
    """
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"A must be square (n×n). Got shape {A.shape}.")

    n = A.shape[0]
    n2 = n * n

    # Scalar mean and variance of the observed matrix (ignoring NaNs).
    A_mu = float(np.nanmean(A))
    A_var = float(np.nanvar(A))

    # Fill missing entries in A with the global mean.
    A_filled = np.nan_to_num(A, nan=A_mu)

    # Demeaned edge-space vector a = vec(A - A_mu).
    a = (A_filled - A_mu).reshape(-1)

    # --- Build Cu explicitly (brute force) ---
    if directed:
        S_source = calculate_profile_similarity(A_filled, threshold=similarity_threshold, mode="source")
        S_target = calculate_profile_similarity(A_filled, threshold=similarity_threshold, mode="target")
        Cu = A_var * np.kron(S_source, S_target)
    else:
        S = calculate_profile_similarity(A_filled, threshold=similarity_threshold, mode="mean")
        K = np.kron(S, S)
        K_tensor = K.reshape(n, n, n, n)
        K_swapped = K_tensor.transpose(0, 1, 3, 2).reshape(n2, n2)
        Cu = A_var * 0.5 * (K + K_swapped)

    # --- Normalize/interpret Cn input into an (n²×n²) matrix, with NaN handling ---
    # Compute a conservative fallback variance for imputation (used only when we interpret Cn as variances).
    mean_noise_var = np.nan
    if Cn.ndim == 2 and Cn.shape == (n, n):
        mean_noise_var = float(np.nanmean(Cn))
    elif Cn.ndim == 1 and Cn.shape[0] == n2:
        mean_noise_var = float(np.nanmean(Cn))
    elif Cn.ndim == 2 and Cn.shape == (n2, n2):
        # For a full covariance, the diagonal carries the variances.
        mean_noise_var = float(np.nanmean(np.diag(Cn)))
    else:
        # Shape errors are handled below.
        pass

    fallback_var = (A_var if np.isfinite(A_var) and A_var > 0 else 0.0) + (
        mean_noise_var if np.isfinite(mean_noise_var) and mean_noise_var > 0 else 0.0
    )

    if Cn.ndim == 2 and Cn.shape == (n2, n2):
        Cn_mat = np.array(Cn, dtype=float, copy=True)
        if np.isnan(Cn_mat).any():
            # Fill off-diagonal NaNs with 0 and diagonal NaNs with fallback_var.
            diag_nan = np.isnan(np.diag(Cn_mat))
            Cn_mat = np.nan_to_num(Cn_mat, nan=0.0)
            if np.any(diag_nan):
                idx = np.where(diag_nan)[0]
                Cn_mat[idx, idx] = fallback_var
        # Symmetrize for numerical stability (users should provide symmetric covariances).
        Cn_mat = 0.5 * (Cn_mat + Cn_mat.T)

    elif Cn.ndim == 2 and Cn.shape == (n, n):
        var = np.nan_to_num(Cn, nan=fallback_var)
        var = np.where(var < 0, 0.0, var)  # guard against invalid negative variances
        Cn_mat = np.diag(var.reshape(-1))

    elif Cn.ndim == 1 and Cn.shape[0] == n2:
        diag = np.nan_to_num(Cn, nan=fallback_var)
        diag = np.where(diag < 0, 0.0, diag)
        Cn_mat = np.diag(diag)

    else:
        raise ValueError(
            "Cn must have shape (n^2,n^2), (n,n) (interpreted as variances), or (n^2,) (diagonal). "
            f"Got shape {Cn.shape}."
        )

    # Solve for x = (Cu + Cn + epsilon I)^{-1} a without explicitly computing an inverse.
    system = Cu + Cn_mat + epsilon * np.eye(n2)
    x = np.linalg.solve(system, a)

    # Wiener map: u_hat = Cu x, then add back the mean to obtain U_hat.
    u_hat = Cu @ x
    U_hat = u_hat.reshape(n, n) + A_mu

    # Optional post-processing.
    if onlypos:
        U_hat[U_hat < 0] = 0.0
    if noSL:
        np.fill_diagonal(U_hat, 0.0)
    if renorm:
        U_hat_mean = float(np.mean(U_hat))
        A_filled_mean = float(np.mean(A_filled))
        if U_hat_mean != 0:
            U_hat = U_hat * (A_filled_mean / U_hat_mean)

    return U_hat


def wiener_filter_iterative(
    A: np.ndarray,
    A_noise_var: np.ndarray,
    *,
    epsilon: float = 1e-2,
    directed: bool = True,
    similarity_threshold: float = -np.inf,
    solver: Literal["cg", "minres"] = "cg",
    rtol: float = 1e-6,
    maxiter: int = 2000,
    onlypos: bool = False,
    noSL: bool = False,
    renorm: bool = False,
    return_info: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, int]]:
    """
    Iterative Wiener filtering with diagonal noise covariance (no n²×n² matrix construction).

    This routine estimates the latent matrix U in the additive noise model A = U + N using the Wiener map
        u_hat = Cu (Cu + Cn + epsilon I)^{-1} a,
    where a = vec(A - A_mu) and U_hat = unvec(u_hat) + A_mu.

    Here Cn is assumed diagonal in edge space:
        Cn = diag(vec(A_noise_var)).
    The signal covariance Cu is built from profile similarity networks (PSNs):
        * directed=True:
            Cu = A_var * (S_source ⊗ S_target)
        * directed=False:
            Cu = A_var * 0.5*(K + K_swapped),   K = S ⊗ S,
          with K_swapped corresponding to swapping the endpoints of one edge (tensor transpose (0,1,3,2)
          in the (A,B,C,D) indexing).

    The linear system (Cu + Cn + epsilon I) x = a is solved by a Krylov method (CG or MINRES) using only
    matrix–vector products, with Cu applied via Kronecker-structured matrix multiplications.

    Missing values
    --------------
    - NaNs in A are filled with A_mu before constructing PSNs and forming a.
    - NaNs in A_noise_var are imputed with the conservative fallback variance:
          fallback = max(A_var, 0) + max(mean(A_noise_var), 0).

    Parameters
    ----------
    A : np.ndarray
        (n×n) observed data matrix. NaNs are treated as missing observations.
    A_noise_var : np.ndarray
        (n×n) per-entry noise variances. NaNs are imputed conservatively.
    epsilon : float
        Regularization strength (adds epsilon I in edge space).
    directed : bool
        If True, use directed Cu construction; otherwise use the undirected swap symmetrization.
    similarity_threshold : float
        Similarities below this value are set to 0 in PSNs.
    solver : {"cg", "minres"}
        Iterative solver. CG assumes a symmetric positive definite operator; MINRES is more robust.
    rtol : float
        Relative tolerance for the iterative solver stopping criterion.
    maxiter : int
        Maximum number of iterations for the iterative solver.
    onlypos : bool
        If True, clip negative entries of U_hat to 0.
    noSL : bool
        If True, set the diagonal of U_hat to 0.
    renorm : bool
        If True, rescale U_hat so that mean(U_hat) == mean(A_filled).
    return_info : bool
        If True, also return the solver diagnostic `info`.

    Returns
    -------
    U_hat : np.ndarray
        (n×n) Wiener estimate of the latent signal matrix U.
    info : int (optional)
        Solver diagnostic: 0 means converged; >0 means no convergence within maxiter; <0 indicates breakdown.
    """
    if A.shape != A_noise_var.shape:
        raise ValueError(f"A and A_noise_var must have the same shape; got {A.shape} vs {A_noise_var.shape}")
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"A must be square (n×n). Got shape {A.shape}.")

    n = A.shape[0]

    # Scalar mean and variance of the observed matrix (ignoring NaNs).
    A_mu = float(np.nanmean(A))
    A_var = float(np.nanvar(A))

    # Mean noise variance (ignoring NaNs) used for conservative imputation below.
    noise_var_mu = float(np.nanmean(A_noise_var))

    # Fill missing entries in A with the global mean.
    A_filled = np.nan_to_num(A, nan=A_mu)

    # Fill missing noise variances conservatively to avoid underestimating noise.
    fallback_var = (A_var if np.isfinite(A_var) and A_var > 0 else 0.0) + (
        noise_var_mu if np.isfinite(noise_var_mu) and noise_var_mu > 0 else 0.0
    )
    A_noise_var_filled = np.nan_to_num(A_noise_var, nan=fallback_var)
    A_noise_var_filled = np.where(A_noise_var_filled < 0, 0.0, A_noise_var_filled)

    # Demeaned data in matrix form and in edge-vector form.
    a = (A_filled - A_mu).reshape(-1)

    # Diagonal of Cn in edge space.
    cn_diag = A_noise_var_filled.reshape(-1)

    # Build PSNs and define the Cu matvec via the Kronecker trick.
    if directed:
        S_source = calculate_profile_similarity(A_filled, threshold=similarity_threshold, mode="source")
        S_target = calculate_profile_similarity(A_filled, threshold=similarity_threshold, mode="target")

        def Cu_mv(v: np.ndarray) -> np.ndarray:
            X = v.reshape(n, n)
            Y = S_source @ X @ S_target.T
            return (A_var * Y).reshape(-1)

        diag_Cu_approx = A_var * np.kron(np.diag(S_source), np.diag(S_target))

    else:
        S = calculate_profile_similarity(A_filled, threshold=similarity_threshold, mode="mean")
        # Enforce exact symmetry for numerical stability.
        S = 0.5 * (S + S.T)

        def Cu_mv(v: np.ndarray) -> np.ndarray:
            X = v.reshape(n, n)
            Y1 = S @ X @ S.T
            Y = 0.5 * (Y1 + Y1.T)  # equals 0.5*(S X S^T + S X^T S^T) when S is symmetric
            return (A_var * Y).reshape(-1)

        diag_Cu_approx = A_var * np.kron(np.diag(S), np.diag(S))

    # Define the full linear operator: (Cu + Cn + epsilon I) v
    def Aop_mv(v: np.ndarray) -> np.ndarray:
        return Cu_mv(v) + (cn_diag + epsilon) * v

    Aop = LinearOperator((n * n, n * n), matvec=Aop_mv, dtype=np.float64)

    # Diagonal preconditioner M ≈ Cu + Cn + epsilon I
    Mdiag = diag_Cu_approx + cn_diag + epsilon
    Mdiag = np.where(Mdiag == 0, 1.0, Mdiag)  # avoid division by zero
    Mop = LinearOperator((n * n, n * n), matvec=lambda v: v / Mdiag, dtype=np.float64)

    # Solve (Cu + Cn + epsilon I) x = a
    solver = solver.lower()
    if solver == "cg":
        x, info = cg(Aop, a, M=Mop, atol=0.0, rtol=rtol, maxiter=maxiter)
    elif solver == "minres":
        x, info = minres(Aop, a, M=Mop, rtol=rtol, maxiter=maxiter)
    else:
        raise ValueError("solver must be 'cg' or 'minres'")

    # Apply the Wiener map: u_hat = Cu x, then add back the mean to obtain U_hat.
    u_hat_vec = Cu_mv(x)
    U_hat = u_hat_vec.reshape(n, n) + A_mu

    # Optional post-processing.
    if onlypos:
        U_hat[U_hat < 0] = 0.0
    if noSL:
        np.fill_diagonal(U_hat, 0.0)
    if renorm:
        U_hat_mean = float(np.mean(U_hat))
        A_filled_mean = float(np.mean(A_filled))
        if U_hat_mean != 0:
            U_hat = U_hat * (A_filled_mean / U_hat_mean)

    return (U_hat, info) if return_info else U_hat

