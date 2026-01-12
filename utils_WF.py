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


ProfileSimilarityMode = Literal["mean", "max", "source", "target"]

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
      - "max": max of the source and target correlations (ignoring NaNs)
      - "source": source-profile correlation only
      - "target": target-profile correlation only

    Parameters
    ----------
    A : np.ndarray
        (n×n) data/adjacency matrix. NaNs are treated as missing observations.
    threshold : float
        Values below this threshold are set to 0 in the returned similarity matrix.
    mode : {"mean", "max", "source", "target"}
        How to combine/choose source/target profile correlations.

    Returns
    -------
    sim : np.ndarray
        (n×n) symmetric node similarity matrix with ones on the diagonal.
    """
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"A must be square (n×n). Got shape {A.shape}.")

    # Backward-compatible aliases (kept internal; public docs use source/target)
    if mode == "source":  # ok
        pass
    elif mode == "target":  # ok
        pass
    elif mode not in ("mean", "max"):
        raise ValueError("Invalid mode. Choose 'mean', 'max', 'source', or 'target'.")

    n = A.shape[0]
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
            elif mode == "source":
                corr = source_corr
            elif mode == "target":
                corr = target_corr
            else:  # defensive (should be unreachable)
                raise ValueError("Invalid mode. Choose 'mean', 'max', 'source', or 'target'.")

            sim[i, j] = corr
            sim[j, i] = corr

    # Threshold small/negative similarities and replace any remaining NaNs with 0.
    sim[sim < threshold] = 0.0
    return np.nan_to_num(sim, nan=0.0)


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
    Wiener filter with diagonal noise covariance, solved iteratively without forming n²×n² matrices.

    Model/notation
    --------------
    Let A ∈ R^{n×n} be the observed data matrix:
        A = U + N
    where U is the latent (signal) matrix and N is additive noise. In edge space:
        a = vec(A - A_mu),   u = vec(U),   n = vec(N).

    This function computes the (demeaned) Wiener estimate
        u_hat = Cu (Cu + Cn + epsilon I)^{-1} a,
    and returns
        A_hat = unvec(u_hat) + A_mu.

    Covariance structure used here
    ------------------------------
    - Noise covariance is assumed diagonal in edge space:
          Cn = diag(vec(A_noise_var)).

    - Signal covariance Cu is built from node-level profile similarity networks (PSNs):

      Directed networks:
        Compute two PSNs:
          sim_source = PSN from source profiles (rows of A)   -> mode="source"
          sim_target = PSN from target profiles (cols of A)   -> mode="target"
        and set
          Cu = A_var * (sim_source ⊗ sim_target),
        which implies edge similarity
          Cu_{(i,j),(k,l)} ∝ sim_source[i,k] * sim_target[j,l].

      Undirected networks:
        Compute one PSN using mode="mean":
          sim = PSN(A, mode="mean")
        and use the same “swap” symmetrization defined by the tensor transpose (0,3,2,1):
          K = sim ⊗ sim,
          K_swapped = reshape(K) with tensor transpose (0,3,2,1), then reshape back,
          Cu = A_var * 0.5 (K + K_swapped).

    Efficiency notes
    ----------------
    The linear system
        (Cu + Cn + epsilon I) x = a
    is solved via a Krylov method (CG or MINRES) using only matrix-vector products. The matvec
    with Cu is implemented via the Kronecker trick in matrix form, avoiding construction of Cu.

    Parameters
    ----------
    A : np.ndarray
        (n×n) observed data matrix. NaNs allowed (treated as missing).
    A_noise_var : np.ndarray
        (n×n) per-entry noise variances. NaNs allowed; missing variances are imputed conservatively.
    epsilon : float
        Regularization strength for numerical stability (adds epsilon I in edge space).
    directed : bool
        If True, build Cu from (sim_source ⊗ sim_target). If False, use undirected swap symmetrization.
    similarity_threshold : float
        Threshold applied to PSNs: similarities below this value are set to 0.
    solver : {"cg", "minres"}
        Iterative solver. CG requires the operator to be symmetric positive definite; MINRES is more robust.
    rtol : float
        Relative tolerance for the iterative solver stopping criterion.
    maxiter : int
        Maximum number of iterations for the iterative solver.
    onlypos : bool
        If True, set negative entries of the output to 0.
    noSL : bool
        If True, set the diagonal entries of the output to 0.
    renorm : bool
        If True, rescale the output so that mean(output) == mean(filled A).
    return_info : bool
        If True, also return the solver diagnostic `info`.

    Returns
    -------
    A_hat : np.ndarray
        (n×n) Wiener-filtered estimate of A.
    info : int (optional)
        Solver diagnostic: 0 means converged, >0 means did not converge within maxiter, <0 indicates breakdown.
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
    # If A_var is degenerate (0 or NaN), fall back to noise_var_mu.
    fallback_var = (A_var if np.isfinite(A_var) and A_var > 0 else 0.0) + (noise_var_mu if np.isfinite(noise_var_mu) else 0.0)
    A_noise_var_filled = np.nan_to_num(A_noise_var, nan=fallback_var)

    # Demeaned data in matrix form and in edge-vector form.
    A_dm = A_filled - A_mu
    a = A_dm.reshape(-1)  # vec(A - A_mu) with row-major ordering

    # Diagonal of Cn in edge space.
    cn_diag = A_noise_var_filled.reshape(-1)

    # Build PSNs and define the Cu matvec via the Kronecker trick.
    if directed:
        # Source PSN: similarity of outgoing profiles (rows)
        sim_source = calculate_profile_similarity(A_filled, threshold=similarity_threshold, mode="source")
        # Target PSN: similarity of incoming profiles (columns)
        sim_target = calculate_profile_similarity(A_filled, threshold=similarity_threshold, mode="target")

        def Cu_mv(v: np.ndarray) -> np.ndarray:
            """Compute Cu v with Cu = A_var * (sim_source ⊗ sim_target) without forming Cu."""
            X = v.reshape(n, n)
            Y = sim_source @ X @ sim_target.T
            return (A_var * Y).reshape(-1)

        # Approximate diagonal of Cu for a diagonal preconditioner.
        diag_Cu_approx = A_var * np.kron(np.diag(sim_source), np.diag(sim_target))

    else:
        # Undirected PSN uses the mean of source/target correlations.
        sim = calculate_profile_similarity(A_filled, threshold=similarity_threshold, mode="mean")

        def Cu_mv(v: np.ndarray) -> np.ndarray:
            """Compute Cu v with Cu = A_var * 0.5*(K + K_swapped) using a matrix-form matvec."""
            X = v.reshape(n, n)

            # K action: vec(sim X sim^T)
            Y1 = sim @ X @ sim.T

            # K_swapped action corresponding to the tensor transpose (0,3,2,1) defined by the tensor transpose (0,3,2,1):
            # vec^{-1}(K_swapped vec(X)) = sim X sim
            Y2 = sim @ X @ sim

            Y = 0.5 * (Y1 + Y2)
            return (A_var * Y).reshape(-1)

        diag_Cu_approx = A_var * np.kron(np.diag(sim), np.diag(sim))

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

    # Apply the Wiener map: u_hat = Cu x
    u_hat_vec = Cu_mv(x)
    U_hat = u_hat_vec.reshape(n, n)

    # Add back the mean to form the filtered observation.
    A_hat = U_hat + A_mu

    # Optional post-processing.
    if onlypos:
        A_hat[A_hat < 0] = 0.0
    if noSL:
        np.fill_diagonal(A_hat, 0.0)
    if renorm:
        A_hat_mean = float(np.mean(A_hat))
        A_filled_mean = float(np.mean(A_filled))
        if A_hat_mean != 0:
            A_hat = A_hat * (A_filled_mean / A_hat_mean)

    return (A_hat, info) if return_info else A_hat


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
    Direct (brute-force) Wiener filtering with a fully general noise covariance in edge space.

    Model/notation
    --------------
    Observed data:
        A = U + N,  with A ∈ R^{n×n}.
    Edge-space vectors:
        a = vec(A - A_mu),   u = vec(U),   n = vec(N).

    This function computes
        u_hat = Cu (Cu + Cn + epsilon I)^{-1} a,
    then returns
        A_hat = unvec(u_hat) + A_mu.

    Covariances
    -----------
    - The signal covariance Cu is constructed from PSNs exactly as in `wiener_filter_iterative`:
        directed=True  -> Cu = A_var * (sim_source ⊗ sim_target)
        directed=False -> Cu = A_var * 0.5*(K + K_swapped) with K = sim ⊗ sim and the same swap rule.

    - The noise covariance Cn is user-provided and may be fully dense. The expected shapes are:
        * (n², n²): full edge-space covariance
        * (n, n): interpreted as per-entry variances and converted to diag(vec(Cn))
        * (n²,): interpreted as the diagonal of Cn

    Computational cost
    ------------------
    This routine explicitly forms Cu (an n²×n² matrix) and solves a dense linear system in that space.
    This scales poorly in n and is intended for small problems or validation.

    Parameters
    ----------
    A : np.ndarray
        (n×n) observed data matrix. NaNs allowed (treated as missing).
    Cn : np.ndarray
        Noise covariance in edge space (or a diagonal/variance shorthand described above).
    epsilon : float
        Regularization strength for numerical stability (adds epsilon I in edge space).
    directed : bool
        If True, use directed Cu construction; if False, use undirected swap symmetrization.
    similarity_threshold : float
        Threshold applied to PSNs: similarities below this value are set to 0.
    onlypos : bool
        If True, set negative entries of the output to 0.
    noSL : bool
        If True, set the diagonal entries of the output to 0.
    renorm : bool
        If True, rescale the output so that mean(output) == mean(filled A).

    Returns
    -------
    A_hat : np.ndarray
        (n×n) Wiener-filtered estimate of A.
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
        sim_source = calculate_profile_similarity(A_filled, threshold=similarity_threshold, mode="source")
        sim_target = calculate_profile_similarity(A_filled, threshold=similarity_threshold, mode="target")

        # Cu_{(i,j),(k,l)} ∝ sim_source[i,k] * sim_target[j,l]
        Cu = A_var * np.kron(sim_source, sim_target)

    else:
        sim = calculate_profile_similarity(A_filled, threshold=similarity_threshold, mode="mean")

        K = np.kron(sim, sim)
        K_tensor = K.reshape(n, n, n, n)
        K_swapped_tensor = K_tensor.transpose(0, 3, 2, 1)
        K_swapped = K_swapped_tensor.reshape(n2, n2)

        Cu = A_var * 0.5 * (K + K_swapped)

    # --- Normalize/interpret Cn input into an (n²×n²) matrix ---
    if Cn.ndim == 2 and Cn.shape == (n2, n2):
        Cn_mat = Cn
    elif Cn.ndim == 2 and Cn.shape == (n, n):
        # Treat as per-entry variances -> diagonal covariance in edge space.
        Cn_mat = np.diag(Cn.reshape(-1))
    elif Cn.ndim == 1 and Cn.shape[0] == n2:
        # Treat as the diagonal of Cn.
        Cn_mat = np.diag(Cn)
    else:
        raise ValueError(
            "Cn must have shape (n^2,n^2), (n,n) (interpreted as variances), or (n^2,) (diagonal). "
            f"Got shape {Cn.shape}."
        )

    # Solve for x = (Cu + Cn + epsilon I)^{-1} a without explicitly computing an inverse.
    A_edge = Cu + Cn_mat + epsilon * np.eye(n2)
    x = np.linalg.solve(A_edge, a)

    # Wiener map: u_hat = Cu x
    u_hat = Cu @ x
    U_hat = u_hat.reshape(n, n)

    # Add back the mean.
    A_hat = U_hat + A_mu

    # Optional post-processing.
    if onlypos:
        A_hat[A_hat < 0] = 0.0
    if noSL:
        np.fill_diagonal(A_hat, 0.0)
    if renorm:
        A_hat_mean = float(np.mean(A_hat))
        A_filled_mean = float(np.mean(A_filled))
        if A_hat_mean != 0:
            A_hat = A_hat * (A_filled_mean / A_hat_mean)

    return A_hat


