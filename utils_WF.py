# Tingyu Zhao

import pickle
import numpy as np
import pandas as pd
import networkx as nx

import scipy.sparse as sp
from scipy.sparse import save_npz, load_npz
import scipy.stats as stats
from scipy.stats import pearsonr
from scipy.stats import rankdata

from sklearn.metrics import auc
from sklearn.utils import resample
from collections import defaultdict
import random
import math

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable



# ----- Calculations -----

def calculate_profile_similarity(matrix, threshold=-np.inf, mode ='mean'):
    """
    Calculates the node level similarity matrix (NxN) from the adjacency matrix (NxN).
    The similarity between nodes i and j is the average of row-wise and column-wise Pearson correlations.
    """
    N = matrix.shape[0]
    similarity_matrix = np.eye(N)
    
    for i in range(N):
        for j in range(i):
            # Compute row-wise correlation
            row_i = matrix[i, :]
            row_j = matrix[j, :]
            valid_mask_row = ~np.isnan(row_i) & ~np.isnan(row_j)
            
            if np.any(valid_mask_row):
                row_i_valid = row_i[valid_mask_row]
                row_j_valid = row_j[valid_mask_row]
                # Check if either row has all identical values
                if np.all(row_i_valid == row_i_valid[0]) or np.all(row_j_valid == row_j_valid[0]):
                    row_corr = np.nan
                else:
                    row_corr, _ = pearsonr(row_i_valid, row_j_valid)
            else:
                row_corr = np.nan  # No valid pairs
            
            # Compute column-wise correlation
            col_i = matrix[:, i]
            col_j = matrix[:, j]
            valid_mask_col = ~np.isnan(col_i) & ~np.isnan(col_j)
            
            if np.any(valid_mask_col):
                col_i_valid = col_i[valid_mask_col]
                col_j_valid = col_j[valid_mask_col]
                # Check if either column has all identical values
                if np.all(col_i_valid == col_i_valid[0]) or np.all(col_j_valid == col_j_valid[0]):
                    col_corr = np.nan
                else:
                    col_corr, _ = pearsonr(col_i_valid, col_j_valid)
            else:
                col_corr = np.nan  # No valid pairs

            if mode == 'mean':
                corr = np.nanmean([row_corr, col_corr])
            elif mode == 'max':
                corr = np.nanmax([row_corr, col_corr])
            elif mode == 'row':
                corr = row_corr
            elif mode == 'col':
                corr = col_corr
            else:
                raise ValueError("Invalid mode. Choose 'mean', 'max', 'row', or 'col'.")
            
            similarity_matrix[i, j] = corr
            similarity_matrix[j, i] = corr

    # Apply threshold and replace NaNs with 0
    similarity_matrix[similarity_matrix < threshold] = 0
    return np.nan_to_num(similarity_matrix)

def compute_edge_correlation_matrix_vectorized_from_sim(similarity_matrix, directed = True):
    """
    Calculates the edge-level similarity matrix (N²xN²) from the node-level similarity matrix (NxN).
    
    The edge similarity between edges AB and CD is:
        S_edge(AB, CD) = 0.5 * (S_AC * S_BD + S_AD * S_BC)
    
    Args:
        similarity_matrix (np.ndarray): NxN node similarity matrix.
        directed (bool): If True, the edges are considered directed. If False, the edges are considered undirected.
    
    Returns:
        np.ndarray: N²xN² edge similarity matrix.
    """
    n = similarity_matrix.shape[0]
    K = np.kron(similarity_matrix, similarity_matrix)  # K_{(A,B),(C,D)} = S_AC * S_BD
    
    if directed:
        return K
    
    else:
        K_tensor = K.reshape(n, n, n, n)
        K_swapped_tensor = K_tensor.transpose(0, 3, 2, 1)
        K_swapped = K_swapped_tensor.reshape(n**2, n**2)
        return 0.5 * (K + K_swapped)



# ----- Filtering -----

def filter_small_var(E_small, E_small_var, epsilon = 1e-2, mode = 'mean', directed = True, onlypos = False, noSL = False, renorm = False):
    """
    Applies the netWF algorithm directly, given an NxN adjacency matrix and an NxN edge noise variance matrix.

    Parameters:
    - E_small (numpy.ndarray): NxN adjacency matrix, potential nan values allowed.
    - E_small_var (numpy.ndarray): NxN edge noise variance matrix, potential nan values allowed.
    - epsilon (float): Regularization term for numerical stability.
    - mode (str): Method to calculate similarity. Options are 'mean', 'max', 'row', or 'col'.
    - directed (bool): Whether the edges are directed.
    - renorm (bool): Whether to renormalize the filtered matrix.
    - onlypos (bool): Whether to only keep the positive entries of the filtered matrix.
    - noSL (bool): Whether to fill the diagonal of the filtered matrix with zeros.
    """

    E_small_mean = np.nanmean(E_small)
    E_small_datavar = np.nanvar(E_small)
    E_small_var_mean = np.nanmean(E_small_var)

    E_small = np.nan_to_num(E_small, nan=E_small_mean)
    E_small_var = np.nan_to_num(E_small_var, nan=E_small_datavar + E_small_var_mean)

    E_small_sim = calculate_profile_similarity(E_small, mode=mode)
    S = (E_small_datavar) * compute_edge_correlation_matrix_vectorized_from_sim(E_small_sim, directed=directed)
    N = np.diag(E_small_var.flatten())

    E_small_dm = E_small - E_small_mean
    E_small_dm_vec = E_small_dm.flatten()
    E_small_dm_vec_filtered = S @ np.linalg.inv(S+N+ epsilon * np.eye(S.shape[0])) @ E_small_dm_vec
    E_small_dm_filtered = E_small_dm_vec_filtered.reshape(E_small.shape)
    E_small_filtered = E_small_dm_filtered + E_small_mean

    if onlypos:
        E_small_filtered[E_small_filtered < 0] = 0
    
    if noSL:
        np.fill_diagonal(E_small_filtered, 0)

    if renorm:
        E_small_filtered_norm = np.mean(E_small_filtered)
        E_small_norm = np.mean(E_small)
        E_small_filtered = E_small_filtered * E_small_norm / E_small_filtered_norm

    return E_small_filtered


def filter_small_cov(E_small, E_small_cov, epsilon = 1e-2, mode = 'mean', directed = True, onlypos = False, noSL = False, renorm = False):
    """
    Applies the netWF algorithm directly, given an NxN adjacency matrix and an N²xN² edge noise covariance matrix.

    Parameters:
    - E_small (numpy.ndarray): NxN adjacency matrix, potential nan values allowed.
    - E_small_cov (numpy.ndarray): N²xN² edge noise covariance matrix, potential nan values NOT allowed.
    - epsilon (float): Regularization term for numerical stability.
    - mode (str): Method to calculate similarity. Options are 'mean', 'max', 'row', or 'col'.
    - directed (bool): Whether the edges are directed.
    - renorm (bool): Whether to renormalize the filtered matrix.
    - onlypos (bool): Whether to only keep the positive entries of the filtered matrix.
    - noSL (bool): Whether to fill the diagonal of the filtered matrix with zeros.
    """

    E_small_mean = np.nanmean(E_small)
    E_small_datavar = np.nanvar(E_small)

    E_small = np.nan_to_num(E_small, nan=E_small_mean)

    E_small_sim = calculate_profile_similarity(E_small, mode=mode)
    S = (E_small_datavar) * compute_edge_correlation_matrix_vectorized_from_sim(E_small_sim, directed=directed)
    N = E_small_cov

    E_small_dm = E_small - E_small_mean
    E_small_dm_vec = E_small_dm.flatten()
    E_small_dm_vec_filtered = S @ np.linalg.inv(S+N+ epsilon * np.eye(S.shape[0])) @ E_small_dm_vec
    E_small_dm_filtered = E_small_dm_vec_filtered.reshape(E_small.shape)
    E_small_filtered = E_small_dm_filtered + E_small_mean

    if onlypos:
        E_small_filtered[E_small_filtered < 0] = 0

    if noSL:
        np.fill_diagonal(E_small_filtered, 0)

    if renorm:
        E_small_filtered_norm = np.mean(E_small_filtered)
        E_small_norm = np.mean(E_small)
        E_small_filtered = E_small_filtered * E_small_norm / E_small_filtered_norm

    return E_small_filtered


def filter_big_var_sss(E_big, E_big_var, p, k, epsilon = 1e-2, mode = 'mean', directed = True, onlypos = False, noSL = False, renorm = False):
    """
    Appplis the netWF algorithm via stochastic submatrix sampling (SSS), given an NxN adjacency matrix and an NxN edge noise variance matrix.

    Parameters:
    - E_big (numpy.ndarray): NxN adjacency matrix, potential nan values allowed.
    - E_big_var (numpy.ndarray): NxN edge noise variance matrix, potential nan values allowed.
    - p (float): Number of nodes in each sample.
    - k (int): Number of samples.
    - epsilon (float): Regularization term for numerical stability.
    - mode (str): Method to calculate similarity. Options are 'mean', 'max', 'row', or 'col'.
    - directed (bool): Whether the edges are directed.
    - renorm (bool): Whether to renormalize the filtered matrix.
    - onlypos (bool): Whether to only keep the positive entries of the filtered matrix.
    - noSL (bool): Whether to fill the diagonal of the filtered matrix with zeros.
    """

    n = E_big.shape[0]
    accumulator = np.zeros((n, n), dtype=np.float64)
    count = np.zeros((n, n), dtype=np.int64)
    
    for _ in range(k):
        # print progress
        print(f'Processing sample {_+1}/{k}...')

        nodes_sampled = np.random.choice(n, p, replace=False)

        E_small = E_big[np.ix_(nodes_sampled, nodes_sampled)]
        E_small_var = E_big_var[np.ix_(nodes_sampled, nodes_sampled)]

        E_small_filtered = filter_small_var(E_small, E_small_var, epsilon=epsilon, mode=mode, directed=directed,
                                            renorm=renorm, onlypos=onlypos, noSL=noSL)

        rows_grid, cols_grid = np.meshgrid(nodes_sampled, nodes_sampled, indexing='ij')
        accumulator[rows_grid, cols_grid] += E_small_filtered
        count[rows_grid, cols_grid] += 1
    
    # Compute the mean, setting entries with zero count to 0
    E_filtered = np.zeros_like(accumulator)
    mask = count != 0
    E_filtered[mask] = accumulator[mask] / count[mask]

    print(f'{np.sum(mask) / n**2 * 100:.2f}% entries sampled at least once')

    return E_filtered


def filter_big_var_dss(E_big, E_big_var, p, epsilon=1e-2, mode = 'mean', directed = True, onlypos = False, noSL = False, renorm = False):
    """
    Applies the netWF algorithm via deterministic submatrix sampling (DSS), given an NxN adjacency matrix and an NxN edge noise variance matrix.
    Instead of randomly sampling nodes, for each node we deterministically sample a fixed amount of the most similar rows and columns based on a profile similarity.

    Parameters:
    - E_big (numpy.ndarray): NxN adjacency matrix.
    - E_big_var (numpy.ndarray): NxN edge noise variance matrix.
    - p (int): Number of most similar nodes to sample for each node.
    - epsilon (float): Regularization term for numerical stability.
    - mode (str): Method to calculate similarity. Options are 'mean', 'max', 'row', or 'col'.
    - directed (bool): Whether the edges are directed.
    - renorm (bool): Whether to renormalize the filtered matrix.
    - onlypos (bool): Whether to only keep the positive entries of the filtered matrix.
    - noSL (bool): Whether to fill the diagonal of the filtered matrix with zeros.

    Returns:
    - E_filtered (numpy.ndarray): The filtered NxN matrix.
    """

    n = E_big.shape[0]
    accumulator = np.zeros((n, n), dtype=np.float64)
    count = np.zeros((n, n), dtype=np.int64)
    
    # Calculate the profile similarity matrix for E_big
    similarity = calculate_profile_similarity(E_big, mode=mode)
    
    # Loop over every node: for each node, select the p most similar nodes (including itself)
    for i in range(n):
        # print progress
        print(f'Processing node {i+1}/{n}...')

        # Get indices of the p most similar nodes for node i (sorted in descending order)
        neighbors = np.argsort(similarity[i])[::-1][:p]
        
        E_small = E_big[np.ix_(neighbors, neighbors)]
        E_small_var = E_big_var[np.ix_(neighbors, neighbors)]
        
        E_small_filtered = filter_small_var(E_small, E_small_var, epsilon=epsilon, mode=mode, directed=directed,
                                            renorm=renorm, onlypos=onlypos, noSL=noSL)
        
        # Update the accumulator and count matrices at the positions corresponding to the chosen neighbors
        rows_grid, cols_grid = np.meshgrid(neighbors, neighbors, indexing='ij')
        accumulator[rows_grid, cols_grid] += E_small_filtered
        count[rows_grid, cols_grid] += 1
    
    # Compute the mean, setting entries with zero count to 0
    E_filtered = np.zeros_like(accumulator)
    mask = count != 0
    E_filtered[mask] = accumulator[mask] / count[mask]
    
    print(f'{np.sum(mask) / n**2 * 100:.2f}% entries sampled at least once')
    
    return E_filtered

