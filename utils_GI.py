# Tingyu Zhao

from utils_WF import *


# ----- Plotting -----

def plot_gi(E, title='Genetic interaction score', save=False):
    """
    Plots a GI network.
    
    Parameters:
    - E (numpy.ndarray): GI network.
    - title (str): Title of the plot.
    - save (bool): Whether to save the plot as a PDF.
    """
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(E, aspect='equal', cmap='seismic', vmin=-1, vmax=1)
    ax.set_title(title)
    ax.set_ylabel('Query')
    ax.set_xlabel('Array')
    ax.set_xticks([])
    ax.set_yticks([])
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)  # Adjust size and padding
    cbar = fig.colorbar(im, cax=cax)
    if save:
        plt.savefig(f'Figures/{title}.pdf', bbox_inches='tight', transparent=True, dpi=1200)
    plt.show()

def plot_sim(E_sim, title='Profile Similarity', save=False):
    """
    Plots a PSN.
    
    Parameters:
    - E_sim (numpy.ndarray): PSN.
    - title (str): Title of the plot.
    - save (bool): Whether to save the plot as a PDF.
    """
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(E_sim, aspect='equal', cmap='Purples', vmin=0, vmax=1)
    ax.set_title(title)
    ax.set_ylabel('Query')
    ax.set_xlabel('Array')
    ax.set_xticks([])
    ax.set_yticks([])
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    fig.colorbar(im, cax=cax)
    if save:
        plt.savefig(f'Figures/{title}.pdf', bbox_inches='tight', transparent=True, dpi=1200)
    plt.show()

def get_precision_recall_auprc(E, gold, direction='neg'):
    """
    Computes precision-recall curve (extended to recall=1 with random predictions) and AUPRC.
    
    Args:
        E (np.ndarray): GI score matrix or similarity matrix.
        gold (np.ndarray): Binary gold standard matrix (1 = interaction exists, 0 otherwise).
        direction (str): 'pos' (most positive first) or 'neg' (most negative first).
    
    Returns:
        tuple: (precision_list, recall_list, auprc)
    """
    E = np.nan_to_num(E, nan=np.nanmean(E))
    n = E.shape[0]  # Assuming E and gold are square matrices of the same size
    
    # Get all possible index pairs (excluding self-interactions)
    all_pairs = [(i, j) for i in range(n) for j in range(n) if i != j]
    
    # Extract scores and labels
    scores = []
    labels = []
    tie_breakers = []
    
    for i, j in all_pairs:
        scores.append(E[i, j])
        labels.append(gold[i, j] or gold[j, i])  # Check both directions
        tie_breakers.append(random.random())  # For breaking ties randomly
    
    scores = np.array(scores)
    labels = np.array(labels)
    tie_breakers = np.array(tie_breakers)
    
    # Adjust scores based on direction
    if direction == 'neg':
        scores = -scores  # Prioritize most negatives
    elif direction == 'pos':
        scores = scores  # Prioritize most positives
    else:
        raise ValueError("direction must be 'pos' or 'neg'")
    
    # Sort in descending order (most extreme first)
    sorted_indices = np.lexsort((tie_breakers, -scores))
    sorted_scores = scores[sorted_indices]
    sorted_labels = labels[sorted_indices]
    
    # Compute precision and recall
    precision_list = []
    recall_list = []
    tp = 0  # True positives
    fp = 0  # False positives
    total_positives = np.sum(labels)
    
    # Step 1: Process ranked predictions (where sorted_scores > 0)
    for i in range(len(sorted_labels)):
        if sorted_scores[i] <= 0:
            break
        if sorted_labels[i] == 1:
            tp += 1
        else:
            fp += 1
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / total_positives if total_positives > 0 else 0
        precision_list.append(precision)
        recall_list.append(recall)
    
    # Step 2: Fill remaining predictions randomly (to reach recall=1)
    remaining_indices = np.where(sorted_scores <= 0)[0]
    remaining_labels = sorted_labels[remaining_indices]
    np.random.shuffle(remaining_labels)
    
    for label in remaining_labels:
        if label == 1:
            tp += 1
        else:
            fp += 1
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / total_positives if total_positives > 0 else 0
        precision_list.append(precision)
        recall_list.append(recall)
    
    # Ensure we end at recall=1 (floating point safety)
    if len(recall_list) > 0 and recall_list[-1] < 1.0 and total_positives > 0:
        precision_list.append(tp / (tp + fp + (total_positives - tp)))
        recall_list.append(1.0)
    
    # Compute AUPRC
    auprc = auc(recall_list, precision_list)
    
    return precision_list, recall_list, auprc

def calculate_fold_enrichment(gold, E_thresh, direction='neg'):
    """
    Calculates fold enrichment of gold standard interactions in significant GI pairs.
    
    Args:
        gold (np.ndarray): Binary gold standard matrix (1 = interaction, 0 = no interaction).
        E_thresh (np.ndarray): Thresholded GI matrix (non-zero = significant, sign indicates direction).
        direction (str): 'pos' (enrichment in positive GIs) or 'neg' (enrichment in negative GIs).
    
    Returns:
        float: Fold enrichment value.
    """
    n = gold.shape[0]
    offdiag = ~np.eye(n, dtype=bool)  # exclude diagonal

    # Total gold standard interactions and possible pairs (over all i != j)
    total_gold_pairs = np.sum(gold[offdiag])
    total_possible_pairs = np.sum(offdiag)

    if total_gold_pairs == 0:
        return 0.0  # Avoid division by zero

    # Direction handling (non-zero values with correct sign)
    if direction == 'neg':
        significant_mask = (E_thresh < 0) & offdiag
    elif direction == 'pos':
        significant_mask = (E_thresh > 0) & offdiag
    else:
        raise ValueError("direction must be 'pos' or 'neg'")

    # Gold interactions in significant GI pairs
    gold_in_sig = np.sum(gold[significant_mask])
    sig_pairs = np.sum(significant_mask)

    if sig_pairs == 0:
        return 0.0  # No significant pairs to evaluate

    # Densities
    density_in_sig = gold_in_sig / sig_pairs
    density_overall = total_gold_pairs / total_possible_pairs

    # Fold enrichment
    fold_enrichment = density_in_sig / density_overall

    return fold_enrichment