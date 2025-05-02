# Tingyu Zhao

from utils_WF import *



# ----- Plotting -----

def plot_gi(E, title='Genetic interaction score', save=False):
    """
    Plots a GI network.
    
    Parameters:
    - E_sim (numpy.ndarray): GI network.
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

def plot_ppi(ppi, title='PPI', save=False):
    """
    Plots a binary PPI network.
    
    Parameters:
    - ppi (numpy.ndarray): Binary PPI network (values 0 or 1).
    - title (str): Title of the plot.
    - save (bool): Whether to save the plot as a PDF.
    """
    fig, ax = plt.subplots(figsize=(4, 4))
    cmap = plt.get_cmap('binary')
    im = ax.imshow(ppi, aspect='equal', cmap=cmap, vmin=0, vmax=1)
    ax.set_title(title)
    ax.set_ylabel('Genes')
    ax.set_xlabel('Genes')
    ax.set_xticks([])
    ax.set_yticks([])
    if save:
        plt.savefig(f'Figures/{title}.pdf', bbox_inches='tight', transparent=True, dpi=1200)
    plt.show()

def plot_corr(E1, E2, title1, title2, save=False):
    """
    Plots the element-wise correlation between two datasets (e.g. two GI networks).
    
    Parameters:
    - E1 (numpy.ndarray): First dataset.
    - E2 (numpy.ndarray): Second dataset.
    - title1 (str): Label for the first dataset.
    - title2 (str): Label for the second dataset.
    - save (bool): Whether to save the plot as a PDF.
    """
    E1 = np.nan_to_num(E1, nan=np.nanmean(E1))
    E2 = np.nan_to_num(E2, nan=np.nanmean(E2))

    # Flatten the data for easier plotting
    x = E1.flatten()
    y = E2.flatten()

    # Compute correlation
    correlation, _ = pearsonr(x, y)

    # Create figure with GridSpec
    fig = plt.figure(figsize=(8, 8))
    grid = plt.GridSpec(4, 4, hspace=0.3, wspace=0.3)

    # Main scatter plot
    ax_scatter = fig.add_subplot(grid[1:4, 0:3])
    ax_scatter.scatter(x, y, alpha=0.5, color='gray')
    ax_scatter.set_xlabel(title1)
    ax_scatter.set_ylabel(title2)
    ax_scatter.legend([f'Correlation: {correlation:.2f}'], loc='upper left', frameon=True)
    ax_scatter.set_xlim(-1, 1)
    ax_scatter.set_ylim(-1, 1)
    # plot y=x line
    ax_scatter.plot([-1, 1], [-1, 1], color='black', linestyle='-', linewidth=1)
    ax_scatter.grid(True)

    # Top histogram
    ax_histx = fig.add_subplot(grid[0, 0:3], sharex=ax_scatter)
    ax_histx.hist(x, bins=30, color='gray', edgecolor='black')
    ax_histx.set_ylabel('Count')
    plt.setp(ax_histx.get_xticklabels(), visible=False)

    # Right histogram
    ax_histy = fig.add_subplot(grid[1:4, 3], sharey=ax_scatter)
    ax_histy.hist(y, bins=30, orientation='horizontal', color='gray', edgecolor='black')
    ax_histy.set_xlabel('Count')
    plt.setp(ax_histy.get_yticklabels(), visible=False)

    if save:
        plt.savefig('Figures/corr.pdf', bbox_inches='tight', transparent=True, dpi=1200)
    plt.show()



# ----- Validations -----

def psn_density(E_sim, sim_threshold=0.2):
    """
    Calculates the density of a PSN, treating similarities below threshold as 0.
    
    Parameters:
    - E_sim (numpy.ndarray): PSN of genes.
    - sim_threshold (float): Minimum similarity threshold.
    
    Returns:
    - float: Density of the PSN.
    """
    E_sim_pos = np.copy(E_sim)
    E_sim_pos[E_sim_pos <= sim_threshold] = 0
    return np.mean(E_sim_pos)

def go_eval(E_sim, genes, go_mapping, sim_threshold=0, K=10, whatrecall=0.25):
    """
    Evaluates the performance of predicting GO terms with profile similarity network,
    using precision at whatrecall in a leave-one-out manner, considering only the K nearest neighbors.
    
    Parameters:
    - E_sim (numpy.ndarray): PSN of genes.
    - genes (list): N-dim list of gene names, ordered as the E_sim's rows/columns.
    - go_mapping (dict): Dictionary mapping genes to sets of GO terms.
    - sim_threshold (float): Minimum similarity threshold for considering neighbors.
    - K (int): Number of nearest neighbors to consider.
    - whatrecall (float): Recall level at which to calculate precision.
    
    Returns:
    - float: Mean precision at 50% recall across all genes with valid evaluations.
    - float: Standard error of the mean precision.
    """
    N = E_sim.shape[0]
    assert len(genes) == N, "Genes list length must match matrix dimension."
    
    # Collect all unique GO terms from the entire go_mapping
    all_go_terms = set()
    for terms in go_mapping.values():
        all_go_terms.update(terms)
    all_go_terms = list(all_go_terms)
    
    precisions = []
    count = 0
    
    for i in range(N):
        gene = genes[i]
        true_terms = go_mapping.get(gene, set())
        if not true_terms:
            continue  # Skip genes with no GO terms
        
        # Get similarities to all other genes
        similarities = E_sim[i, :]
        
        # Filter out the test gene and genes below the similarity threshold
        valid_indices = [j for j in range(N) if j != i and similarities[j] > sim_threshold]
        valid_similarities = similarities[valid_indices]
        valid_genes = [genes[j] for j in valid_indices]
        
        # Select the top K nearest neighbors
        if len(valid_indices) > K:
            top_k_indices = np.argsort(valid_similarities)[-K:]  # Indices of top K neighbors
            top_k_genes = [valid_genes[idx] for idx in top_k_indices]
        else:
            top_k_genes = valid_genes  # Use all valid neighbors if fewer than K
        
        # Aggregate scores from the K nearest neighbors
        sum_scores = defaultdict(int)  # Use int to count occurrences
        for neighbor_gene in top_k_genes:
            terms = go_mapping.get(neighbor_gene, set())
            for term in terms:
                sum_scores[term] += 1  # Increment by 1 for each occurrence
        
        # Prepare scores and tie-breakers for sorting
        scores = np.array([sum_scores.get(term, 0) for term in all_go_terms])
        random.seed()  # Seed based on system time
        tie_breakers = np.array([random.random() for _ in all_go_terms])
        
        # Sort terms by descending score, then by random tie-breaker
        sorted_indices = np.lexsort((tie_breakers, -scores))
        sorted_terms = [all_go_terms[idx] for idx in sorted_indices]
        
        # Calculate precision at 50% recall
        tp = 0
        fp = 0
        total_positives = len(true_terms)
        required_tp = math.ceil(whatrecall * total_positives)
        found = False
        
        for term in sorted_terms:
            if term in true_terms:
                tp += 1
            else:
                fp += 1
            
            if tp >= required_tp:
                precision_at_50 = tp / (tp + fp) if (tp + fp) != 0 else 0.0
                found = True
                break
        
        precisions.append(precision_at_50 if found else 0.0)
        count += 1
    
    return np.sum(precisions) / count if count > 0 else 0.0, np.std(precisions) / np.sqrt(count) if count > 0 else 0.0

def get_precision_recall_auprc(E, gold, direction='neg'):
    """
    Compute precision-recall curve (extended to recall=1 with random predictions) and AUPRC.
    
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
    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    
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
            break  # Stop when no more high-confidence predictions
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
    np.random.shuffle(remaining_labels)  # Random order
    
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
    if recall_list[-1] < 1.0 and total_positives > 0:
        precision_list.append(tp / (tp + fp + (total_positives - tp)))
        recall_list.append(1.0)
    
    # Compute AUPRC
    auprc = auc(recall_list, precision_list)
    
    return precision_list, recall_list, auprc

def calculate_fold_enrichment(gold, E_thresh, direction='neg'):
    """
    Calculate fold enrichment of gold standard interactions in significant GI pairs.
    
    Args:
        gold (np.ndarray): Binary gold standard matrix (1 = interaction, 0 = no interaction).
        E_thresh (np.ndarray): Thresholded GI matrix (non-zero = significant, sign indicates direction).
        direction (str): 'pos' (enrichment in positive GIs) or 'neg' (enrichment in negative GIs).
    
    Returns:
        float: Fold enrichment value.
    """
    # Ensure matrices are symmetric and exclude diagonals
    gold = np.triu(gold, k=1)  # Upper triangle only (no diagonal)
    E_thresh = np.triu(E_thresh, k=1)
    
    # Total gold standard interactions and possible pairs
    total_gold_pairs = np.sum(gold)
    total_possible_pairs = gold.size - np.diag(gold).size  # Exclude diagonal
    
    if total_gold_pairs == 0:
        return 0.0  # Avoid division by zero
    
    # Direction handling (non-zero values with correct sign)
    if direction == 'neg':
        significant_mask = (E_thresh < 0)  # Negative GIs
    elif direction == 'pos':
        significant_mask = (E_thresh > 0)  # Positive GIs
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