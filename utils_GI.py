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
    - E_sim (numpy.ndarray): NxN PSN of genes.
    - sim_threshold (float): Minimum similarity threshold.
    
    Returns:
    - float: Density of the PSN.
    """
    E_sim_pos = np.copy(E_sim)
    E_sim_pos[E_sim_pos <= sim_threshold] = 0
    return np.mean(E_sim_pos)

def go_eval(E_sim, genes, go_mapping, sim_threshold=0.2, K=10, whatrecall=0.5):
    """
    Evaluates the performance of predicting GO terms with profile similarity network,
    using precision at whatrecall in a leave-one-out manner, considering only the K nearest neighbors.
    
    Parameters:
    - E_sim (numpy.ndarray): NxN PSN of genes.
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

def get_precision_recall(E, genes, ppi):
    """
    Evaluates the performance of predicting PPIs with negative genetic interaction scores.
    
    Parameters:
    - E (numpy.ndarray): NxN matrix of genetic interaction scores.
    - genes (list): N-dim list of gene names, ordered as the matrix's rows/columns.
    - ppi (numpy.ndarray): NxN binary matrix of known PPIs.
    
    Returns:
    - precision_list (list): List of precision values at each threshold.
    - recall_list (list): List of recall values at each threshold.
    """
    E = np.nan_to_num(E, nan=np.nanmean(E))

    ppi_set = set()
    for i in range(len(genes)):
        for j in range(i+1, len(genes)):
            if ppi[i, j] == 1:
                ppi_set.add((genes[i], genes[j]))

    gi_matrix = pd.DataFrame(E, index=genes, columns=genes)
    all_pairs = [(genes[i], genes[j]) for i in range(len(genes)) for j in range(i+1, len(genes))]

    # Initialize lists to store scores and labels
    scores = []
    labels = []
    tie_breakers = []

    for pair in all_pairs:
        gene1, gene2 = pair
        # Retrieve the GI score; assume 0 if not present
        score = gi_matrix.at[gene1, gene2] if gene1 in gi_matrix.index and gene2 in gi_matrix.columns else 0
        scores.append(score)
        # Label as 1 if it's a known PPI, else 0
        label = 1 if pair in ppi_set or (gene2, gene1) in ppi_set else 0
        labels.append(label)
        tie_breakers.append(random.random())

    scores = np.array(scores)
    scores[scores >= 0] = 0
    scores = np.abs(scores)
    labels = np.array(labels)
    tie_breakers = np.array(tie_breakers)

    sorted_indices = np.lexsort((tie_breakers, -scores))
    sorted_scores = scores[sorted_indices]
    sorted_labels = labels[sorted_indices]

    # Compute precision and recall iteratively
    precision_list = []
    recall_list = []
    
    tp = 0  # True positives
    fp = 0  # False positives
    total_positives = np.sum(labels)

    for i in range(len(sorted_labels)):
        if sorted_scores[i] <= 0:
            break 
        if sorted_labels[i] == 1:
            tp += 1
        else:
            fp += 1

        precision = tp / (tp + fp)
        recall = tp / total_positives
        precision_list.append(precision)
        recall_list.append(recall)

    return precision_list, recall_list

