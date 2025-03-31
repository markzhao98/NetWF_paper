# Tingyu Zhao

from utils_WF import *



# ----- Plotting -----

def transform_weight_general(w, step=0.2, scale = 0.4):
    if w < step:
        return 0
    return scale * round(w // step * step, 1)

def plot_year(G, pos, title='Enron Corpus Network', save=False):
    plt.figure(figsize=(8, 8))
    weights = [G[u][v]['weight'] for u, v in G.edges()]
    widths = [transform_weight_general(w) for w in weights]
    node_sizes = [5*sum([d['weight'] for u, v, d in G.edges(data=True) if u == node or v == node]) for node in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color="wheat", alpha=0.8)
    nx.draw_networkx_edges(G, pos, edge_color="darkslateblue", 
                           connectionstyle='arc3,rad=0.25', 
                           arrowstyle='->', 
                           width=widths, alpha=0.6)
    ax = plt.gca()
    ax.axis("off")
    ax.set_frame_on(False)
    ax.set_title(title)
    if save:
        plt.savefig(f'Figures/{title}.pdf', bbox_inches='tight', transparent=True, dpi=1200)
    plt.show()

def plot_months(G_months, pos, title='Monthly Networks', save=False):
    plt.figure(figsize = (24, 32))
    for month in range(1, 13):
        G_month = G_months[month-1]
        weights_month = [G_month[u][v]['weight'] for u, v in G_month.edges()]
        weights_month = [transform_weight_general(w) for w in weights_month]
        edges_to_draw = [(u, v, w) for (u, v), w in zip(G_month.edges(), weights_month) if w > 0]
        node_sizes_month = [5*sum([d['weight'] for u, v, d in G_month.edges(data=True) if u == node or v == node]) for node in G_month.nodes()]
        plt.subplot(4,3,month)
        nx.draw_networkx_nodes(G_month, pos, node_size=node_sizes_month, node_color="wheat", alpha=0.8)
        nx.draw_networkx_edges(G_month, pos, edgelist=[(u, v) for u, v, _ in edges_to_draw], 
                               edge_color="darkslateblue", 
                               connectionstyle='arc3,rad=0.25', 
                               arrowstyle='->', 
                               width=[w for _, _, w in edges_to_draw], 
                               alpha=0.6)
        plt.title(f"{pd.to_datetime(f'2001-{month:02d}-01').strftime('%B')}", fontsize=60)
        ax = plt.gca()
        ax.axis("off")  # Turn off the axis
        ax.set_frame_on(False)  # Remove the frame entirely
    if save:
        plt.savefig(f'Figures/{title}.pdf', bbox_inches='tight', transparent=True, dpi=1200)
    plt.show()



# ----- Validations -----

def MSE(A_true, A_data):
    return np.sum((A_true - A_data) ** 2) / A_true.size

def MSE_months(A_months, A_total, save=False):
    MSE_list = []
    for month in range(1, 13):
        A_month = A_months[month-1]
        MSE_list.append(MSE(A_total, A_month))
    if save:
        MSE_df = pd.DataFrame({'Month': range(1, 13), 'MSE': MSE_list})
        MSE_df.to_csv('Enron-data/results/MSE_months.csv', index=False)
    return MSE_list

def rank_with_random_ties(row):
    """Rank a row with random tie-breaking"""
    # Get random permutation of indices
    permuted_indices = np.random.permutation(len(row))
    # Sort these randomly permuted indices by their actual values
    sorted_indices = sorted(permuted_indices, key=lambda i: row[i])
    # Create array to store ranks
    ranks = np.empty_like(row)
    # Assign ranks based on the randomly-tie-broken order
    ranks[sorted_indices] = np.arange(1, len(row)+1)
    return ranks

def MSE_rank(A_true, A_data):
    A_true_ranked = np.apply_along_axis(rank_with_random_ties, 1, A_true)
    A_data_ranked = np.apply_along_axis(rank_with_random_ties, 1, A_data)
    return np.mean((A_true_ranked - A_data_ranked) ** 2)

def MSE_rank_months(A_months, A_total, save=False):
    MSE_list = []
    for month in range(1, 13):
        A_month = A_months[month-1]
        MSE_list.append(MSE_rank(A_total, A_month))
    if save:
        MSE_df = pd.DataFrame({'Month': range(1, 13), 'MSE': MSE_list})
        MSE_df.to_csv('Enron-data/results/MSE_rank_months.csv', index=False)
    return MSE_list