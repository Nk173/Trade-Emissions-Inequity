import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import pickle
import matplotlib.colors as mcolors
from collections import defaultdict
import pycountry_convert as pc
import copy
from scipy.stats import lognorm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mutual_info_score, normalized_mutual_info_score
from scipy.stats import pearsonr, spearmanr
from collections import Counter
from matplotlib.colors import PowerNorm, TwoSlopeNorm
import matplotlib as mpl
from io import BytesIO
import base64
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.gridspec as gridspec
import trophic_tools as ta  # Requires your re-uploaded module
from collections import defaultdict

def build_all_networks_trade_volume(df, years, weight_col):
    graphs = {}
    for year in years:
        df_year = df[df['year'] == year].copy()
        df_year['node_pair'] = df_year.apply(lambda row: tuple(sorted([row['Source'], row['Target']])), axis=1)
        df_sym = df_year.groupby('node_pair')[weight_col].sum().reset_index()
        G = nx.Graph()
        for row in df_sym.itertuples(index=False):
            source, target = row.node_pair
            weight = getattr(row, weight_col)
            if source != target:
                G.add_edge(source, target, weight=weight)
        graphs[year] = G
    return graphs

import networkx as nx

def build_all_directed_networks(df, years, weight_col):
    graphs = {}
    for year in years:
        df_year = df[df['year'] == year].copy()
        G = nx.DiGraph()
        for row in df_year.itertuples(index=False):
            source = row.Source
            target = row.Target
            weight = getattr(row, weight_col)
            if source != target:
                # If edge exists, accumulate the weight
                if G.has_edge(source, target):
                    G[source][target]['weight'] += weight
                else:
                    G.add_edge(source, target, weight=weight)
        graphs[year] = G
    return graphs

def enforce_min_node_distance(pos, min_dist=0.1, iterations=10):
    """
    Adjusts manually assigned positions so that all node pairs are at least min_dist apart.
    
    Parameters:
        pos (dict): Node -> (x, y) positions
        min_dist (float): Minimum distance between any two nodes
        iterations (int): How many times to iterate the repulsion
    
    Returns:
        dict: Adjusted positions
    """
    import numpy as np

    pos = {k: np.array(v, dtype=float) for k, v in pos.items()}
    keys = list(pos.keys())

    for _ in range(iterations):
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                u, v = keys[i], keys[j]
                delta = pos[v] - pos[u]
                distance = np.linalg.norm(delta)
                if distance < min_dist:
                    if distance == 0:
                        delta = np.random.rand(2) - 0.5
                        distance = np.linalg.norm(delta)
                    shift = (min_dist - distance) * delta / distance
                    pos[v] += shift / 2
                    pos[u] -= shift / 2
    return pos


def plot_enhanced_network(
    G1, pos,
    G2=None,
    year1=1995,
    year2=2019,
    threshold=0.01,
    weight_col='weight',
    node_color='white',
    cmap='Reds',
    scale_edge_alpha=True,
    min_alpha=0.2,
    max_alpha=1.0,
    scale_node_size=True,
    min_node_size=300,
    max_node_size=1000,
    figsize=(18, 9),
    cbar_label='Share of Trade Volume'
):
    def prepare_data(G):
        Gf = nx.Graph()
        for u, v, d in G.edges(data=True):
            if d[weight_col] >= threshold:
                Gf.add_edge(u, v, weight=d[weight_col])
        return Gf

    Gf1 = prepare_data(G1)
    Gf2 = prepare_data(G2) if G2 is not None else None

    all_weights = [d[weight_col] for Gf in [Gf1, Gf2] if Gf is not None for _, _, d in Gf.edges(data=True)]
    if not all_weights:
        print("No edges above threshold.")
        return

    norm = PowerNorm(gamma=0.4, vmin=min(all_weights), vmax=max(all_weights))
    cmap_obj = plt.cm.get_cmap(cmap)

    if G2 is not None:
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        graphs = [(Gf1, axes[0], f"Trade Network ({year1})"), (Gf2, axes[1], f"Trade Network ({year2})")]
    else:
        fig, ax = plt.subplots(1, 1, figsize=(figsize[0] // 2, figsize[1]))
        graphs = [(Gf1, ax, f"Trade Network ({year1})")]

    for Gf, ax, title in graphs:
        weights = [Gf[u][v][weight_col] for u, v in Gf.edges()]
        edge_colors = [cmap_obj(norm(w)) for w in weights]

        top_percent = 0.01
        num_top = int(len(weights) * top_percent)
        sorted_weights = sorted(weights, reverse=True)
        threshold_top = sorted_weights[num_top - 1] if num_top > 0 else max(weights)

        edge_alphas = [
            max_alpha if w >= threshold_top
            else min_alpha + (w - min(weights)) / (threshold_top - min(weights) + 1e-9) * (max_alpha - min_alpha)
            for w in weights
        ]

        if scale_node_size:
            strength = dict(Gf.degree(weight=weight_col))
            max_s, min_s = max(strength.values()), min(strength.values())
            node_sizes = [
                min_node_size + (strength[n] - min_s) / (max_s - min_s + 1e-9) * (max_node_size - min_node_size)
                for n in Gf.nodes()
            ]
        else:
            node_sizes = [300] * len(Gf.nodes())

        edge_attrs = [(u, v, Gf[u][v][weight_col], edge_colors[i], edge_alphas[i])
                      for i, (u, v) in enumerate(Gf.edges())]
        edge_attrs.sort(key=lambda x: x[2])  # sort edges by weight

        for u, v, w, color, alpha in edge_attrs:
            nx.draw_networkx_edges(Gf, pos, edgelist=[(u, v)], edge_color=[color], alpha=alpha, width=2, ax=ax)

        nx.draw_networkx_nodes(Gf, pos, node_size=node_sizes, node_color=node_color,
                               edgecolors='black', linewidths=1.0, ax=ax)
        nx.draw_networkx_labels(Gf, pos, font_size=9, ax=ax)

#         ax.set_title(title, fontsize=14)
        ax.axis('off')

    # Create colorbar between subplots
    cbar_ax = fig.add_axes([0.0, 0.6, 0.015, 0.2]) if G2 else fig.add_axes([0.0, 0.6, 0.015, 0.2])
    sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
    sm.set_array(all_weights)
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.ax.set_title(cbar_label, fontsize=10, pad=10, loc='center')
    plt.show()
    
def create_combined_plots(
    # Parameters for plot_enhanced_network
    G1, pos, G2=None, year1=1995, year2=2019, threshold_enhanced=0.01,
    weight_col='weight', node_color='white', cmap_enhanced='Reds',
    scale_edge_alpha=True, min_alpha_enhanced=0.2, max_alpha_enhanced=1.0,
    scale_node_size=True, min_node_size=500, max_node_size=1400,
    cbar_label_enhanced=r'$\tilde{w}^T_{ij}$',
    
    # Parameters for plot_difference_network_powernorm
    G_T=None, G_C=None, threshold_diff=0.0, cmap_diff='RdBu',
    gamma=10, min_alpha_diff=0.2, max_alpha_diff=1.0,
    cbar_label_diff='Trade − GHG',
    darken_diff_edges=True, diff_alpha_multiplier=1.5,
    
    # Combined figure parameters
    figsize=(20, 8)
):
    """
    Creates a combined figure with both network plots as subplots.
    Left subplot: Enhanced network plot
    Right subplot: Same network structure but with difference-based edge colors
    """
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # ========== PREPARE DATA FOR BOTH PLOTS ==========
    
    def prepare_data(G, threshold):
        Gf = nx.Graph()
        for u, v, d in G.edges(data=True):
            if d[weight_col] >= threshold:
                Gf.add_edge(u, v, weight=d[weight_col])
        return Gf

    Gf1 = prepare_data(G1, threshold_enhanced)
    Gf2 = prepare_data(G2, threshold_enhanced) if G2 is not None else None

    all_weights = [d[weight_col] for Gf in [Gf1, Gf2] if Gf is not None for _, _, d in Gf.edges(data=True)]
    
    # Use the second network (G2) if available, otherwise use G1
    Gf_plot = Gf2 if Gf2 is not None else Gf1
    year_plot = year2 if Gf2 is not None else year1
    
    # Create difference values for the same edges as in the enhanced network
    difference_values = {}
    if G_T is not None and G_C is not None:
        for u, v in Gf_plot.edges():
            w_T = G_T[u][v][weight_col] if G_T.has_edge(u, v) else 0.0
            w_C = G_C[u][v][weight_col] if G_C.has_edge(u, v) else 0.0
            difference_values[(u, v)] = w_T - w_C
    
    if not all_weights:
        print("No edges above threshold.")
        return fig
    
    # ========== SHARED NETWORK STRUCTURE CALCULATIONS ==========
    
    weights = [Gf_plot[u][v][weight_col] for u, v in Gf_plot.edges()]
    
    # Calculate edge alphas based on original weights (same for both plots)
    top_percent = 0.01
    num_top = int(len(weights) * top_percent)
    sorted_weights = sorted(weights, reverse=True)
    threshold_top = sorted_weights[num_top - 1] if num_top > 0 else max(weights)

    edge_alphas = [
        max_alpha_enhanced if w >= threshold_top
        else min_alpha_enhanced + (w - min(weights)) / (threshold_top - min(weights) + 1e-9) * (max_alpha_enhanced - min_alpha_enhanced)
        for w in weights
    ]

    # Calculate node sizes (same for both plots)
    if scale_node_size:
        strength = dict(Gf_plot.degree(weight=weight_col))
        max_s, min_s = max(strength.values()), min(strength.values())
        node_sizes = [
            min_node_size + (strength[n] - min_s) / (max_s - min_s + 1e-9) * (max_node_size - min_node_size)
            for n in Gf_plot.nodes()
        ]
    else:
        node_sizes = [300] * len(Gf_plot.nodes())
    
    # ========== LEFT SUBPLOT: Enhanced Network ==========
    
    norm_enhanced = PowerNorm(gamma=0.4, vmin=min(all_weights), vmax=max(all_weights))
    cmap_obj_enhanced = plt.cm.get_cmap(cmap_enhanced)
    
    edge_colors_enhanced = [cmap_obj_enhanced(norm_enhanced(w)) for w in weights]

    edge_attrs_enhanced = [(u, v, Gf_plot[u][v][weight_col], edge_colors_enhanced[i], edge_alphas[i])
                          for i, (u, v) in enumerate(Gf_plot.edges())]
    edge_attrs_enhanced.sort(key=lambda x: x[2])

    for u, v, w, color, alpha in edge_attrs_enhanced:
        nx.draw_networkx_edges(Gf_plot, pos, edgelist=[(u, v)], edge_color=[color], alpha=alpha, width=5, ax=ax1)

    nx.draw_networkx_nodes(Gf_plot, pos, node_size=node_sizes, node_color=node_color,
                           edgecolors='black', linewidths=1.0, ax=ax1)
    nx.draw_networkx_labels(Gf_plot, pos, font_size=12, ax=ax1)

#     ax1.set_title(f"Enhanced Network ({year_plot})", fontsize=14)
    ax1.text(0.0, 0.75, "(a)", transform=ax1.transAxes,
              fontsize=25, fontweight='bold', va='top', ha='left')
    
    ax1.axis('off')
    
    # ========== RIGHT SUBPLOT: Same Network with Difference Colors ==========
    
    if difference_values:
        # Get difference values for the edges in the same order as the enhanced network
        diff_values = [difference_values.get((u, v), 0) for u, v in Gf_plot.edges()]
        
        # Color normalization for differences (centered at 0)
        if diff_values:
            vmax = np.max(np.abs(diff_values))
            norm_diff = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
            cmap_obj_diff = plt.cm.get_cmap(cmap_diff)
            
            edge_colors_diff = [cmap_obj_diff(norm_diff(d)) for d in diff_values]
            
            # Make edges darker for difference network
            if darken_diff_edges:
                edge_alphas_diff = [min(1.0, alpha * diff_alpha_multiplier) for alpha in edge_alphas]
            else:
                edge_alphas_diff = edge_alphas
            
            # Create edge attributes with difference colors but same structure
            edge_attrs_diff = [(u, v, Gf_plot[u][v][weight_col], edge_colors_diff[i], edge_alphas_diff[i])
                              for i, (u, v) in enumerate(Gf_plot.edges())]
            
            edge_attrs_diff.sort(key=lambda x: x[2])  # Sort by original weight to match enhanced network

            for u, v, w, color, alpha in edge_attrs_diff:
                nx.draw_networkx_edges(Gf_plot, pos, edgelist=[(u, v)], edge_color=[color], alpha=alpha, width=5, ax=ax2)

            nx.draw_networkx_nodes(Gf_plot, pos, node_size=node_sizes, node_color=node_color,
                                   edgecolors='black', linewidths=1.0, ax=ax2)
            nx.draw_networkx_labels(Gf_plot, pos, font_size=12, ax=ax2)
    else:
        # Fallback: plot same as enhanced network if no difference data
        for u, v, w, color, alpha in edge_attrs_enhanced:
            nx.draw_networkx_edges(Gf_plot, pos, edgelist=[(u, v)], edge_color=[color], alpha=alpha, width=5, ax=ax2)

        nx.draw_networkx_nodes(Gf_plot, pos, node_size=node_sizes, node_color=node_color,
                               edgecolors='black', linewidths=1.0, ax=ax2)
        nx.draw_networkx_labels(Gf_plot, pos, font_size=12, ax=ax2)

#     ax2.set_title("Same Network with Difference Colors", fontsize=14)
    ax2.axis('off')
    ax2.text(0.0, 0.75, "(b)", transform=ax2.transAxes,
              fontsize=25, fontweight='bold', va='top', ha='left')
    
    # ========== COLORBARS ==========
    
    # Colorbar for enhanced network (left)
    cbar_ax1 = fig.add_axes([0.02, 0.8, 0.015, 0.2])
    sm1 = plt.cm.ScalarMappable(cmap=cmap_obj_enhanced, norm=norm_enhanced)
    sm1.set_array(all_weights)
    cbar1 = fig.colorbar(sm1, cax=cbar_ax1)
    cbar1.ax.set_title(cbar_label_enhanced, fontsize=15, pad=10, loc='center')
    cbar1.ax.tick_params(labelsize=12)
    # Colorbar for difference network (right)
    if difference_values and diff_values:
        cbar_ax2 = fig.add_axes([0.52, 0.8, 0.015, 0.2])
        sm2 = mpl.cm.ScalarMappable(cmap=cmap_obj_diff, norm=norm_diff)
        sm2.set_array(diff_values)
        cbar2 = fig.colorbar(sm2, cax=cbar_ax2)
        cbar2.ax.set_title(cbar_label_diff, fontsize=15, pad=10, loc='center')
        cbar2.ax.tick_params(labelsize=12)
    
    plt.tight_layout()
    plt.show()
    
    return fig

def compute_joint_prob(x_bins, y_bins):
    """Compute joint and marginal probabilities."""
    n = len(x_bins)
    joint_counts = Counter(zip(x_bins, y_bins))
    px = Counter(x_bins)
    py = Counter(y_bins)

    pxy = {k: v / n for k, v in joint_counts.items()}
    px = {k: v / n for k, v in px.items()}
    py = {k: v / n for k, v in py.items()}
    
    return pxy, px, py

def compute_custom_nmi_from_scratch(x_bins, y_bins):
    """Compute full NMI without sklearn functions."""
    pxy, px, py = compute_joint_prob(x_bins, y_bins)

    mi = 0.0
    for (x, y), p_xy in pxy.items():
        p_x = px[x]
        p_y = py[y]
        mi += p_xy * np.log(p_xy / (p_x * p_y ) )  # add epsilon for numerical safety

    Hx = -sum(p * np.log(p ) for p in px.values())
    Hy = -sum(p * np.log(p ) for p in py.values())

    nmi = 2 * mi / (Hx + Hy )
    return nmi, pxy, px, py, Hx, Hy, mi

def bin_weights_with_threshold(w_trade, w_ghg, threshold=1e-9, num_bins=100):
    # Filter out very small weights
    mask = (w_trade > threshold) & (w_ghg > threshold)
    w_trade_filtered = w_trade[mask]
    w_ghg_filtered = w_ghg[mask]

    # Apply log transform
    log_trade = np.log(w_trade_filtered)
    log_ghg = np.log(w_ghg_filtered)

    # Compute common range
    all_logs = np.concatenate([log_trade, log_ghg])
    vmin, vmax = np.min(all_logs), np.max(all_logs)

    # Bin edges and digitize
    bin_edges = np.linspace(vmin, vmax, num_bins + 1)
    trade_bins = pd.cut(log_trade, bins=bin_edges, labels=False, include_lowest=True)
    ghg_bins = pd.cut(log_ghg, bins=bin_edges, labels=False, include_lowest=True)

    return trade_bins, ghg_bins, bin_edges

def clean_weights(W):
    weights = W.flatten()
    return weights[weights > 0]

def compute_joint_prob(x_bins, y_bins):
    n = len(x_bins)
    joint_counts = Counter(zip(x_bins, y_bins))
    px = Counter(x_bins)
    py = Counter(y_bins)

    pxy = {k: v / n for k, v in joint_counts.items()}
    px = {k: v / n for k, v in px.items()}
    py = {k: v / n for k, v in py.items()}

    return pxy, px, py

def compute_custom_nmi_from_scratch(x_bins, y_bins):
    pxy, px, py = compute_joint_prob(x_bins, y_bins)
    mi = sum(pxy[(x, y)] * np.log((pxy[(x, y)]) / (px[x] * py[y])) for x, y in pxy)
    Hx = -sum(p * np.log(p) for p in px.values())
    Hy = -sum(p * np.log(p) for p in py.values())
    nmi = 2 * mi / (Hx + Hy)
    return nmi

def bin_weights_with_threshold(w_trade, w_ghg, threshold=1e-9, num_bins=50):
    mask = (w_trade > threshold) & (w_ghg > threshold)
    w_trade_filtered = w_trade[mask]
    w_ghg_filtered = w_ghg[mask]
    log_trade = np.log(w_trade_filtered)
    log_ghg = np.log(w_ghg_filtered)
    all_logs = np.concatenate([log_trade, log_ghg])
    vmin, vmax = np.min(all_logs), np.max(all_logs)
    bin_edges = np.linspace(vmin, vmax, num_bins + 1)
    trade_bins = pd.cut(log_trade, bins=bin_edges, labels=False, include_lowest=True)
    ghg_bins = pd.cut(log_ghg, bins=bin_edges, labels=False, include_lowest=True)
    return trade_bins, ghg_bins

def custom_nmi_wrapper(w1, w2, num_bins=50):
    trade_bins, ghg_bins = bin_weights_with_threshold(w1, w2, threshold=1e-15, num_bins=num_bins)
    return compute_custom_nmi_from_scratch(trade_bins, ghg_bins)

def pearson_wrapper(w1, w2, **kwargs):
    return pearsonr(w1, w2)[0]

def compute_cumulative_metric(G1, G2, metric_func, num_points=50, threshold=1e-15, num_bins=50):
    edges = list(G1.edges())
    weights_1 = nx.get_edge_attributes(G1, 'weight')
    weights_2 = nx.get_edge_attributes(G2, 'weight')

    combined_edges = []
    for (u, v) in edges:
        w1 = weights_1.get((u, v), 0)
        w2 = weights_2.get((u, v), 0)
        avg = (w1 + w2) / 2
        combined_edges.append(((u, v), w1, w2, avg))

    combined_edges_sorted = sorted(combined_edges, key=lambda x: x[3])

    fractions, values = [], []
    for k in np.linspace(10, len(combined_edges_sorted), num_points, dtype=int):
        subset = combined_edges_sorted[:k]
        w1_arr = np.array([w1 for (_, w1, _, _) in subset])
        w2_arr = np.array([w2 for (_, _, w2, _) in subset])
        valid_mask = (w1_arr > threshold) & (w2_arr > threshold)
        if np.count_nonzero(valid_mask) < 10:
            continue
        w1_valid = w1_arr[valid_mask]
        w2_valid = w2_arr[valid_mask]
        try:
            val = metric_func(w1_valid, w2_valid, num_bins=num_bins)
            values.append(val)
            fractions.append(k / len(combined_edges_sorted))
        except Exception:
            continue
    return fractions, values

#####################################################################################################

def reorder_matrix_by_trade_volume(matrix):
    """
    Reorder rows and columns using the same order based on total strength
    (export + import) per country. Assumes a square matrix.

    Returns:
        reordered_matrix, new_index_order
    """
    assert matrix.shape[0] == matrix.shape[1], "Matrix must be square for identity alignment."

    total_strength = matrix.sum(axis=1) + matrix.sum(axis=0)
    order = np.argsort(-total_strength)  # descending order

    reordered = matrix[np.ix_(order, order)]
    return reordered, order

def get_adjacency_matrix(G):
    nodes = sorted(G.nodes())
    A = nx.to_numpy_array(G, nodelist=nodes, weight='weight')
    return A, nodes

def fit_gamma(SiSj, Wij, min_weight=1e-6, min_sisj=1e-6, max_weight=1, max_sisj=1):
    """
    Fits log(w_ij) = gamma * log(s_i * s_j), filtering out small values.
    
    Parameters:
        SiSj (array): s_i * s_j
        Wij (array): w_ij
        min_weight (float): minimum w_ij to include
        min_sisj (float): minimum s_i * s_j to include
    """
    mask = (SiSj > min_sisj) & (Wij > min_weight) & (SiSj < max_sisj) & (Wij < max_weight)
    if np.sum(mask) < 10:
        raise ValueError("Too few data points after filtering.")
    
    x = np.log(SiSj[mask]).reshape(-1, 1)
    y = np.log(Wij[mask])
    
    model = LinearRegression().fit(x, y)
    gamma = model.coef_[0]
    return gamma, model

def plot_si_sj_vs_wij(G, year, ax_main, gamma, color=None):
    A, nodes = get_adjacency_matrix(G)
    s = A.sum(axis=1)  # strength
    SiSj = []
    Wij = []

    for i in range(len(nodes)):
        for j in range(i+1, len(nodes)):
            if A[i, j] > 0:
                SiSj.append(s[i] * s[j])
                Wij.append(A[i, j])

    SiSj = np.array(SiSj)
    Wij = np.array(Wij)

    ax_main.scatter(SiSj, Wij, alpha=0.5, color=color, label=f"Year {year}")
    ax_main.set_xscale("log")
    ax_main.set_yscale("log")
    ax_main.set_xlabel(r"$s_i s_j$", fontsize=20)
    ax_main.set_ylabel(r"$w_{ij}$", fontsize=20)
    ax_main.set_title(rf"$w_{{ij}} \sim (s_i s_j)^{{\\gamma}}$, $\gamma$ ≈ {gamma:.2f}")
    
    # Plot fitted line
    x_vals = np.linspace(min(SiSj), max(SiSj), 100)
    y_fit = x_vals ** gamma
    ax_main.plot(x_vals, y_fit, color='k', lw=2, label=rf"Fit: $\gamma$ = {gamma:.2f}")
    ax_main.legend()


def analyze_trade_data(df, years, weight_col='weight'):
    graphs = build_all_networks_trade_volume(df, years, weight_col)
    gammas = []

    for year in years:
        G = graphs[year]
        A, nodes = get_adjacency_matrix(G)
        s = A.sum(axis=1)
        SiSj = []
        Wij = []

        for i in range(len(nodes)):
            for j in range(i+1, len(nodes)):
                if A[i, j] > 0:
                    SiSj.append(s[i] * s[j])
                    Wij.append(A[i, j])

        SiSj = np.array(SiSj)
        Wij = np.array(Wij)
        min_sisj = np.percentile(SiSj, 1)
        min_weight = np.percentile(Wij, 1)
        
        max_sisj = np.percentile(SiSj, 99)
        max_weight = np.percentile(Wij, 99)

        gamma, _ = fit_gamma(SiSj, Wij, min_weight, min_sisj, max_weight, max_sisj)
        gammas.append((year, gamma))

    return graphs, gammas


def process_and_plot(df, years, ax_main, ax_inset, title_prefix=""):
    graphs, gammas = analyze_trade_data(df, years)
    year_2019_graph = graphs[2019]
    gamma_2019 = dict(gammas)[2019]

    plot_si_sj_vs_wij(year_2019_graph, 2019, ax_main, gamma_2019)
    ax_main.set_title(f"{title_prefix} $w_{{ij}} \\sim (s_i s_j)^{{\\gamma}}$, γ ≈ {gamma_2019:.2f}")

    # Inset: Gamma over years
    years_sorted, gamma_vals = zip(*sorted(gammas))
    ax_inset.plot(years_sorted, gamma_vals, marker='o')
    ax_inset.set_title("γ over years", fontsize=8)
    ax_inset.set_xlabel("Year", fontsize=7)
    ax_inset.set_ylabel("γ", fontsize=7)
    ax_inset.tick_params(axis='both', labelsize=6)
    ax_inset.set_ylim([0.7, 1.2])

def scaling_gamma(df1, df2, df3, df4):
    years1 = sorted(df1['year'].unique())
    years2 = sorted(df2['year'].unique())
    years3 = sorted(df3['year'].unique())
    years4 = sorted(df4['year'].unique())

    fig = plt.figure(figsize=(16, 8))
    gs = gridspec.GridSpec(2, 3, height_ratios=[2, 1])

    # Main large plot for df1
    
    # Main large plot for df1
#     ax_main1 = fig.add_subplot(gs[0, 0:2])
    ax_main1 = fig.add_axes([0.3, 0.5, 0.4, 0.5])  # [left, bottom, width, height]
    ax_main1.text(0.05, 0.85, "(a)", transform=ax_main1.transAxes,
              fontsize=20, fontweight='bold', va='top', ha='left')

    ax_inset1 = fig.add_axes([0.5, 0.55, 0.2, 0.15])

    # Top right plot for df2
    ax_main2 = fig.add_subplot(gs[1, 0])
    ax_main2.text(0.05, 0.7, "(b)", transform=ax_main2.transAxes,
                  fontsize=20, fontweight='bold', va='top', ha='left')

    # Bottom right plot for df3
    ax_main3 = fig.add_subplot(gs[1, 1])
    ax_main3.text(0.05, 0.7, "(c)", transform=ax_main3.transAxes,
                  fontsize=20, fontweight='bold', va='top', ha='left')
    
    # Bottom right plot for df3
    ax_main4 = fig.add_subplot(gs[1, 2])
    ax_main4.text(0.05, 0.7, "(d)", transform=ax_main4.transAxes,
                  fontsize=20, fontweight='bold', va='top', ha='left')


    # Process and plot all three
    graphs1, gammas1 = analyze_trade_data(df1, years1)
    graphs2, gammas2 = analyze_trade_data(df2, years2)
    graphs3, gammas3 = analyze_trade_data(df3, years3)
    graphs4, gammas4 = analyze_trade_data(df4, years4)

    colors = plt.get_cmap("tab10").colors  # or pick your own RGB tuples
    color1 = colors[0]
    color2 = colors[1]
    color3 = colors[2]
    color4 = colors[3]

    # === Plot scatter for each year 2019 ===
    
    gamma_2019_1 = dict(gammas1)[2019]
    plot_si_sj_vs_wij(graphs1[2019], 2019, ax_main1, gamma_2019_1, color=color1)

    sc1 = ax_main1.scatter([], [], label="Total")  # placeholder for color extraction
    ax_main1.set_title(f"Total $w_{{ij}} \\sim (s_i s_j)^{{\\gamma}}$, $\gamma$ ≈ {gamma_2019_1:.2f}", fontsize=25)
    color1 = ax_main1.collections[0].get_facecolor()[0]

    gamma_2019_2 = dict(gammas2)[2019]
    plot_si_sj_vs_wij(graphs2[2019], 2019, ax_main2, gamma_2019_2, color=color2)

    sc2 = ax_main2.scatter([], [], label="Primary")
    ax_main2.set_title(rf"Primary $\gamma$ ≈ {gamma_2019_2:.2f}", fontsize=25)
    color2 = ax_main2.collections[0].get_facecolor()[0]

    gamma_2019_3 = dict(gammas3)[2019]
    plot_si_sj_vs_wij(graphs3[2019], 2019, ax_main3, gamma_2019_3, color=color3)

    sc3 = ax_main3.scatter([], [], label="Secondary")
    ax_main3.set_title(rf"Secondary $\gamma$ ≈ {gamma_2019_3:.2f}", fontsize=25)
    color3 = ax_main3.collections[0].get_facecolor()[0]

    gamma_2019_4 = dict(gammas4)[2019]
    plot_si_sj_vs_wij(graphs4[2019], 2019, ax_main4, gamma_2019_4, color=color4)

    sc4 = ax_main4.scatter([], [], label="Services")
    ax_main4.set_title(rf"Services $\gamma$ ≈ {gamma_2019_4:.2f}", fontsize=25)
    color4 = ax_main4.collections[0].get_facecolor()[0]
    
    # === Plot inset with all gamma evolutions ===
    def plot_gamma_inset(ax, gamma_data, label, color):
        years, gammas = zip(*sorted(gamma_data))
        ax.plot(years, gammas, marker='o', label=label, color=color)

    plot_gamma_inset(ax_inset1, gammas1, "Total", color1)
    plot_gamma_inset(ax_inset1, gammas2, "Primary", color2)
    plot_gamma_inset(ax_inset1, gammas3, "Secondary", color3)
    plot_gamma_inset(ax_inset1, gammas4, "Services", color4)


#     ax_inset1.set_title("γ over years", fontsize=8)
#     ax_inset1.set_xlabel("Year", fontsize=7)
    ax_inset1.set_ylabel(r"$\gamma$", fontsize=15)
    ax_inset1.tick_params(axis='both', labelsize=8)
    ax_inset1.set_ylim([0.7, 1.2])

    # Move the legend above the inset
    ax_inset1.legend(loc='lower center', bbox_to_anchor=(1.5, 0.1), fontsize=15, ncol=2, frameon=False)

    plt.tight_layout()
    plt.show()
#     fig.savefig('gamma_trade.png',bbox_inches='tight', dpi=300)
#######################################################################################################

# === Utility Functions ===
def compute_strengths(df):
    G = nx.Graph()
    for _, row in df.iterrows():
        G.add_edge(row['Source'], row['Target'], weight=row['weight'])
    return dict(G.degree(weight='weight'))

def compute_weighted_rich_club(df, thresholds):
    G = nx.Graph()
    for _, row in df.iterrows():
        G.add_edge(row['Source'], row['Target'], weight=row['weight'])
    strengths = compute_strengths(df)
    results = []
    for s_thresh in thresholds:
        rich_nodes = [n for n in G.nodes() if strengths.get(n, 0) >= s_thresh]
        if len(rich_nodes) < 2:
            results.append(np.nan)
            continue
        subG = G.subgraph(rich_nodes)
        total_weight = sum(subG[u][v]['weight'] for u, v in subG.edges())
        Rw = (2 * total_weight) / (len(rich_nodes) * (len(rich_nodes) - 1))
        results.append(Rw)
    return results

def symmetrize_all(df, year, var):
    df_year = df[df['year'] == year].copy()
    df_year['pair_key'] = df_year.apply(lambda row: tuple(sorted([row['Source'], row['Target']])), axis=1)
    trade_df = df_year.groupby('pair_key')[var].sum().reset_index()
    trade_df[['Source', 'Target']] = pd.DataFrame(trade_df['pair_key'].tolist(), index=trade_df.index)
    trade_df = trade_df.drop(columns='pair_key').rename(columns={var: 'weight'})
    return trade_df[['Source', 'Target', 'weight']]

# === Plotting Function ===
def plot_all_rich_club(ax_rho, df, results, label, year, var, color, marker):

    df_real = symmetrize_all(df, year, var)
    strengths_real = np.array(list(compute_strengths(df_real).values()))
    quantile_levels = np.linspace(0, 1, 20)
    thresholds = np.quantile(strengths_real, quantile_levels)
    rc_real = compute_weighted_rich_club(df_real, thresholds)

    # Simulations
    rc_all = []
    for r in results:
        df_sim = pd.DataFrame(r['W_model_trade']).unstack().reset_index()
        df_sim.columns = ['Source', 'Target', 'weight']
        df_sim['weight'] /= df_sim['weight'].sum()
        df_sim['pair_key'] = df_sim.apply(lambda row: tuple(sorted([row['Source'], row['Target']])), axis=1)
        df_sim = df_sim.groupby('pair_key')['weight'].sum().reset_index()
        df_sim[['Source', 'Target']] = pd.DataFrame(df_sim['pair_key'].tolist(), index=df_sim.index)
        df_sim = df_sim.drop(columns='pair_key')
        strengths_sim = np.array(list(compute_strengths(df_sim).values()))
        thresholds_sim = np.quantile(strengths_sim, quantile_levels)
        rc_sim = compute_weighted_rich_club(df_sim, thresholds_sim)
        rc_all.append(rc_sim)

    rc_all = np.array(rc_all)
    rc_mean = np.nanmean(rc_all, axis=0)
    rc_low = np.nanquantile(rc_all, 0.1, axis=0)
    rc_high = np.nanquantile(rc_all, 0.9, axis=0)

    # Ratio plot
    ratio = np.array(rc_real) / rc_mean
    ax_rho.plot(thresholds, ratio, linestyle='-', marker=marker, linewidth=2, label=label, color=color)

# === Core Plotting Utilities ===
def compute_avg_and_ci(data_lists, confidence=0.9):
    # First, sort each list and determine minimum length
    sorted_lists = [sorted(lst) for lst in data_lists]
    min_len = min(len(lst) for lst in sorted_lists)
    
    # Truncate all lists to the same minimum length
    clipped_lists = [lst[:min_len] for lst in sorted_lists]
    data_array = np.array(clipped_lists)
    
    # Compute mean and confidence intervals
    avg = np.mean(data_array, axis=0)
    lower = np.quantile(data_array, (1 - confidence) / 2, axis=0)
    upper = np.quantile(data_array, 1 - (1 - confidence) / 2, axis=0)
    
    return avg, lower, upper, min_len

def plot_cumulative_node_strength_from_graph(G, ax=None, label='Real', **kwargs):
    strengths = sorted([val for _, val in G.degree(weight='weight')])
    cdf = 1.0 - np.arange(1, len(strengths) + 1) / len(strengths)
    if ax is not None:
        ax.plot(strengths, cdf, label=label, **kwargs)
    else:
        plt.figure(figsize=(6, 4))
        plt.plot(strengths, cdf, label=label, **kwargs)
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel("Node strength (weighted degree)")
        plt.ylabel("P(s > S)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

def plot_cumulative_link_weight_from_graph(G, ax=None, label='Real', **kwargs):
    weights = sorted([d['weight'] for _, _, d in G.edges(data=True)])
    cdf = 1.0 - np.arange(1, len(weights) + 1) / len(weights)
    if ax is not None:
        ax.plot(weights, cdf, label=label, **kwargs)
    else:
        plt.figure(figsize=(6, 4))
        plt.plot(weights, cdf, label=label, **kwargs)
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel("Link weight")
        plt.ylabel("P(w > W)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

# === Simulation Graph Conversion ===

def convert_sim_result_to_graph(sim_result):
    df_sim = pd.DataFrame(sim_result['W_model_trade']).unstack().reset_index()
    df_sim.columns = ['Source', 'Target', 'weight']
    df_sim['weight'] /= df_sim['weight'].sum()
    df_sim['pair_key'] = df_sim.apply(lambda row: tuple(sorted([row['Source'], row['Target']])), axis=1)
    df_sim = df_sim.groupby('pair_key')['weight'].sum().reset_index()
    df_sim[['Source', 'Target']] = pd.DataFrame(df_sim['pair_key'].tolist(), index=df_sim.index)
    df_sim = df_sim.drop(columns='pair_key')
    G_sim = nx.Graph()
    for _, row in df_sim.iterrows():
        G_sim.add_edge(row['Source'], row['Target'], weight=row['weight'])
    return G_sim

# === Average Sorted Plotting Functions ===

def plot_avg_sorted_node_strength(sim_graphs, ax=None, label='Sim Avg', **kwargs):
    sorted_strengths = [sorted([val for _, val in G.degree(weight='weight')]) for G in sim_graphs]
    avg, low, high, n = compute_avg_and_ci(sorted_strengths)

    min_len = min(map(len, sorted_strengths))
    clipped = np.array([s[:min_len] for s in sorted_strengths])
    avg_strengths = clipped.mean(axis=0)
    cdf = 1.0 - np.arange(1, min_len + 1) / min_len
    if ax is not None:
        ax.plot(avg_strengths, cdf, label=label, **kwargs)
        ax.fill_betweenx(cdf, low, high,alpha=0.2)


def plot_avg_sorted_link_weight(sim_graphs, ax=None, label='Sim Avg', **kwargs):
    sorted_weights = [sorted([d['weight'] for _, _, d in G.edges(data=True)]) for G in sim_graphs]
    avg, low, high, n = compute_avg_and_ci(sorted_weights)

    min_len = min(map(len, sorted_weights))
    clipped = np.array([w[:min_len] for w in sorted_weights])
    avg_weights = clipped.mean(axis=0)
    cdf = 1.0 - np.arange(1, min_len + 1) / min_len
    if ax is not None:
        ax.plot(avg_weights, cdf, label=label, **kwargs)
        ax.fill_betweenx(cdf, low, high, alpha=0.2)

        
############################################################################################
# Compute node-level inequity = sum_j w^C_ij / sum_j w^T_ij
def compute_node_inequity(G_trade, G_ghg):
    inequity = {}
    for node in G_trade.nodes():
        trade_out = sum(G_trade[node][nbr]['weight'] for nbr in G_trade.successors(node) if G_trade.has_edge(node, nbr))
        ghg_out = sum(G_ghg[node][nbr]['weight'] for nbr in G_ghg.successors(node) if G_ghg.has_edge(node, nbr))
        ratio = ghg_out / trade_out if trade_out > 0 else None
        inequity[node] = ratio
    return inequity

def get_inequity_df(G_trade, G_ghg, merged_df):
    import networkx as nx
    
    inequity = {}
    for node in G_trade.nodes():
        trade_out = sum(G_trade[node][nbr]['weight'] for nbr in G_trade.successors(node) if G_trade.has_edge(node, nbr))
        ghg_out = sum(G_ghg[node][nbr]['weight'] for nbr in G_ghg.successors(node) if G_ghg.has_edge(node, nbr))
        ratio = ghg_out / trade_out if trade_out > 0 else None
        inequity[node] = ratio

    inequity_df = pd.DataFrame(list(inequity.items()), columns=['node', 'inequity'])
    gdppc_df = merged_df[merged_df['year'] == 2019][['Source', 'gdppc_exp']].drop_duplicates().rename(columns={'Source': 'node'})
    final_df = pd.merge(inequity_df, gdppc_df, on='node', how='left')

    return final_df


def plot_inequity_and_degree(G_trade, G_ghg, merged_df, threshold=1.5):
    df = get_inequity_df(G_trade, G_ghg, merged_df)
    df = df.dropna(subset=['gdppc_exp', 'inequity'])
    df = df.sort_values(by='gdppc_exp', ascending=False).reset_index(drop=True)
    df['rank'] = np.arange(1, len(df) + 1)

    # Count inequitable edges
    inequitable_counts = defaultdict(int)
    for u, v in G_trade.edges():
        w_trade = G_trade[u][v].get('weight', 0)
        w_ghg = G_ghg[u][v].get('weight', 0)
        if w_trade > 0 and (w_ghg / w_trade) > threshold:
            inequitable_counts[u] += 1
    df['k_ineq'] = df['node'].map(inequitable_counts).fillna(0).astype(int)

    fig, (axs0, axs1) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot node-level inequity
    axs0.plot(df['rank'], df['inequity'], marker='o', linestyle='-', color='black')
    axs0.axhline(1, linestyle='--', color='gray')
    axs0.fill_between(df['rank'], 1, df['inequity'], where=(df['inequity'] > 1), color='red', alpha=0.2)
    axs0.fill_between(df['rank'], df['inequity'], 1, where=(df['inequity'] < 1), color='green', alpha=0.2)
    axs0.set_xlabel("GDP per capita rank", fontsize=15)
    axs0.set_ylabel("Node-level Inequity $E_i$", fontsize=15)
    axs0.set_ylim([-1, 6])
    axs0.text(0.01, 0.95, '(a)', transform=axs0.transAxes, fontsize=25, fontweight='bold', va='top', ha='left')
    print(df.loc[df['inequity']==max(df['inequity'])])

#     axs0.set_title("(a) Inequity by GDP rank")

    # Annotate selected countries
    selected = ['LUX', 'USA', 'GBR', 'DEU', 'FRA', 'LAO', 'CHE', 'RUS', 'UKR', 'IND', 'VNM', 'BRA', 'ZAF','CHN']
    for _, row in df[df['node'].isin(selected)].iterrows():
        y_offset = -0.8 if row['inequity'] < 1 else 0.3
        axs0.text(row['rank'],  row['inequity'] + y_offset, row['node'], fontsize=10, ha='center', rotation=90)

    # Plot inequitable degree and cumulative
    axs1.plot(df['rank'], (df['k_ineq']/df['k_ineq'].sum()).cumsum(), color='purple')
    axs1.axvline(34, linestyle=':', color='black', linewidth=4)
    axs1.set_ylabel("Cumulative $k_i^{ineq}$", fontsize=15)
    axs1.set_xlabel("GDP per capita rank", fontsize=15)
    axs1.text(0.01, 0.95, '(b)', transform=axs1.transAxes, fontsize=25, fontweight='bold', va='top', ha='left')
#     fig.savefig('fig6_ineq.png')
#     axs1.set_title("(b) Cumulative Inequitable Degree")

    plt.tight_layout()
    plt.show()


def plot_trophic_levels(G_trade, G_ghg, gdp_rank_dict=None, threshold=0.0):
    edge_data = []
    for u, v in G_trade.edges():
        w_trade = G_trade[u][v].get('weight', 0)
        w_ghg = G_ghg[u][v].get('weight', 0)
        if w_trade > 0:
            e_ij = w_ghg / w_trade
            if e_ij > threshold:
                edge_data.append((u, v, e_ij))

    G_ineq = nx.DiGraph()
    for u, v, e in edge_data:
        G_ineq.add_edge(u, v, weight=e)
        

    # Convert to dictionary
    if gdp_rank_dict:
        ta.trophic_plot_gdp(G_ineq, k=1, gdp_dict=gdp_rank_dict, cmap='RdYlGn', title='Inequity Network Colored by GDP per capita')
        
    else:
        ta.trophic_plot(G_ineq, k=1)
        
    h_vals = ta.trophic_levels(G_ineq)
    F_0, _ = ta.trophic_incoherence(G_ineq)
    print('Trophic incoherence =', round(F_0, 3))

    nodes = list(G_ineq.nodes())
    sorted_items = sorted(zip(nodes, h_vals), key=lambda x: x[1])
#     nodes_sorted, h_sorted = zip(*sorted_items)

#     plt.figure(figsize=(12, 6))
#     plt.scatter(range(len(nodes_sorted)), h_sorted, c=h_sorted, cmap='viridis', s=100)
#     for i, (node, h) in enumerate(zip(nodes_sorted, h_sorted)):
#         plt.text(i, h + 0.02, node, ha='center', fontsize=8)
#     plt.xlabel("Nodes (sorted by trophic level)")
#     plt.ylabel("Trophic Level")
#     plt.title("Trophic Levels in Inequitable Subnetwork")
#     plt.xticks([])
#     plt.tight_layout()
#     plt.show()

    return sorted_items

# def get_inequity_df(G_trade, G_ghg, merged_df):
#     inequity = {}
#     for node in G_trade.nodes():
#         trade_out = sum(G_trade[node][nbr]['weight'] for nbr in G_trade.successors(node) if G_trade.has_edge(node, nbr))
#         ghg_out = sum(G_ghg[node][nbr]['weight'] for nbr in G_ghg.successors(node) if G_ghg.has_edge(node, nbr))
#         ratio = ghg_out / trade_out if trade_out > 0 else None
#         inequity[node] = ratio

#     inequity_df = pd.DataFrame(list(inequity.items()), columns=['node', 'inequity'])
#     gdppc_df = merged_df[merged_df['year'] == 2019][['Source', 'gdppc_exp']].drop_duplicates().rename(columns={'Source': 'node'})
#     final_df = pd.merge(inequity_df, gdppc_df, on='node', how='left')
#     return final_df

def compute_trophic_incoherence_series(G_dir, G_ghg_dir, threshold=1.0):
    years = range(1995, 2021)
    results = []
    for year in years:
        G_trade = G_dir[year]
        G_ghg = G_ghg_dir[year]
        edge_data = []
        for u, v in G_trade.edges():
            w_trade = G_trade[u][v].get('weight', 0)
            w_ghg = G_ghg[u][v].get('weight', 0)
            if w_trade > 0:
                e_ij = w_ghg / w_trade
                if e_ij > threshold:
                    edge_data.append((u, v, e_ij))
        G_ineq = nx.DiGraph()
        for u, v, e in edge_data:
            G_ineq.add_edge(u, v, weight=e)
        if G_ineq.number_of_edges() > 0:
            F_0, _ = ta.trophic_incoherence(G_ineq)
            results.append((year, F_0))
        else:
            results.append((year, np.nan))
    return pd.DataFrame(results, columns=['Year', 'F_0'])

def compute_trophic_gdp_correlations(sorted_items, inequity_df):
    # Extract trophic levels
    trophic_df = pd.DataFrame(sorted_items, columns=['node', 'trophic_level'])

    # Merge with inequity and GDP data
    merged = pd.merge(trophic_df, inequity_df[['node', 'gdppc_exp']], on='node', how='left').dropna()

    # Rank by GDP per capita (higher GDP = lower rank)
    merged['gdp_rank'] = merged['gdppc_exp'].rank(ascending=False)

    # Correlations
    pearson_gdp = pearsonr(merged['gdppc_exp'], merged['trophic_level'])[0]
    spearman_gdp = spearmanr(merged['gdppc_exp'], merged['trophic_level'])[0]
    pearson_rank = pearsonr(merged['gdp_rank'], merged['trophic_level'])[0]
    spearman_rank = spearmanr(merged['gdp_rank'], merged['trophic_level'])[0]

    print("Pearson corr (GDP per capita vs trophic level):", round(pearson_gdp, 3))
    print("Spearman corr (GDP per capita vs trophic level):", round(spearman_gdp, 3))
    print("Pearson corr (GDP rank vs trophic level):", round(pearson_rank, 3))
    print("Spearman corr (GDP rank vs trophic level):", round(spearman_rank, 3))

    return merged  # Optional: return for inspection or plotting


# ----------- Main Multi-panel Plot Function -----------

def plot_trophic_overview(G_dir, G_ghg_dir,
                          G_primary_dir, G_primary_ghg_dir,
                          G_secondary_dir, G_secondary_ghg_dir,
                          G_services_dir, G_services_ghg_dir,
                          gdp_rank_dict):
    # Compute trophic incoherence over time
    df_total = compute_trophic_incoherence_series(G_dir, G_ghg_dir)
    df_primary = compute_trophic_incoherence_series(G_primary_dir, G_primary_ghg_dir)
    df_secondary = compute_trophic_incoherence_series(G_secondary_dir, G_secondary_ghg_dir)
    df_services = compute_trophic_incoherence_series(G_services_dir, G_services_ghg_dir)

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, height_ratios=[2, 1])

    def build_inequity_graph(G, G_ghg):
        G_ineq = nx.DiGraph()
        for u, v in G.edges():
            if G.has_edge(u, v) and G_ghg.has_edge(u, v):
                w_trade = G[u][v].get('weight', 0)
                w_ghg = G_ghg[u][v].get('weight', 0)
                if w_trade > 0:
                    e = w_ghg / w_trade
                    if e > 1:
                        G_ineq.add_edge(u, v, weight=e)
        return G_ineq

    # (a) Total Trophic Plot
    ax0 = fig.add_subplot(gs[0, 0:2])
    plt.sca(ax0)
    G_ineq_2019 = build_inequity_graph(G_dir[2019], G_ghg_dir[2019])

    # Compute trophic levels
    trophic_levels = ta.trophic_levels(G_ineq_2019)
    nodes = list(G_ineq_2019.nodes())
    sorted_levels = sorted(zip(nodes, trophic_levels), key=lambda x: x[1])

    # Print top/bottom countries
    top_5 = sorted_levels[-14:]
    bottom_5 = sorted_levels[:10]

    print(f"\nTop 5 countries in trophic level:")
    for country, level in reversed(top_5):
        print(f"{country}: {level:.2f}")

    print(f"Bottom 5 countries in trophic level:")
    for country, level in bottom_5:
        print(f"{country}: {level:.2f}")

    # Compute Spearman correlation with GDP rank
    gdp_ranks = [gdp_rank_dict.get(node, np.nan) for node in nodes]
    valid_idx = [i for i, r in enumerate(gdp_ranks) if not np.isnan(r)]

    if valid_idx:
        ranks_filtered = [gdp_ranks[i] for i in valid_idx]
        trophic_filtered = [trophic_levels[i] for i in valid_idx]
        rho, pval = spearmanr(ranks_filtered, trophic_filtered)
        print(f"Spearman correlation between GDP rank and trophic level (Total): rho = {rho:.2f}, p = {pval:.4f}")
    else:
        print("No valid GDP rank data for Total sector.")



    ta.trophic_plot_gdp(G_ineq_2019, k=1, gdp_dict=gdp_rank_dict, cmap='coolwarm', ax=ax0)
    ax0.text(0.01, 0.95, "(a)", transform=ax0.transAxes, fontsize=25, fontweight='bold', va='top', ha='left')

    # (b) Time Series Plot
    ax1 = fig.add_subplot(gs[0, 2])
    ax1.plot(df_total['Year'], 1- df_total['F_0'], label='Total', lw=2)
    ax1.plot(df_primary['Year'], 1- df_primary['F_0'], label='Primary')
    ax1.plot(df_secondary['Year'], 1- df_secondary['F_0'], label='Secondary')
    ax1.plot(df_services['Year'], 1- df_services['F_0'], label='Services')
    ax1.set_xlabel("Year", fontsize=15)
    ax1.set_ylabel("Trophic Coherence ($1 - F_0$)", fontsize=15)
    ax1.tick_params(axis='both', labelsize=15)
    ax1.legend(fontsize=15)
    ax1.text(0.01, 0.95, "(b)", transform=ax1.transAxes, fontsize=25, fontweight='bold', va='top', ha='left')

#     # Bottom: Primary, Secondary, Services
#     for idx, (Gset, Gghgset, subplot_idx, label, sector_label) in enumerate([
#         (G_primary_dir, G_primary_ghg_dir, gs[1, 0], "(c)", "Primary"),
#         (G_secondary_dir, G_secondary_ghg_dir, gs[1, 1], "(d)", "Secondary"),
#         (G_services_dir, G_services_ghg_dir, gs[1, 2], "(e)", "Services")
#     ]):
#         ax = fig.add_subplot(subplot_idx)
#         plt.sca(ax)
#         G_ineq = build_inequity_graph(Gset[2019], Gghgset[2019])
#         ta.trophic_plot_gdp(G_ineq, k=1, gdp_dict=gdp_rank_dict, cmap='coolwarm', ax=ax)
#         ax.text(0.01, 0.95, label, transform=ax.transAxes, fontsize=25, fontweight='bold', va='top', ha='left')
#         ax.text(0.75, 0.025, sector_label, transform=ax.transAxes, fontsize=15,
#                 fontweight='bold', ha='center', va='bottom')
    # Bottom: Primary, Secondary, Services
    for idx, (Gset, Gghgset, subplot_idx, label, sector_label) in enumerate([
        (G_primary_dir, G_primary_ghg_dir, gs[1, 0], "(c)", "Primary"),
        (G_secondary_dir, G_secondary_ghg_dir, gs[1, 1], "(d)", "Secondary"),
        (G_services_dir, G_services_ghg_dir, gs[1, 2], "(e)", "Services")
    ]):
        ax = fig.add_subplot(subplot_idx)
        plt.sca(ax)
        G_ineq = build_inequity_graph(Gset[2019], Gghgset[2019])

        # Compute trophic levels
        if len(G_ineq) > 0:
            trophic_levels = ta.trophic_levels(G_ineq)
            nodes = list(G_ineq.nodes())

            # Get GDP rank for those nodes
            gdp_ranks = [gdp_rank_dict.get(node, np.nan) for node in nodes]
            valid_idx = [i for i, r in enumerate(gdp_ranks) if not np.isnan(r)]

            if valid_idx:
                ranks_filtered = [gdp_ranks[i] for i in valid_idx]
                trophic_filtered = [trophic_levels[i] for i in valid_idx]
                rho, pval = spearmanr(ranks_filtered, trophic_filtered)
                print(f"Spearman correlation between GDP rank and trophic level ({sector_label}): rho = {rho:.2f}, p = {pval:.4f}")
            else:
                print(f"No valid GDP rank data for {sector_label} sector.")
        else:
            print(f"No edges in G_ineq for {sector_label} sector.")

        # Plot the trophic layout
        ta.trophic_plot_gdp(G_ineq, k=1, gdp_dict=gdp_rank_dict, cmap='coolwarm', ax=ax)
        ax.text(0.01, 0.95, label, transform=ax.transAxes, fontsize=25, fontweight='bold', va='top', ha='left')
        ax.text(0.75, 0.025, sector_label, transform=ax.transAxes, fontsize=15,
                fontweight='bold', ha='center', va='bottom')

    plt.tight_layout()
    plt.show()
#     fig.savefig('fig7_tc.png')
    
    return df_total

# Function to compute total weights between groups
def compute_group_weights(G, group1, group2):
    flows = {
        'G1 → G1': 0.0,
        'G1 → G2': 0.0,
        'G2 → G1': 0.0,
        'G2 → G2': 0.0
    }
    for u, v, data in G.edges(data=True):
        w = data.get('weight', 0.0)
        if u in group1 and v in group1:
            flows['G1 → G1'] += w
        elif u in group1 and v in group2:
            flows['G1 → G2'] += w
        elif u in group2 and v in group1:
            flows['G2 → G1'] += w
        elif u in group2 and v in group2:
            flows['G2 → G2'] += w
    return flows
