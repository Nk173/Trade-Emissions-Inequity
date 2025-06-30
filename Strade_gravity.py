import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.colors as mcolors
from scipy.optimize import minimize
from scipy.stats import spearmanr, pearsonr
import pickle

class Strade_gravity:
    def __init__(self, total_exporters, total_importers, target_links,
                 alpha=1.0, beta=1.0, delta=0.0, seed_size=3,
                 max_tokens=None, max_steps=None, seed=1):

        self.total_exporters = total_exporters
        self.total_importers = total_importers
        self.target_links = target_links
        self.alpha = alpha
        self.beta = beta
        self.seed_size = seed_size
        self.max_tokens = max_tokens
        self.max_steps = max_steps
        self.seed = seed
        self.delta=delta

        np.random.seed(seed)

        # Trade matrix: rows = exporters, cols = importers
        self.trade_matrix = np.zeros((total_exporters, total_importers), dtype=int)

        # Start with a seed block
        self.active_exporters = seed_size
        self.active_importers = seed_size
        self.token_count = 0
        self.step_count = 0

        # Logs
        self.exporter_growth = []
        self.importer_growth = []
        self.link_growth = []

        # Initialize with seed
        for i in range(seed_size):
            for j in range(seed_size):
                if i != j:
                    self.trade_matrix[i, j] = 1
                    self.token_count += 1

        # Estimate lambda values using Synthrade paper method
        self.lambda_exp, self.lambda_imp = self.estimate_lambda()

    def estimate_lambda(self):
        s = self.seed_size
        texp = self.total_exporters * (self.total_exporters - s) / (2 * s)
        timp = self.total_importers * (self.total_importers - s) / (2 * s)
        tf = max(texp, timp)

        lambda_exp = self.total_exporters * (self.total_exporters - s) / (2 * tf)
        lambda_imp = self.total_importers * (self.total_importers - s) / (2 * tf)
        return lambda_exp, lambda_imp

    def run(self):
        while True:
            self.step_count += 1

            current_links = np.sum(self.trade_matrix[:self.active_exporters, :self.active_importers] > 0)

            # Log evolution
            self.exporter_growth.append(self.active_exporters)
            self.importer_growth.append(self.active_importers)
            self.link_growth.append(current_links)

            # Stopping criteria
            if self.target_links is not None and current_links >= self.target_links:
                print("Stopping: Target number of links reached.")
                break
            if self.max_tokens is not None and self.token_count >= self.max_tokens:
                print("Stopping: Maximum number of tokens reached.")
                break
            if self.max_steps is not None and self.step_count >= self.max_steps:
                print("Stopping: Maximum number of steps reached.")
                break

            new_node = False

            # Add new exporter
            if self.active_exporters < self.total_exporters:
                prob = min(1.0, self.lambda_exp / self.active_exporters)
                if np.random.rand() < prob:
                    probs = self.trade_matrix[:self.active_exporters, :self.active_importers].sum(axis=0) ** (self.alpha + self.delta)
                    rand = np.random.rand(self.active_importers)
                    targets = np.where(rand < probs / probs.sum())[0] if probs.sum() > 0 else []
                    if len(targets) > 0:
                        self.trade_matrix[self.active_exporters, targets] += 1
                        self.active_exporters += 1
                        self.token_count += len(targets)
                        new_node = True

            # Add new importer
            if self.active_importers < self.total_importers:
                prob = min(1.0, self.lambda_imp / self.active_importers)
                if np.random.rand() < prob:
                    probs = self.trade_matrix[:self.active_exporters, :self.active_importers].sum(axis=1) ** self.beta
                    rand = np.random.rand(self.active_exporters)
                    targets = np.where(rand < probs / probs.sum())[0] if probs.sum() > 0 else []
                    if len(targets) > 0:
                        self.trade_matrix[targets, self.active_importers] += 1
                        self.active_importers += 1
                        self.token_count += len(targets)
                        new_node = True

            # Weight aggregation if no new node added
            if not new_node:
                exports = self.trade_matrix[:self.active_exporters, :self.active_importers].sum(axis=1)
                imports = self.trade_matrix[:self.active_exporters, :self.active_importers].sum(axis=0)

                exp_weights = np.power(exports, self.alpha, where=exports > 0, out=np.zeros_like(exports, dtype=float))
                imp_weights = np.power(imports, self.beta, where=imports > 0, out=np.zeros_like(imports, dtype=float))

                P_link = np.outer(exp_weights, imp_weights)
                np.fill_diagonal(P_link, 0)

#                 if P_link.sum() == 0:
#                     continue
#                 P_link /= P_link.sum()
                total_prob = P_link.sum()
                if total_prob == 0:
                    continue

                P_link /= total_prob

                i, j = np.unravel_index(np.random.choice(P_link.size, p=P_link.ravel()), P_link.shape)
                
                self.trade_matrix[i, j] += 1
                self.token_count += 1

#                 delT = self.trade_matrix[i, j]*(self.trade_matrix[i, j]/np.sum(self.trade_matrix))
#                 self.trade_matrix[i, j]+= delT
#                 self.token_count += delT
                
                
                
#         print('final:',P_link)
        # Final reporting
        final_links = np.sum(self.trade_matrix > 0)
        print(f"Simulation completed in {self.step_count} steps.")
        print(f"Total nonzero trade links: {final_links}")
        print(f"Final exporter count: {self.active_exporters} / {self.total_exporters}")
        print(f"Final importer count: {self.active_importers} / {self.total_importers}")
        print(f"Total tokens thrown: {self.token_count}")
        return self.trade_matrix

    def plot_growth_curves(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.exporter_growth, label='Exporters (active)')
        plt.plot(self.importer_growth, label='Importers (active)')
        plt.plot(self.link_growth, label='Trade links (> 0)')
        plt.xlabel("Simulation steps")
        plt.ylabel("Count")
        plt.title("Node and Link Growth Over Time")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
    def plot_strength_distribution(self, normalize=True, logscale=True, title="Strength Distribution"):
        """
        Plot the in-strength (imports) and out-strength (exports) distributions.

        Args:
                normalize (bool): Normalize to share of total trade
                logscale (bool): Use log-log scale
                title (str): Plot title
        """
        exports = self.trade_matrix.sum(axis=1)
        imports = self.trade_matrix.sum(axis=0)

        exports_sorted = np.sort(exports[exports > 0])[::-1]
        imports_sorted = np.sort(imports[imports > 0])[::-1]

        if normalize:
                exports_sorted = exports_sorted / exports_sorted.sum()
                imports_sorted = imports_sorted / imports_sorted.sum()

        plt.figure(figsize=(8, 6))
        plt.plot(exports_sorted, 'o-', label="Exports (Out-strength)")
        plt.plot(imports_sorted, 'x--', label="Imports (In-strength)")
        plt.xlabel("Rank")
        plt.ylabel("Share" if normalize else "Trade Volume")
        plt.title(title)
        if logscale:
            plt.xscale("log")
            plt.yscale("log")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
    def plot_cumulative_strength_distribution(self, normalize=True, logscale=True, diff_case=False,
                                              title="Complementary CDF of Strength"):
        """
        Plot the complementary cumulative distribution function (CCDF) of 
        in-strength and out-strength: P(s_i >= s)

        Args:
            normalize (bool): Normalize to share of total trade
            logscale (bool): Use log-log scale
            title (str): Plot title
        """
        exports = self.trade_matrix.sum(axis=1)
        imports = self.trade_matrix.sum(axis=0)

        exports_sorted = np.sort(exports[exports > 0])[::-1]
        imports_sorted = np.sort(imports[imports > 0])[::-1]

        if normalize:
            exports_sorted = exports_sorted / exports_sorted.sum()
            imports_sorted = imports_sorted / imports_sorted.sum()
            
        if diff_case:
            diff = np.abs((exports/exports.sum()) - (imports/imports.sum()))
            diff = np.sort(diff[diff>0])[::-1]
            P_c_diff = 1- np.arange(1, len(diff) +1) [::-1] / len(diff)
            

        P_c_exports = 1 - np.arange(1, len(exports_sorted) + 1)[::-1] / len(exports_sorted)
        P_c_imports = 1 - np.arange(1, len(imports_sorted) + 1)[::-1] / len(imports_sorted)
                        
        plt.figure(figsize=(8, 6))
        plt.plot(exports_sorted, P_c_exports, 'o-', label="Exports (Out-strength)")
        plt.plot(imports_sorted, P_c_imports, 'x--', label="Imports (In-strength)")
        if diff_case:
            plt.plot(diff, P_c_diff, 'v--', label="Diff")
        plt.xlabel("Strength (s)")
        plt.ylabel(r"$P_c(s_i \geq s)$")
        plt.title(title)
        if logscale:
            plt.xscale("log")
            plt.yscale("log")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        
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

def marginal_strength_distance(W_real, W_model):
    A = W_real
    B = W_model / W_model.sum()
    return np.linalg.norm(np.log1p(A.sum(axis=0)) - np.log1p(B.sum(axis=0))) + np.linalg.norm(np.log1p(A.sum(axis=1)) - np.log1p(B.sum(axis=1)))


def spearman_rank_correlation(W_real, W_model, exclude_zeros=True):
    """
    Spearman rank correlation between flattened real and model matrices.
    """
    W_real = W_real
    W_model = W_model / W_model.sum().sum()

    vec_real = W_real.flatten()
    vec_model = W_model.flatten()

    if exclude_zeros:
        mask = (vec_real + vec_model) > 0
        vec_real = vec_real[mask]
        vec_model = vec_model[mask]

    return spearmanr(vec_real, vec_model).correlation


def pearson_correlation(W_real, W_model, exclude_zeros=True):
    """
    Pearson correlation between flattened real and model matrices.
    """
    W_real = W_real 
    W_model = W_model / W_model.sum().sum()

    vec_real = W_real.flatten()
    vec_model = W_model.flatten()

    if exclude_zeros:
        mask = (vec_real + vec_model) > 0
        vec_real = vec_real[mask]
        vec_model = vec_model[mask]

    return pearsonr(vec_real, vec_model)[0]

def mean_absolute_error(W_real, W_model, exclude_zeros=True):
    """
    Mean Absolute Error between normalized flattened matrices.
    """
    W_real = W_real / W_real.sum().sum()
    W_model = W_model / W_model.sum().sum()
    
    # Get sorting order by row sums (export strength)
    row_order_real = np.argsort(-W_real.sum(axis=1))
    row_order_model = np.argsort(-W_model.sum(axis=1))

    # Reorder rows and columns of both matrices
    W_real_sorted = W_real[row_order_real, :][:, row_order_real]
    W_model_sorted = W_model[row_order_model, :][:, row_order_model]

    vec_real = W_real_sorted.flatten()
    vec_model = W_model_sorted.flatten()

    if exclude_zeros:
        mask = (vec_real + vec_model) > 0
        vec_real = vec_real[mask]
        vec_model = vec_model[mask]

    return np.mean(np.abs(vec_real - vec_model))


### Plotting Functions

def plot_error_surface(year, sector, plot=True):
    with open(f'/rds/general/user/nk821/home/anaconda3/CCS/Emissions/OECD/{sector}/Pickles/2_alphabeta_parallel_{year}_n100.pkl', 'rb') as f:
        data = pickle.load(f)
        for row in data:
            row['year'] = year

    df = pd.DataFrame(data)
    df['alpha']=np.round(df['alpha'],2)
    df['beta']=np.round(df['beta'],2)

    avg_df_trade = df.groupby(['alpha', 'beta'], as_index=False)['trade_error'].mean()
    avg_df_trade['trade_error'] = np.log10(avg_df_trade['trade_error'])

    avg_df_ghg = df.groupby(['alpha', 'beta'], as_index=False)['ghg_error'].mean()
    avg_df_ghg['ghg_error'] = np.log10(avg_df_ghg['ghg_error'])

    heatmap_data_trade = avg_df_trade.pivot(index='alpha', columns='beta', values='trade_error')
    heatmap_data_ghg = avg_df_ghg.pivot(index='alpha', columns='beta', values='ghg_error')

    heatmap_data_trade_2005= heatmap_data_trade.sort_index().sort_index(axis=1)
    heatmap_data_ghg_2005= heatmap_data_ghg.sort_index().sort_index(axis=1)

    # Find alpha-beta pair with minimum trade error
    min_trade_row = avg_df_trade.loc[avg_df_trade['trade_error'].idxmin()]
    min_alpha_trade = min_trade_row['alpha']
    min_beta_trade = min_trade_row['beta']
    min_trade_val = min_trade_row['trade_error']

    # Find alpha-beta pair with minimum ghg error
    min_ghg_row = avg_df_ghg.loc[avg_df_ghg['ghg_error'].idxmin()]
    min_alpha_ghg = min_ghg_row['alpha']
    min_beta_ghg = min_ghg_row['beta']
    min_ghg_val = min_ghg_row['ghg_error']

    # Print
    print(f"🔹 Min Trade Error at alpha={min_alpha_trade}, beta={min_beta_trade}, log10(error)={min_trade_val:.4f}")
    print(f"🔹 Min GHG Error at alpha={min_alpha_ghg}, beta={min_beta_ghg}, log10(error)={min_ghg_val:.4f}")

    fig, ax = plt.subplots(1,2, figsize=(20, 10))
    sns.heatmap(heatmap_data_trade_2005, annot=True, fmt=".3f", cmap='viridis', cbar_kws={'label': 'Trade Error'}, ax=ax[0])
    sns.heatmap(heatmap_data_ghg_2005, annot=True, fmt=".3f", cmap='viridis', cbar_kws={'label': 'Ghg Error'}, ax=ax[1])

    ax[0].set_xlabel("Beta")
    ax[0].set_ylabel("Alpha")

    ax[1].set_xlabel("Beta")
    ax[1].set_ylabel("Alpha")

    plt.tight_layout()
    plt.show()
    
    return min_alpha_trade, min_beta_trade, min_alpha_ghg, min_beta_ghg

