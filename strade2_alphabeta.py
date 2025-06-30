import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import pickle
from multiprocessing import Pool, cpu_count
from Strade_gravity import Strade_gravity, build_all_networks_trade_volume, reorder_matrix_by_trade_volume, mean_absolute_error
from tqdm import tqdm

merged_df = pd.read_csv('/rds/general/user/nk821/home/anaconda3/CCS/Emissions/OECD/Total/merged_df.csv')

    
def run_single_sim(args):
    alpha, beta, year, seed = args
    print(f"Running α={alpha:.3f}, seed={seed}", flush=True)


    G_trade = build_all_networks_trade_volume(merged_df, [year], 'trade_share')[year]
    W_real_trade = reorder_matrix_by_trade_volume(nx.to_numpy_array(G_trade))[0]

    G_ghg = build_all_networks_trade_volume(merged_df, [year], 'ghg_share')[year]
    W_real_ghg = reorder_matrix_by_trade_volume(nx.to_numpy_array(G_ghg))[0]

    model_trade = Strade_gravity(W_real_trade.shape[0], W_real_trade.shape[1],
                                 np.sum(W_real_trade > 0), alpha, beta, delta=0,
                                 max_tokens=merged_df.shape[0], max_steps=merged_df.shape[0], seed=seed)
    W_model_trade = model_trade.run()
    trade_err = mean_absolute_error(W_real_trade, W_model_trade)

    model_ghg = Strade_gravity(W_real_ghg.shape[0], W_real_ghg.shape[1],
                               np.sum(W_real_ghg > 0), alpha, beta, delta=0,
                               max_tokens=merged_df.shape[0], max_steps=merged_df.shape[0], seed=seed)
    W_model_ghg = model_ghg.run()
    ghg_err = mean_absolute_error(W_real_ghg, W_model_ghg)

    return {
        'year': year,
        'alpha': alpha,
        'beta': beta,
        'seed': seed,
        'trade_error': trade_err,
        'ghg_error': ghg_err,
        'W_model_trade':W_model_trade,
        'W_model_ghg':W_model_ghg
    }

def joint_alpha_equal_beta_fit_parallel(year_index, n_runs=50):
#     year = range(1995,2021)[year_index - 1]
    year = [1995, 2005, 2019][year_index - 1]
#     alphas = np.linspace(0.85, 1.02, 18)
#     betas = np.linspace(0.85, 1.02, 18)
#     alphas = [1, 0.96, 0.9]
#     betas = [1, 0.96, 0.9]
    alphas = np.array([1, 0.96, 1.02])
    betas = np.array([1, 0.96, 1.02])
    tasks = [(alpha, beta, year, seed) for beta in betas for alpha in alphas for seed in range(1, n_runs + 1)]


    print(f"Total jobs: {len(tasks)}. Using {256} cores.")

    with Pool(processes=256) as pool:
        results = list(tqdm(pool.imap(run_single_sim, tasks), total=len(tasks)))


#     df = pd.DataFrame(results)

#     # Aggregate by alpha (same as beta)
#     grouped = df.groupby("alpha").agg({
#         "trade_error": "mean",
#         "ghg_error": "mean"
#     }).reset_index()

#     # Plot results
#     plt.figure(figsize=(10, 5))
#     plt.plot(grouped['alpha'], grouped['trade_error'], label="Trade Error", marker='o')
#     plt.plot(grouped['alpha'], grouped['ghg_error'], label="GHG Error", marker='s')
#     plt.xlabel(r"$\alpha = \beta$")
#     plt.ylabel("Average Error")
#     plt.title(f"Average Error vs α=β for year {year}")
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
#     plt.savefig(f'/rds/general/user/nk821/home/anaconda3/CCS/Emissions/OECD/Secondary/2_error_plot_alpha_equal_beta_{year}_n{n_runs}.png')
#     plt.show()

    # Save all detailed results
    with open(f'/rds/general/user/nk821/home/anaconda3/CCS/Emissions/OECD/Total/Pickles/31_alphabeta_parallel_{year}_n{n_runs}.pkl', 'wb') as f:
        pickle.dump(results, f)

    print(f"Saved logs and error plot for year {year}")

# Run all years
for i in range(1, 4):
    joint_alpha_equal_beta_fit_parallel(i, n_runs=100)
