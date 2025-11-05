# --------------------------------------------------------------
# INSTALL PACKAGES (run once)
# --------------------------------------------------------------
!pip install -q pandas scipy numpy networkx matplotlib

# --------------------------------------------------------------
# FOUR CLEAN NETWORKS + CORRECT METRICS (tau >= 0.5 only)
# --------------------------------------------------------------
import pandas as pd
import numpy as np
from scipy.stats import kendalltau
import networkx as nx
import matplotlib.pyplot as plt
import itertools
import warnings
warnings.filterwarnings("ignore")

# 1. Load data
intra = pd.read_csv("Pit_Fresh_Intra.csv")
extra = pd.read_csv("Pit_Fresh_Extra.csv")

genes = ['tet33', 'tetG', 'tetM', 'tetX', 'ermB', 'sul1', 'intI1', 'intI2', 'intI3']

# Extract groups
pit_ex   = extra[extra['exDNA type'] == 'Pit'][genes]
fresh_ex = extra[extra['exDNA type'] == 'Fresh'][genes]
pit_i    = intra[intra['iDNA type'] == 'Pit'][genes]
fresh_i  = intra[intra['iDNA type'] == 'Fresh'][genes]

# 2. Build full network with tau values
def build_full_network(df):
    G = nx.Graph()
    G.add_nodes_from(genes)
    for (i, j) in itertools.combinations(range(len(genes)), 2):
        g1, g2 = genes[i], genes[j]
        tau, _ = kendalltau(df.iloc[:, i], df.iloc[:, j])
        G.add_edge(g1, g2, tau=tau, strength='strong' if tau >= 0.5 else 'weak')
    return G

G_pit_ex   = build_full_network(pit_ex)
G_fresh_ex = build_full_network(fresh_ex)
G_pit_i    = build_full_network(pit_i)
G_fresh_i  = build_full_network(fresh_i)

# 3. Unified layout
all_edges = [(u, v) for G in [G_pit_ex, G_fresh_ex, G_pit_i, G_fresh_i] for u, v in G.edges()]
pos = nx.kamada_kawai_layout(nx.Graph(all_edges))

# 4. METRICS: ONLY STRONG EDGES (tau >= 0.5)
def print_strong_metrics(G, name):
    strong_G = nx.Graph()
    strong_edges = [(u, v) for u, v, d in G.edges(data=True) if d['strength'] == 'strong']
    strong_G.add_edges_from(strong_edges)
    strong_G.add_nodes_from(genes)
    
    n_edges = len(strong_edges)
    density = nx.density(strong_G)
    avg_degree = np.mean([d for n, d in strong_G.degree()])
    avg_clust = nx.average_clustering(strong_G)
    
    print(f"{name} (tau >= 0.5 only)")
    print(f"  Strong edges:           {n_edges}")
    print(f"  Density:                {density:.3f}")
    print(f"  Avg degree:             {avg_degree:.2f}")
    print(f"  Avg clustering:         {avg_clust:.3f}\n")

print("=== NETWORK METRICS (tau >= 0.5 only) ===\n")
print_strong_metrics(G_pit_ex,   "Pit exDNA")
print_strong_metrics(G_fresh_ex, "Fresh exDNA")
print_strong_metrics(G_pit_i,    "Pit iDNA")
print_strong_metrics(G_fresh_i,  "Fresh iDNA")

# 5. Plot function (exDNA = light blue, iDNA = light orange)
def plot_clean_network(G, title, filename, strong_color, node_color):
    plt.figure(figsize=(11, 10))
    
    # Strong edges: straight, scaled by |tau|, thinner
    strong = [(u, v) for u, v, d in G.edges(data=True) if d['strength'] == 'strong']
    widths = [5 * abs(G[u][v]['tau']) for u, v in strong]
    nx.draw_networkx_edges(G, pos, edgelist=strong, width=widths,
                           edge_color=strong_color, alpha=0.9, style='solid')

    # Weak edges: curved, very thin
    weak = [(u, v) for u, v, d in G.edges(data=True) if d['strength'] == 'weak']
    nx.draw_networkx_edges(G, pos, edgelist=weak, width=2,
                           edge_color='lightgray', alpha=0.35, style='solid',
                           connectionstyle='arc3,rad=0.25')

    # Nodes
    nx.draw_networkx_nodes(G, pos, node_color=node_color, node_size=3000,
                           edgecolors='k', linewidths=1.5)
    nx.draw_networkx_labels(G, pos, font_size=11, font_weight='bold', font_color='black')

    plt.title(title, fontsize=16, pad=25, weight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    print(f"Saved: {filename}\n")

# 6. Generate 4 figures with NEW COLORS
plot_clean_network(G_pit_ex,   "Pit exDNA (tau >= 0.5 strong | tau < 0.5 weak)",   "Pit_exDNA_clean.png",     '#1F4E79', '#A0D8F1')   # dark blue / light blue
plot_clean_network(G_fresh_ex, "Fresh exDNA (tau >= 0.5 strong | tau < 0.5 weak)", "Fresh_exDNA_clean.png",   '#2E5B88', '#A0D8F1')   # darker blue / light blue
plot_clean_network(G_pit_i,    "Pit iDNA (tau >= 0.5 strong | tau < 0.5 weak)",    "Pit_iDNA_clean.png",      '#D2691E', '#FFDAB9')   # chocolate / light orange
plot_clean_network(G_fresh_i,  "Fresh iDNA (tau >= 0.5 strong | tau < 0.5 weak)",  "Fresh_iDNA_clean.png",    '#FF8C00', '#FFDAB9')   # dark orange / light orange