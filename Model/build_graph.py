import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pickle
import os
import random
import math

def build_transaction_graph(csv_path='data/transactions.csv'):
    print("--- Starting MuleGuard Graph Construction Engine ---")

    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found! Run generate_data.py first.")
        return None, None

    df = pd.read_csv(csv_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    print(f"Loaded {len(df)} transactions from {csv_path}")

    # --- Build the directed graph ---
    G = nx.DiGraph()
    for _, row in df.iterrows():
        G.add_edge(
            row['sender'],
            row['receiver'],
            amount=row['amount'],
            timestamp=row['timestamp'],
            is_fraud=row['is_fraud']
        )

    print(f"Graph built: {G.number_of_nodes()} accounts (nodes), {G.number_of_edges()} transactions (edges)")

    # --- Calculate a RICHER profile card for every account ---
    node_features = {}

    for node in G.nodes():
        in_degree = G.in_degree(node)
        out_degree = G.out_degree(node)

        incoming = list(G.in_edges(node, data=True))
        outgoing = list(G.out_edges(node, data=True))

        total_received = sum(d['amount'] for _, _, d in incoming)
        total_sent = sum(d['amount'] for _, _, d in outgoing)
        avg_received = total_received / in_degree if in_degree > 0 else 0

        # FEATURE 1: time_span_minutes
        # How long (in minutes) between the FIRST and LAST incoming payment?
        # Mule centers receive 30+ payments within ~120 minutes.
        # Normal merchants receive payments spread across hours or days.
        if incoming:
            timestamps = [d['timestamp'] for _, _, d in incoming]
            time_span_minutes = (max(timestamps) - min(timestamps)).total_seconds() / 60
        else:
            time_span_minutes = 0

        # FEATURE 2: unique_senders_ratio
        senders = [s for s, _, _ in incoming]
        unique_senders_ratio = len(set(senders)) / in_degree if in_degree > 0 else 0

        # FEATURE 3: burst_score
        # Transactions per minute during the account's busiest window.
        # High burst_score = money arrived in a rapid-fire stream (automation signature).
        burst_score = in_degree / time_span_minutes if time_span_minutes > 0 else in_degree

        # temp storage for amounts, used below to calculate low_amount_clustering
        node_features[node] = {
            'in_degree': in_degree,
            'out_degree': out_degree,
            'total_received': round(total_received, 2),
            'total_sent': round(total_sent, 2),
            'avg_received': round(avg_received, 2),
            'time_span_minutes': round(time_span_minutes, 1),
            'unique_senders_ratio': round(unique_senders_ratio, 2),
            'burst_score': round(burst_score, 2),
            '_amounts': [d['amount'] for _, _, d in incoming],
        }

    # --- FEATURE 4: low_amount_clustering (ADAPTIVE - no hardcoded Rs.500) ---
    # Instead of assuming fraud always hides under Rs.500, we ask a RELATIVE question:
    # "does this account receive payments that are unusually consistent and LOW
    #  compared to the overall spread of amounts in THIS dataset?"
    # This recalculates automatically for any dataset, any currency, any amount range.
    all_amounts = df['amount'].tolist()
    low_amount_cutoff = pd.Series(all_amounts).quantile(0.20)
    print(f"\n(Adaptive threshold) 20th percentile amount in this dataset: Rs.{low_amount_cutoff:.2f}")
    print("This replaces a hardcoded Rs.500 assumption - it recalculates for any dataset.")

    for node in node_features:
        amounts = node_features[node].pop('_amounts')
        if amounts:
            fraction_low = sum(1 for a in amounts if a < low_amount_cutoff) / len(amounts)
            mean_amt = sum(amounts) / len(amounts)
            if len(amounts) > 1 and mean_amt > 0:
                variance = sum((a - mean_amt) ** 2 for a in amounts) / len(amounts)
                std_dev = variance ** 0.5
                coefficient_of_variation = std_dev / mean_amt
            else:
                coefficient_of_variation = 1.0
            uniformity_score = max(0, 1 - coefficient_of_variation)
            low_amount_clustering = round((fraction_low * 0.6) + (uniformity_score * 0.4), 3)
        else:
            low_amount_clustering = 0
        node_features[node]['low_amount_clustering'] = low_amount_clustering

    nx.set_node_attributes(G, node_features)

    # --- Combine features into one overall "suspicion score" for ranking ---
    # FIX: burst_score is now the DOMINANT signal, not in_degree alone.
    # A merchant with 78 customers spread across a full day (low burst_score) is normal.
    # A mule center with 30 senders crammed into 2 hours (high burst_score) is the crime.
    # in_degree still contributes but is CAPPED at 50 so raw customer count alone
    # can't outrank genuinely fast, coordinated fraud activity.
    for node in node_features:
        f = node_features[node]
        if f['in_degree'] < 5:
            f['suspicion_score'] = 0.0
            continue
        f['suspicion_score'] = round(
            (f['burst_score'] * 60 * 0.5) +
            (f['low_amount_clustering'] * 30 * 0.3) +
            (min(f['in_degree'], 50) * 0.2),
            2
        )
    nx.set_node_attributes(G, node_features)

    sorted_by_suspicion = sorted(node_features.items(), key=lambda x: x[1]['suspicion_score'], reverse=True)

    print("\n--- Top 5 most suspicious accounts (full profile) ---")
    for account, f in sorted_by_suspicion[:5]:
        print(f"\n{account}")
        print(f"   in_degree: {f['in_degree']}  |  time_span: {f['time_span_minutes']} min  |  burst_score: {f['burst_score']}")
        print(f"   low_amount_clustering: {f['low_amount_clustering']}  |  suspicion_score: {f['suspicion_score']}")

    os.makedirs('model', exist_ok=True)
    with open('model/transaction_graph.pkl', 'wb') as f:
        pickle.dump(G, f)
    print("\nGraph saved to model/transaction_graph.pkl")

    return G, sorted_by_suspicion


def plot_top_mule_ring(G, sorted_by_suspicion, save_path='model/mule_ring_plot.png'):
    print("\n--- Drawing upgraded high-contrast network map ---")

    top_account, top_features = sorted_by_suspicion[0]
    fraud_senders = [s for s, _, d in G.in_edges(top_account, data=True) if d.get('is_fraud') == 1]
    normal_nodes = [n for n in G.nodes() if 'smurf' not in str(n) and 'mule' not in str(n)][:40]

    sub_nodes = [top_account] + fraud_senders + normal_nodes
    subgraph = G.subgraph(sub_nodes)

    fig = plt.figure(figsize=(14, 10), facecolor='#FFFFFF')
    gs = fig.add_gridspec(1, 4)
    ax_main = fig.add_subplot(gs[0, :3])
    ax_side = fig.add_subplot(gs[0, 3])

    ax_main.set_facecolor('#FFFFFF')
    ax_side.set_facecolor('#F7F7F9')
    ax_side.axis('off')

    pos = {top_account: (0.5, 0.0)}
    for idx, node in enumerate(fraud_senders):
        angle = (2 * math.pi * idx) / max(len(fraud_senders), 1)
        pos[node] = (0.5 + 0.4 * math.cos(angle), 0.0 + 0.4 * math.sin(angle))
    for idx, node in enumerate(normal_nodes):
        pos[node] = (random.uniform(-0.6, -0.1), random.uniform(-0.5, 0.5))

    node_colors, node_sizes = [], []
    for node in subgraph.nodes():
        in_deg = G.nodes[node].get('in_degree', 1)
        if node == top_account:
            node_colors.append('#990000')
            node_sizes.append(1400)
        elif node in fraud_senders:
            node_colors.append('#FFCC00')
            node_sizes.append(120 + min(in_deg, 10) * 15)
        else:
            node_colors.append('#0066CC')
            node_sizes.append(120 + min(in_deg, 10) * 10)

    nx.draw_networkx_nodes(subgraph, pos, ax=ax_main, node_color=node_colors,
                            node_size=node_sizes, edgecolors='black', linewidths=1.3)

    edge_colors, edge_widths = [], []
    for u, v in subgraph.edges():
        amt = G[u][v].get('amount', 0)
        is_fraud = G[u][v].get('is_fraud') == 1
        edge_colors.append('#CC0000' if is_fraud else '#999999')
        edge_widths.append(0.6 + (amt / 500) * 2.2)

    nx.draw_networkx_edges(subgraph, pos, ax=ax_main, edge_color=edge_colors,
                            width=edge_widths, arrows=True, arrowsize=10, alpha=0.75)

    nx.draw_networkx_labels(subgraph, pos, ax=ax_main,
                             labels={top_account: 'MULE\nTARGET'},
                             font_size=9, font_color='white', font_weight='bold')

    ax_main.set_title("MuleGuard Network Topology Analysis", fontsize=16,
                       color='black', fontweight='bold', pad=14)
    ax_main.text(0.5, 1.04, "Yellow = coordinated micro-structuring smurfs   |   Blue = normal retail flows",
                 transform=ax_main.transAxes, ha='center', fontsize=10, color='#333333')
    ax_main.axis('off')

    legend_handles = [
        mpatches.Patch(color='#990000', label='Mule Aggregator (target)'),
        mpatches.Patch(color='#FFCC00', label='Fraud Smurf Account'),
        mpatches.Patch(color='#0066CC', label='Normal Account'),
    ]
    ax_main.legend(handles=legend_handles, loc='lower left', fontsize=9, frameon=True)

    stats_text = (
        f"SUSPECT PROFILE\n"
        f"{'-'*22}\n\n"
        f"Account:\n{top_account[:24]}\n\n"
        f"Senders (in_degree):\n  {top_features['in_degree']}\n\n"
        f"Time span:\n  {top_features['time_span_minutes']} min\n\n"
        f"Burst score:\n  {top_features['burst_score']} txn/min\n\n"
        f"Low-amount clustering:\n  {top_features['low_amount_clustering']}\n\n"
        f"Avg received:\n  ₹{top_features['avg_received']}\n\n"
        f"{'-'*22}\n"
        f"SUSPICION SCORE\n"
        f"  {top_features['suspicion_score']}"
    )
    ax_side.text(0.05, 0.95, stats_text, transform=ax_side.transAxes,
                 fontsize=9.5, va='top', family='monospace', color='#1a1a1a',
                 bbox=dict(boxstyle='round,pad=0.6', facecolor='white', edgecolor='#990000', linewidth=1.2))

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, facecolor='#FFFFFF')
    print(f"Upgraded chart saved to {save_path}")
    plt.close()


if __name__ == "__main__":
    graph, ranked_accounts = build_transaction_graph()
    if graph is not None and ranked_accounts is not None:
        plot_top_mule_ring(graph, ranked_accounts)
        print("\n--- Week 2 (Upgraded) Complete ---")