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

    node_features = {}

    for node in G.nodes():
        in_degree  = G.in_degree(node)
        out_degree = G.out_degree(node)

        incoming = list(G.in_edges(node, data=True))
        outgoing = list(G.out_edges(node, data=True))

        total_received = sum(d['amount'] for _, _, d in incoming)
        total_sent     = sum(d['amount'] for _, _, d in outgoing)
        avg_received   = total_received / in_degree if in_degree > 0 else 0

        # FEATURE 1: time_span_minutes
        # How many minutes between the first and last incoming payment?
        # Mule centers get 30+ payments within ~2 hours.
        # Normal merchants get payments spread across a full day or longer.
        if incoming:
            timestamps       = [d['timestamp'] for _, _, d in incoming]
            time_span_minutes = (max(timestamps) - min(timestamps)).total_seconds() / 60
        else:
            time_span_minutes = 0

        # FEATURE 2: unique_senders_ratio
        # What fraction of senders are unique (one-time)?
        # Mule smurfs are freshly created single-use accounts.
        # Real customers are repeat visitors who also shop elsewhere.
        senders              = [s for s, _, _ in incoming]
        unique_senders_ratio = len(set(senders)) / in_degree if in_degree > 0 else 0

        # FEATURE 3: burst_score
        # Transactions per minute — high value = money arriving in rapid automated bursts.
        # This is the signature of a coordinated attack, not natural human shopping.
        burst_score = in_degree / time_span_minutes if time_span_minutes > 0 else in_degree

        # FEATURE 4: fan_in_fan_out_ratio  ← SOLVES THE CHAI SHOP PROBLEM
        # How many accounts paid INTO this one vs how many this one paid OUT TO?
        # Mule center:  30 senders in, 1 destination out  → ratio = 30.0  🚩
        # Chai shop:    80 customers in, 15 payees out     → ratio =  5.3  ✅
        # This single feature cleanly separates high-volume merchants from mule centers.
        fan_in_fan_out_ratio = in_degree / max(out_degree, 1)

        # FEATURE 5: neighbor_freshness_score  ← ALSO SOLVES THE CHAI SHOP PROBLEM
        # Of all the accounts that paid into this one, what fraction ONLY ever paid
        # this one account and nobody else?
        # Mule smurfs are single-use — they sent to the mule center and disappeared.
        # Real customers also pay Zomato, auto drivers, other shops, etc.
        fresh_senders = 0
        for sender in senders:
            if G.out_degree(sender) <= 1:
                fresh_senders += 1
        neighbor_freshness_score = fresh_senders / max(in_degree, 1)

        node_features[node] = {
            'in_degree':              in_degree,
            'out_degree':             out_degree,
            'total_received':         round(total_received, 2),
            'total_sent':             round(total_sent, 2),
            'avg_received':           round(avg_received, 2),
            'time_span_minutes':      round(time_span_minutes, 1),
            'unique_senders_ratio':   round(unique_senders_ratio, 2),
            'burst_score':            round(burst_score, 2),
            'fan_in_fan_out_ratio':   round(fan_in_fan_out_ratio, 2),
            'neighbor_freshness_score': round(neighbor_freshness_score, 2),
            '_amounts':               [d['amount'] for _, _, d in incoming],
        }

    # FEATURE 6: low_amount_clustering  (ADAPTIVE — no hardcoded Rs.500)
    # What fraction of incoming payments are below the 20th percentile of ALL
    # amounts in this dataset, AND how tightly clustered are those amounts?
    # Mule rings deliberately keep amounts similar and low. Real merchants receive
    # wildly varied amounts. This recalculates for any dataset automatically.
    all_amounts      = df['amount'].tolist()
    low_amount_cutoff = pd.Series(all_amounts).quantile(0.20)
    print(f"\nAdaptive low-amount threshold (20th percentile): Rs.{low_amount_cutoff:.2f}")

    for node in node_features:
        amounts = node_features[node].pop('_amounts')
        if amounts:
            fraction_low = sum(1 for a in amounts if a < low_amount_cutoff) / len(amounts)
            mean_amt     = sum(amounts) / len(amounts)
            if len(amounts) > 1 and mean_amt > 0:
                variance              = sum((a - mean_amt)**2 for a in amounts) / len(amounts)
                std_dev               = variance ** 0.5
                coefficient_of_variation = std_dev / mean_amt
            else:
                coefficient_of_variation = 1.0
            uniformity_score      = max(0, 1 - coefficient_of_variation)
            low_amount_clustering = round((fraction_low * 0.6) + (uniformity_score * 0.4), 3)
        else:
            low_amount_clustering = 0
        node_features[node]['low_amount_clustering'] = low_amount_clustering

    nx.set_node_attributes(G, node_features)

    # SUSPICION SCORE
    # burst_score is the dominant signal — it catches rapid coordinated attacks.
    # fan_in_fan_out_ratio directly separates merchants from mule centers.
    # neighbor_freshness_score catches the single-use smurf pattern.
    # in_degree is capped at 50 so raw customer count alone can't dominate.
    for node in node_features:
        f = node_features[node]
        if f['in_degree'] < 5:
            f['suspicion_score'] = 0.0
            continue
        f['suspicion_score'] = round(
            (f['burst_score']              * 60   * 0.35) +
            (f['fan_in_fan_out_ratio']     * 0.5  * 0.25) +
            (f['neighbor_freshness_score'] * 30   * 0.20) +
            (f['low_amount_clustering']    * 30   * 0.10) +
            (min(f['in_degree'], 50)               * 0.10),
            2
        )
    nx.set_node_attributes(G, node_features)

    sorted_by_suspicion = sorted(
        node_features.items(),
        key=lambda x: x[1]['suspicion_score'],
        reverse=True
    )

    print("\n--- Top 5 most suspicious accounts ---")
    for account, f in sorted_by_suspicion[:5]:
        print(f"\n  {account}")
        print(f"  in_degree: {f['in_degree']}  |  burst_score: {f['burst_score']}  |  "
              f"fan_ratio: {f['fan_in_fan_out_ratio']}  |  "
              f"neighbor_fresh: {f['neighbor_freshness_score']}  |  "
              f"suspicion: {f['suspicion_score']}")

    os.makedirs('model', exist_ok=True)
    with open('model/transaction_graph.pkl', 'wb') as f:
        pickle.dump(G, f)
    print("\nGraph saved to model/transaction_graph.pkl")

    return G, sorted_by_suspicion


def plot_top_mule_ring(G, sorted_by_suspicion, save_path='model/mule_ring_plot.png'):
    print("\n--- Drawing network map ---")

    top_account, top_features = sorted_by_suspicion[0]
    fraud_senders = [s for s, _, d in G.in_edges(top_account, data=True)
                     if d.get('is_fraud') == 1]
    normal_nodes  = [n for n in G.nodes()
                     if 'smurf' not in str(n) and 'mule' not in str(n)][:40]

    sub_nodes = [top_account] + fraud_senders + normal_nodes
    subgraph  = G.subgraph(sub_nodes)

    fig = plt.figure(figsize=(14, 10), facecolor='#FFFFFF')
    gs      = fig.add_gridspec(1, 4)
    ax_main = fig.add_subplot(gs[0, :3])
    ax_side = fig.add_subplot(gs[0, 3])
    ax_main.set_facecolor('#FFFFFF')
    ax_side.set_facecolor('#F7F7F9')
    ax_side.axis('off')

    pos = {top_account: (0.5, 0.0)}
    for idx, node in enumerate(fraud_senders):
        angle    = (2 * math.pi * idx) / max(len(fraud_senders), 1)
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

    nx.draw_networkx_nodes(subgraph, pos, ax=ax_main,
                           node_color=node_colors, node_size=node_sizes,
                           edgecolors='black', linewidths=1.3)

    edge_colors, edge_widths = [], []
    for u, v in subgraph.edges():
        amt      = G[u][v].get('amount', 0)
        is_fraud = G[u][v].get('is_fraud') == 1
        edge_colors.append('#CC0000' if is_fraud else '#999999')
        edge_widths.append(0.6 + (amt / 500) * 2.2)

    nx.draw_networkx_edges(subgraph, pos, ax=ax_main,
                           edge_color=edge_colors, width=edge_widths,
                           arrows=True, arrowsize=10, alpha=0.75)

    nx.draw_networkx_labels(subgraph, pos, ax=ax_main,
                            labels={top_account: 'MULE\nTARGET'},
                            font_size=9, font_color='white', font_weight='bold')

    ax_main.set_title("MuleGuard Network Topology Analysis",
                      fontsize=16, color='black', fontweight='bold', pad=14)
    ax_main.text(0.5, 1.04,
                 "Yellow = fraud smurfs   |   Blue = normal accounts",
                 transform=ax_main.transAxes,
                 ha='center', fontsize=10, color='#333333')
    ax_main.axis('off')

    legend_handles = [
        mpatches.Patch(color='#990000', label='Mule Aggregator'),
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
        f"Fan in/out ratio:\n  {top_features['fan_in_fan_out_ratio']}\n\n"
        f"Neighbor freshness:\n  {top_features['neighbor_freshness_score']}\n\n"
        f"Low-amt clustering:\n  {top_features['low_amount_clustering']}\n\n"
        f"{'-'*22}\n"
        f"SUSPICION SCORE\n"
        f"  {top_features['suspicion_score']}"
    )
    ax_side.text(0.05, 0.95, stats_text,
                 transform=ax_side.transAxes,
                 fontsize=9.5, va='top', family='monospace', color='#1a1a1a',
                 bbox=dict(boxstyle='round,pad=0.6',
                           facecolor='white', edgecolor='#990000', linewidth=1.2))

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, facecolor='#FFFFFF')
    print(f"Chart saved to {save_path}")
    plt.close()


if __name__ == "__main__":
    graph, ranked_accounts = build_transaction_graph()
    if graph is not None and ranked_accounts is not None:
        plot_top_mule_ring(graph, ranked_accounts)
        print("\n--- Week 2 (Final) Complete ---")