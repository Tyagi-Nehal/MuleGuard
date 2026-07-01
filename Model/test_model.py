import pickle
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch.nn import Linear

# -------------------------------------------------------------------------
# TEST THE TRAINED MODEL  (Final version — 11 features)
# -------------------------------------------------------------------------

FEATURE_NAMES = [
    'in_degree',
    'out_degree',
    'total_received',
    'total_sent',
    'avg_received',
    'time_span_minutes',
    'unique_senders_ratio',
    'burst_score',
    'low_amount_clustering',
    'fan_in_fan_out_ratio',
    'neighbor_freshness_score',
]


class MuleGuardGCN(torch.nn.Module):
    def __init__(self, num_features, hidden_channels=16):
        super().__init__()
        self.conv1      = GCNConv(num_features, hidden_channels)
        self.conv2      = GCNConv(hidden_channels, hidden_channels)
        self.classifier = Linear(hidden_channels, 2)

    def forward(self, x, edge_index):
        x   = self.conv1(x, edge_index)
        x   = F.relu(x)
        x   = F.dropout(x, p=0.3, training=self.training)
        x   = self.conv2(x, edge_index)
        x   = F.relu(x)
        out = self.classifier(x)
        return out


def load_graph(path='model/transaction_graph.pkl'):
    with open(path, 'rb') as f:
        G = pickle.load(f)
    return G


def graph_to_features(G, node_list):
    feature_rows = []
    for node in node_list:
        node_data = G.nodes[node]
        row = [node_data.get(feat, 0.0) for feat in FEATURE_NAMES]
        feature_rows.append(row)

    x      = torch.tensor(feature_rows, dtype=torch.float)
    x_mean = x.mean(dim=0, keepdim=True)
    x_std  = x.std(dim=0, keepdim=True) + 1e-8
    x      = (x - x_mean) / x_std

    node_to_idx = {node: idx for idx, node in enumerate(node_list)}
    sources, targets = [], []
    for u, v in G.edges():
        sources.append(node_to_idx[u])
        targets.append(node_to_idx[v])
    edge_index = torch.tensor([sources, targets], dtype=torch.long)

    return x, edge_index


def run_inference(model_path='model/muleguard_gnn.pt', top_n=15):
    print("--- Loading trained MuleGuard model ---")
    checkpoint = torch.load(model_path, weights_only=False)
    node_list  = checkpoint['node_list']
    metrics    = checkpoint['metrics']

    print(f"Trained with: Accuracy {metrics['accuracy']:.2%} | "
          f"Precision {metrics['precision']:.2%} | "
          f"Recall {metrics['recall']:.2%} | "
          f"F1 {metrics['f1']:.2%}")

    G = load_graph()
    x, edge_index = graph_to_features(G, node_list)

    model = MuleGuardGCN(num_features=x.shape[1])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"\n--- Running on all {len(node_list)} accounts ---")
    with torch.no_grad():
        out         = model(x, edge_index)
        probs       = F.softmax(out, dim=1)
        fraud_probs = probs[:, 1]

    results = list(zip(node_list, fraud_probs.tolist()))
    results.sort(key=lambda x: x[1], reverse=True)

    print(f"\n--- Top {top_n} most suspicious accounts ---\n")
    for rank, (account, prob) in enumerate(results[:top_n], 1):
        flag = "🚩 FRAUD" if prob >= 0.5 else "  (normal)"
        bar  = "█" * int(prob * 20)
        print(f"{rank:2d}. {account:<42s}  {prob:.1%}  {bar}  {flag}")

    # Cross-check against real fraud labels from Week 1
    actual_fraud = set()
    for node in G.nodes():
        for _, _, d in G.in_edges(node, data=True):
            if d.get('is_fraud') == 1:
                actual_fraud.add(node)
        for _, _, d in G.out_edges(node, data=True):
            if d.get('is_fraud') == 1:
                actual_fraud.add(node)

    flagged        = set(acc for acc, prob in results if prob >= 0.5)
    caught         = flagged & actual_fraud
    missed         = actual_fraud - flagged
    false_alarms   = flagged - actual_fraud

    print(f"\n--- Sanity check ---")
    print(f"Total accounts flagged:          {len(flagged)}")
    print(f"Real fraud accounts caught:      {len(caught)} / {len(actual_fraud)}")
    print(f"Missed (slipped through):        {len(missed)}")
    print(f"False alarms (normal, flagged):  {len(false_alarms)}")

    if missed:
        print(f"\nMissed accounts: {list(missed)[:5]}")
    if false_alarms:
        print(f"\nSample false alarms: {list(false_alarms)[:3]}")


if __name__ == "__main__":
    run_inference()