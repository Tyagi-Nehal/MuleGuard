import pickle
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch.nn import Linear

# -------------------------------------------------------------------------
# WEEK 3 BONUS: TEST THE TRAINED MODEL
# -------------------------------------------------------------------------
# This script loads your already-trained model and runs it on the FULL
# dataset, then shows you exactly which accounts it flags as fraud -
# so you can manually verify it's catching the right ones.
# -------------------------------------------------------------------------

FEATURE_NAMES = [
    'in_degree', 'out_degree', 'total_received', 'total_sent',
    'avg_received', 'time_span_minutes', 'unique_senders_ratio',
    'burst_score', 'low_amount_clustering'
]


# This class definition must be IDENTICAL to the one in train_gnn.py -
# PyTorch needs to know the model's "shape" before it can load the saved weights.
class MuleGuardGCN(torch.nn.Module):
    def __init__(self, num_features, hidden_channels=16):
        super().__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.classifier = Linear(hidden_channels, 2)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        out = self.classifier(x)
        return out


def load_graph(path='model/transaction_graph.pkl'):
    with open(path, 'rb') as f:
        G = pickle.load(f)
    return G


def graph_to_features(G, node_list):
    """Same conversion logic as training - must match exactly."""
    feature_rows = []
    for node in node_list:
        node_data = G.nodes[node]
        row = [node_data.get(feat, 0.0) for feat in FEATURE_NAMES]
        feature_rows.append(row)
    x = torch.tensor(feature_rows, dtype=torch.float)

    x_mean = x.mean(dim=0, keepdim=True)
    x_std = x.std(dim=0, keepdim=True) + 1e-8
    x = (x - x_mean) / x_std

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
    node_list = checkpoint['node_list']
    metrics = checkpoint['metrics']

    print(f"Model was trained with: Accuracy {metrics['accuracy']:.2%} | "
          f"Precision {metrics['precision']:.2%} | Recall {metrics['recall']:.2%}")

    # Rebuild the same graph + features used during training
    G = load_graph()
    x, edge_index = graph_to_features(G, node_list)

    # Rebuild the model "shape" and load the trained weights into it
    model = MuleGuardGCN(num_features=x.shape[1])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # switches off dropout - we want the model's best, final answer now

    print(f"\n--- Running the model on all {len(node_list)} accounts ---")
    with torch.no_grad():
        out = model(x, edge_index)
        probs = F.softmax(out, dim=1)  # converts raw scores into 0-1 probabilities
        fraud_probs = probs[:, 1]      # column 1 = probability of being fraud

    # Pair each account with its fraud probability, sort highest first
    results = list(zip(node_list, fraud_probs.tolist()))
    results.sort(key=lambda x: x[1], reverse=True)

    print(f"\n--- Top {top_n} accounts the model considers MOST suspicious ---\n")
    for rank, (account, prob) in enumerate(results[:top_n], 1):
        flag = "🚩 FLAGGED AS FRAUD" if prob >= 0.5 else "   (below threshold)"
        print(f"{rank:2d}. {account:<40s}  fraud probability: {prob:.1%}   {flag}")

    # Cross-check against the ACTUAL fraud labels you know from Week 1
    actual_fraud_accounts = set()
    for node in G.nodes():
        for _, _, data in G.in_edges(node, data=True):
            if data.get('is_fraud') == 1:
                actual_fraud_accounts.add(node)
        for _, _, data in G.out_edges(node, data=True):
            if data.get('is_fraud') == 1:
                actual_fraud_accounts.add(node)

    flagged_accounts = set(acc for acc, prob in results if prob >= 0.5)
    correctly_caught = flagged_accounts & actual_fraud_accounts
    missed = actual_fraud_accounts - flagged_accounts
    false_alarms = flagged_accounts - actual_fraud_accounts

    print(f"\n--- Sanity check against known fraud labels ---")
    print(f"Total accounts flagged by model: {len(flagged_accounts)}")
    print(f"Correctly caught real fraud accounts: {len(correctly_caught)}")
    print(f"Missed fraud accounts: {len(missed)}")
    print(f"False alarms (flagged but actually normal): {len(false_alarms)}")

    if missed:
        print(f"\nMissed accounts (model failed to flag these): {list(missed)[:5]}")


if __name__ == "__main__":
    run_inference()