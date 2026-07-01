import pickle
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split

# -------------------------------------------------------------------------
# WEEK 3: TRAIN THE GRAPH NEURAL NETWORK  (Final version — 11 features)
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
    'fan_in_fan_out_ratio',       # NEW: separates merchants from mule centers
    'neighbor_freshness_score',   # NEW: catches single-use smurf accounts
]


def load_graph(path='model/transaction_graph.pkl'):
    print("--- Loading graph ---")
    with open(path, 'rb') as f:
        G = pickle.load(f)
    print(f"Loaded: {G.number_of_nodes()} accounts, {G.number_of_edges()} transactions")
    return G


def graph_to_pyg_data(G):
    print("\n--- Converting to PyTorch Geometric format ---")

    node_list   = list(G.nodes())
    node_to_idx = {node: idx for idx, node in enumerate(node_list)}

    # Feature matrix: one row per account, one column per feature
    feature_rows = []
    for node in node_list:
        node_data = G.nodes[node]
        row = [node_data.get(feat, 0.0) for feat in FEATURE_NAMES]
        feature_rows.append(row)

    x = torch.tensor(feature_rows, dtype=torch.float)

    # Normalize so no single feature dominates due to scale differences
    x_mean = x.mean(dim=0, keepdim=True)
    x_std  = x.std(dim=0, keepdim=True) + 1e-8
    x      = (x - x_mean) / x_std

    # Edge list: who sent money to whom
    sources, targets = [], []
    for u, v in G.edges():
        sources.append(node_to_idx[u])
        targets.append(node_to_idx[v])
    edge_index = torch.tensor([sources, targets], dtype=torch.long)

    # Labels: was this account ever involved in fraud?
    labels = []
    for node in node_list:
        involved = 0
        for _, _, d in G.in_edges(node, data=True):
            if d.get('is_fraud') == 1:
                involved = 1
                break
        if not involved:
            for _, _, d in G.out_edges(node, data=True):
                if d.get('is_fraud') == 1:
                    involved = 1
                    break
        labels.append(involved)

    y = torch.tensor(labels, dtype=torch.long)

    print(f"Feature matrix: {x.shape}  (accounts x features)")
    print(f"Edge index:     {edge_index.shape}")
    print(f"Fraud accounts: {y.sum().item()} out of {len(y)} total")

    return Data(x=x, edge_index=edge_index, y=y), node_list


class MuleGuardGCN(torch.nn.Module):
    """
    2-layer Graph Convolutional Network.
    Layer 1: each account absorbs info from direct neighbors
    Layer 2: absorbs info from neighbors-of-neighbors (2 hops)
    Classifier: decides fraud or normal based on absorbed info
    """
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


def train_model(data, num_epochs=150):
    print("\n--- Splitting into train / test sets ---")

    num_nodes = data.x.shape[0]
    indices   = np.arange(num_nodes)

    train_idx, test_idx = train_test_split(
        indices, test_size=0.3, random_state=42, stratify=data.y.numpy()
    )

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask  = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[train_idx] = True
    test_mask[test_idx]   = True

    print(f"Training on {train_mask.sum().item()} accounts, "
          f"testing on {test_mask.sum().item()} accounts")

    model     = MuleGuardGCN(num_features=data.x.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    # Class imbalance correction: fraud is rare, so missing fraud is penalized more
    num_fraud  = (data.y == 1).sum().item()
    num_normal = (data.y == 0).sum().item()
    fraud_weight   = num_normal / max(num_fraud, 1)
    class_weights  = torch.tensor([1.0, fraud_weight], dtype=torch.float)
    print(f"Fraud class weighted {fraud_weight:.1f}x higher to handle imbalance")

    print("\n--- Training ---")
    model.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        out  = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[train_mask], data.y[train_mask],
                               weight=class_weights)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 30 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}  Loss: {loss.item():.4f}")

    print("--- Training complete ---")
    return model, train_mask, test_mask


def evaluate_model(model, data, test_mask):
    print("\n--- Evaluating on held-out test accounts ---")
    model.eval()
    with torch.no_grad():
        out   = model(data.x, data.edge_index)
        preds = out.argmax(dim=1)

    test_preds  = preds[test_mask]
    test_labels = data.y[test_mask]

    correct  = (test_preds == test_labels).sum().item()
    total    = test_mask.sum().item()
    accuracy = correct / total

    tp = ((test_preds == 1) & (test_labels == 1)).sum().item()
    fp = ((test_preds == 1) & (test_labels == 0)).sum().item()
    fn = ((test_preds == 0) & (test_labels == 1)).sum().item()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"Accuracy:  {accuracy:.2%}")
    print(f"Precision: {precision:.2%}  (flagged accounts that were actually fraud)")
    print(f"Recall:    {recall:.2%}  (real fraud accounts we caught)")
    print(f"F1 Score:  {f1:.2%}  (overall balance)")

    return {'accuracy': accuracy, 'precision': precision,
            'recall': recall, 'f1': f1}


def save_model(model, node_list, metrics, path='model/muleguard_gnn.pt'):
    os.makedirs('model', exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'node_list':        node_list,
        'feature_names':    FEATURE_NAMES,
        'metrics':          metrics,
    }, path)
    print(f"\nModel saved to {path}")


if __name__ == "__main__":
    G                        = load_graph()
    data, node_list          = graph_to_pyg_data(G)
    model, train_mask, test_mask = train_model(data, num_epochs=150)
    metrics                  = evaluate_model(model, data, test_mask)
    save_model(model, node_list, metrics)
    print("\n--- Week 3 Complete: GNN trained and saved ---")