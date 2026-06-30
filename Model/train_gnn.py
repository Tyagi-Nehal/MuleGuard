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
# WEEK 3: TRAIN THE GRAPH NEURAL NETWORK
# -------------------------------------------------------------------------
# Goal: teach an AI to look at the graph we built in Week 2 and predict,
# for every single account, whether it behaves like a mule account or not -
# WITHOUT us telling it the answer in advance.
# -------------------------------------------------------------------------

FEATURE_NAMES = [
    'in_degree', 'out_degree', 'total_received', 'total_sent',
    'avg_received', 'time_span_minutes', 'unique_senders_ratio',
    'burst_score', 'low_amount_clustering'
]


def load_graph(path='model/transaction_graph.pkl'):
    """Load the NetworkX graph we built and saved in Week 2."""
    print("--- Loading Week 2 graph ---")
    with open(path, 'rb') as f:
        G = pickle.load(f)
    print(f"Loaded graph: {G.number_of_nodes()} accounts, {G.number_of_edges()} transactions")
    return G


def graph_to_pyg_data(G):
    """
    Convert a NetworkX graph into the format PyTorch Geometric understands.

    PyTorch Geometric doesn't work with NetworkX graphs directly - it needs
    everything as numerical tensors (PyTorch's word for "arrays of numbers").
    This function does that translation.
    """
    print("\n--- Converting graph to PyTorch Geometric format ---")

    # Every account needs a fixed numerical ID (0, 1, 2, 3...) instead of its
    # text name (like "smurf_node_0_5@paytm"). We build that lookup table here.
    node_list = list(G.nodes())
    node_to_idx = {node: idx for idx, node in enumerate(node_list)}

    # --- Build the FEATURE MATRIX (x) ---
    # This is a table: one row per account, one column per feature we calculated
    # in Week 2 (in_degree, burst_score, low_amount_clustering, etc.)
    # This is literally the "profile card" data from Week 2, just reshaped
    # into the numbers-only format the GNN needs.
    feature_rows = []
    for node in node_list:
        node_data = G.nodes[node]
        row = [node_data.get(feat, 0.0) for feat in FEATURE_NAMES]
        feature_rows.append(row)

    x = torch.tensor(feature_rows, dtype=torch.float)

    # --- Normalize the features ---
    # Different features have wildly different ranges - in_degree might be 0-80,
    # but total_received might be 0-50000. Without normalizing, the GNN would
    # think total_received matters way more just because the numbers are bigger.
    # This scales every column to have mean 0 and standard deviation 1.
    x_mean = x.mean(dim=0, keepdim=True)
    x_std = x.std(dim=0, keepdim=True) + 1e-8  # tiny number added to avoid divide-by-zero
    x = (x - x_mean) / x_std

    # --- Build the EDGE LIST (edge_index) ---
    # PyTorch Geometric wants edges as two parallel lists:
    # source_nodes = [0, 0, 1, 2, ...]   target_nodes = [3, 5, 2, 7, ...]
    # meaning: edge 0->3, edge 0->5, edge 1->2, edge 2->7, and so on.
    sources, targets = [], []
    for u, v in G.edges():
        sources.append(node_to_idx[u])
        targets.append(node_to_idx[v])
    edge_index = torch.tensor([sources, targets], dtype=torch.long)

    # --- Build the LABELS (y) ---
    # For every account, was it EVER involved in a fraud transaction
    # (either sending or receiving)? This is the "answer key" the model
    # learns from during training, and is graded against during testing.
    labels = []
    for node in node_list:
        involved_in_fraud = 0
        for _, _, data in G.in_edges(node, data=True):
            if data.get('is_fraud') == 1:
                involved_in_fraud = 1
                break
        if involved_in_fraud == 0:
            for _, _, data in G.out_edges(node, data=True):
                if data.get('is_fraud') == 1:
                    involved_in_fraud = 1
                    break
        labels.append(involved_in_fraud)

    y = torch.tensor(labels, dtype=torch.long)

    print(f"Feature matrix shape: {x.shape}  (accounts x features)")
    print(f"Edge index shape: {edge_index.shape}  (2 x number of transactions)")
    print(f"Fraud-labeled accounts: {y.sum().item()} out of {len(y)} total accounts")

    data = Data(x=x, edge_index=edge_index, y=y)
    return data, node_list


class MuleGuardGCN(torch.nn.Module):
    """
    The actual AI model - a Graph Convolutional Network (GCN).

    This is a 2-layer network:
    Layer 1: each account looks at its DIRECT neighbours and blends in their info
    Layer 2: each account looks at its neighbours' updated info (which already
             contains THEIR neighbours' info) - so by now, information from
             2 hops away has reached every node. This is the "message passing"
             concept we discussed earlier.
    Final step: a small classifier decides fraud (1) or normal (0) for each account.
    """
    def __init__(self, num_features, hidden_channels=16):
        super().__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.classifier = Linear(hidden_channels, 2)  # 2 classes: normal, fraud

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)              # introduces non-linearity, lets the model learn complex patterns
        x = F.dropout(x, p=0.3, training=self.training)  # randomly "forgets" some info during training - prevents overfitting
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        out = self.classifier(x)
        return out


def train_model(data, num_epochs=100):
    """
    The actual training loop.

    'Training' means: show the model examples, let it guess, tell it how wrong
    it was, let it adjust slightly, repeat hundreds of times until it gets good.
    """
    print("\n--- Splitting accounts into training set and test set ---")

    num_nodes = data.x.shape[0]
    indices = np.arange(num_nodes)

    # We split which ACCOUNTS the model is allowed to "study" (train) vs
    # which accounts are held back as a "surprise quiz" (test) to check if
    # it actually learned the pattern, rather than just memorizing answers.
    train_idx, test_idx = train_test_split(
        indices, test_size=0.3, random_state=42, stratify=data.y.numpy()
    )

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[train_idx] = True
    test_mask[test_idx] = True

    print(f"Training on {train_mask.sum().item()} accounts, testing on {test_mask.sum().item()} accounts")

    model = MuleGuardGCN(num_features=data.x.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    # --- Handle class imbalance ---
    # Remember: fraud accounts are a small minority (~3% in our dataset).
    # Without correcting for this, the model could get 97% accuracy by just
    # always guessing "normal" for everyone - technically high accuracy,
    # completely useless in practice. weight tells the loss function to
    # penalize missed fraud cases much more heavily than missed normal cases.
    num_fraud = (data.y == 1).sum().item()
    num_normal = (data.y == 0).sum().item()
    fraud_weight = num_normal / max(num_fraud, 1)
    class_weights = torch.tensor([1.0, fraud_weight], dtype=torch.float)
    print(f"Class imbalance correction: fraud class weighted {fraud_weight:.1f}x higher than normal class")

    print("\n--- Training started ---")
    model.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[train_mask], data.y[train_mask], weight=class_weights)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}  -  Loss: {loss.item():.4f}")

    print("--- Training complete ---")
    return model, train_mask, test_mask


def evaluate_model(model, data, test_mask):
    """
    The 'surprise quiz' - check how well the model performs on accounts
    it never saw during training.
    """
    print("\n--- Evaluating on held-out test accounts ---")
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        probs = F.softmax(out, dim=1)
        preds = out.argmax(dim=1)

    test_preds = preds[test_mask]
    test_labels = data.y[test_mask]

    correct = (test_preds == test_labels).sum().item()
    total = test_mask.sum().item()
    accuracy = correct / total

    # Precision/Recall matter MORE than accuracy here, because of class imbalance.
    true_positives = ((test_preds == 1) & (test_labels == 1)).sum().item()
    false_positives = ((test_preds == 1) & (test_labels == 0)).sum().item()
    false_negatives = ((test_preds == 0) & (test_labels == 1)).sum().item()

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"Accuracy:  {accuracy:.2%}")
    print(f"Precision: {precision:.2%}  (of accounts we flagged as fraud, how many actually were)")
    print(f"Recall:    {recall:.2%}  (of all actual fraud accounts, how many we caught)")
    print(f"F1 Score:  {f1:.2%}  (balance between precision and recall)")

    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}


def save_model(model, node_list, metrics, path='model/muleguard_gnn.pt'):
    """Save the trained model so Week 5/6 can load it without retraining."""
    os.makedirs('model', exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'node_list': node_list,
        'feature_names': FEATURE_NAMES,
        'metrics': metrics,
    }, path)
    print(f"\nModel saved to {path}")


if __name__ == "__main__":
    G = load_graph()
    data, node_list = graph_to_pyg_data(G)
    model, train_mask, test_mask = train_model(data, num_epochs=100)
    metrics = evaluate_model(model, data, test_mask)
    save_model(model, node_list, metrics)
    print("\n--- Week 3 Complete: GNN trained and saved ---")