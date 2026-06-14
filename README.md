# 🛡️ MuleGuard

### Real-Time UPI Micro-Structuring & Mule Account Detection using Graph Neural Networks

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch_Geometric-GNN-orange?style=flat-square&logo=pytorch)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-green?style=flat-square&logo=fastapi)
![React](https://img.shields.io/badge/React-Dashboard-61DAFB?style=flat-square&logo=react)
![Docker](https://img.shields.io/badge/Docker-Deployed-2496ED?style=flat-square&logo=docker)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

---

## 🚨 The Problem

India's UPI network processed **18.68 billion transactions in May 2025 alone**. Fraudsters exploit this scale using a technique called **micro-structuring** — breaking large stolen sums into hundreds of tiny transactions under ₹500, routed through networks of fake "mule" accounts to evade detection.

**Every existing fraud system looks at transactions one by one.**
A ₹300 payment always looks innocent in isolation. But when 200 fake accounts all send ₹300 to the same account within 2 hours — that's ₹60,000 in laundered money that no current system catches.

> UPI fraud cases jumped **85% in FY2024** · Losses crossed **₹1,087 crore** in a single year · Karnataka accounts for **18% of all UPI fraud** in India

---

## 💡 What MuleGuard Does

MuleGuard steps back and looks at the **entire network** of who is sending money to whom — not individual transactions.

- Models all UPI accounts as **nodes** and transactions as **edges** in a graph
- Uses a **Graph Convolutional Network (GCN)** to detect the star-shaped mule ring patterns invisible to per-transaction systems
- Assigns a **fraud risk score (0–1)** to every account in real time
- Generates a **plain-English explanation** for every flag via GNNExplainer
- Visualises fraud rings **lighting up in red** on a live network dashboard

---

## 🏗️ System Architecture

```
UPI Transaction Data
        │
        ▼
┌─────────────────┐
│  Graph Builder  │  ← NetworkX converts transactions into account graph
│   (NetworkX)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   GNN Model     │  ← PyTorch Geometric GCN detects mule ring patterns
│ (PyTorch Geo.)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ GNNExplainer +  │  ← Explains WHY each account was flagged
│     SHAP        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  FastAPI Server │  ← REST API: /detect · /explain · /health
│   (Backend)     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ React Dashboard │  ← Live network graph · Red fraud rings · Analyst tools
│  (Frontend)     │
└─────────────────┘
```

---

## 🧰 Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| Data | Python, NetworkX, Pandas | Generate synthetic UPI graph data |
| AI / ML | PyTorch Geometric (GCN) | Graph Neural Network for fraud detection |
| Explainability | GNNExplainer, SHAP | Plain-English reasons for every flag |
| Experiment Tracking | MLflow | Log and compare training runs |
| Model Monitoring | Evidently AI | Detect data drift in production |
| Backend | FastAPI, Uvicorn | REST API serving the trained model |
| Frontend | React.js, React Force Graph, Tailwind CSS | Live analyst dashboard |
| Deployment | Docker, Railway | Containerised cloud deployment |
| Dataset | PaySim (Kaggle), IEEE-CIS Fraud | Training and benchmarking |

---

## 📁 Project Structure

```
muleguard/
├── data/
│   ├── generate_data.py        ← Synthetic UPI transaction simulator
│   └── transactions.csv        ← Generated dataset
├── model/
│   ├── build_graph.py          ← Converts CSV into NetworkX graph
│   ├── train_gnn.py            ← PyTorch Geometric GCN training
│   ├── explain.py              ← GNNExplainer integration
│   └── muleguard_model.pt      ← Saved trained model
├── backend/
│   ├── main.py                 ← FastAPI server (3 endpoints)
│   └── requirements.txt        ← Python dependencies
├── frontend/
│   ├── src/App.jsx             ← Main React dashboard
│   ├── src/Graph.jsx           ← D3 Force Graph visualisation
│   └── src/Table.jsx           ← Flagged accounts table
├── docker/
│   └── Dockerfile              ← Container configuration
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.11
- Node.js 20 LTS
- Git

### 1. Clone the repository
```bash
git clone https://github.com/Tyagi-Nehal/MuleGuard.git
cd MuleGuard
```

### 2. Install Python dependencies
```bash
python -m pip install -r backend/requirements.txt
```

### 3. Generate the dataset
```bash
python data/generate_data.py
```

### 4. Train the GNN model
```bash
python model/train_gnn.py
```

### 5. Start the backend server
```bash
cd backend
uvicorn main:app --reload
```

### 6. Start the React dashboard
```bash
cd frontend
npm install
npm start
```

Open `http://localhost:3000` in your browser.

---

## 📊 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/detect` | Submit transactions → get fraud risk scores |
| POST | `/explain` | Get plain-English explanation for a flagged account |
| GET | `/health` | Check if the server is running |

---

## 🎯 Evaluation Targets

- AUC-ROC **> 0.88** on held-out test graph
- Inference latency **< 200ms** per batch
- GNNExplainer correctly identifies top edges for **> 85%** of flagged nodes
- Evidently AI drift report auto-generated on distribution shift

---

## 📚 Research References

1. Worldpay from FIS. *Are microtransactions the next big thing in digital payments?* Worldpay Insights, 2022.
2. Moody, K. *Exploring the Role of Microtransactions in the Video Game Industry.* University of Virginia, 2024.
3. NPCI / RBI Annual Report FY2024 — UPI Fraud Statistics.
4. FiosCompliance. *Inside India's UPI Fraud Surge: A Silent Enabler of Illicit Finance*, 2025.

---

## 👨‍💻 Author

**Nehal Tyagi**
Final Year B.Tech — Information Technology
Manipal Institute of Technology, Bengaluru

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Nehal_Tyagi-blue?style=flat-square&logo=linkedin)](https://linkedin.com/in/nehaltyagi/)
[![GitHub](https://img.shields.io/badge/GitHub-Tyagi--Nehal-black?style=flat-square&logo=github)](https://github.com/Tyagi-Nehal)


*Built as a Final Year Minor Specialization Project — Manipal Institute of Technology Bengaluru, 2025–26*
