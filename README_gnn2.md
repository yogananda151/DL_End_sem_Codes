# 🕸️ Graph Convolutional Network (GCN) on Cora Dataset

## 📌 Overview

This script implements a **Graph Convolutional Network (GCN)** to perform **node classification** on the [Cora citation dataset](https://linqs.soe.ucsc.edu/data). Each node represents a research paper, edges represent citations, and the goal is to classify papers into one of 7 research topics.

---

## 🗂️ File

`gnn2.py`

---

## 📦 Libraries Used

| Library | Purpose |
|---|---|
| `torch_geometric` | Graph deep learning framework (graphs, GCN layers) |
| `torch` | PyTorch for tensor operations and model training |
| `collections.Counter` | Counting occurrences of node degrees |
| `matplotlib.pyplot` | Plotting the node degree distribution |

---

## 🔢 Code Walkthrough — Line by Line

### 📥 Imports (Lines 8–11)

```python
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import degree
from collections import Counter
import matplotlib.pyplot as plt
```
- `Planetoid` → Built-in dataset loader for graph benchmark datasets like Cora, CiteSeer, PubMed.
- `degree` → Computes the degree (number of connections/edges) of each node in the graph.
- `Counter` → Python dictionary subclass that counts how many times each degree value appears.
- `matplotlib.pyplot` → For plotting the degree distribution bar chart.

---

### 📊 Loading Dataset & Degree Distribution Plot (Lines 13–21)

```python
dataset = Planetoid(root=".", name="Cora")
```
- Downloads and loads the **Cora** dataset into the current directory (`"."`).
- Cora has: **2,708 nodes** (papers), **5,429 edges** (citations), **7 classes** (topics), **1,433 features per node**.

```python
data = dataset[0]
```
- Retrieves the first (and only) graph object from the dataset.
- `data` contains:
  - `data.x` → Node feature matrix: shape (2708, 1433)
  - `data.edge_index` → Edge connectivity: shape (2, num_edges) in COO format
  - `data.y` → Node labels: shape (2708,), values 0–6
  - `data.train_mask`, `data.val_mask`, `data.test_mask` → Boolean masks for splits

```python
degrees = degree(data.edge_index[0]).numpy()
```
- `data.edge_index[0]` → Source node indices of all edges (first row of the edge index tensor).
- `degree(...)` → Counts how many edges each node has (its degree). Returns a tensor of size (num_nodes,).
- `.numpy()` → Converts to NumPy array for use with Counter and matplotlib.

```python
numbers = Counter(degrees)
```
- Counts how many nodes have each degree value.
- Example: `{1.0: 500, 2.0: 300, ...}` means 500 nodes have degree 1, etc.

```python
fig, ax = plt.subplots()
ax.set_xlabel('Node degree')
ax.set_ylabel('Number of nodes')
plt.bar(numbers.keys(), numbers.values())
```
- Creates a bar chart showing the degree distribution of the graph.
- `numbers.keys()` → Degree values (x-axis).
- `numbers.values()` → Count of nodes with that degree (y-axis).

---

### 📥 Additional Imports (Lines 24–26)

```python
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
```
- `torch` → Core tensor and autograd operations.
- `F` → Functional API (e.g., `F.log_softmax` for producing log-probability outputs).
- `GCNConv` → PyTorch Geometric's implementation of the **Graph Convolutional Layer** from the paper *"Semi-Supervised Classification with Graph Convolutional Networks"* (Kipf & Welling, 2017).

---

### 📐 Accuracy Helper Function (Lines 28–29)

```python
def accuracy(pred_y, y):
    return ((pred_y == y).sum() / len(y)).item()
```
- `pred_y` → Predicted class labels (integers).
- `y` → True class labels (integers).
- `(pred_y == y).sum()` → Counts how many predictions are correct.
- `/ len(y)` → Divides by total number of predictions to get accuracy ratio.
- `.item()` → Converts single-element tensor to a plain Python float.

---

### 🏗️ GCN Model Class (Lines 32–65)

```python
class GCN(torch.nn.Module):
    def __init__(self, dim_in, dim_h, dim_out):
        super().__init__()
```
- Defines the `GCN` class inheriting from `torch.nn.Module`.
- `dim_in` → Size of input node features (1433 for Cora).
- `dim_h` → Size of the hidden layer (16 in this script).
- `dim_out` → Number of output classes (7 for Cora).
- `super().__init__()` → Initializes the parent `nn.Module`.

```python
        self.gcn1 = GCNConv(dim_in, dim_h)
        self.gcn2 = GCNConv(dim_h, dim_out)
```
- **First GCN layer**: Takes node features (1433-dim) and produces 16-dim embeddings.
- **Second GCN layer**: Takes 16-dim embeddings and produces 7-dim class scores.
- Each `GCNConv` layer aggregates information from neighboring nodes using the graph structure.

#### Forward Pass

```python
    def forward(self, x, edge_index):
        h = self.gcn1(x, edge_index)
        h = torch.relu(h)
        h = self.gcn2(h, edge_index)
        return F.log_softmax(h, dim=1)
```
- `x` → Node feature matrix.
- `edge_index` → Graph edge connectivity (which nodes are connected).
- `self.gcn1(x, edge_index)` → First graph convolution: each node aggregates features from its neighbors, output shape: (2708, 16).
- `torch.relu(h)` → Applies ReLU activation; sets negative values to 0, introduces non-linearity.
- `self.gcn2(h, edge_index)` → Second graph convolution: produces class scores, output shape: (2708, 7).
- `F.log_softmax(h, dim=1)` → Converts raw scores to log-probabilities along class dimension. Used with `NLLLoss` or `CrossEntropyLoss`.

#### Fit (Training) Method

```python
    def fit(self, data, epochs):
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01, weight_decay=5e-4)
```
- `CrossEntropyLoss` → Loss function for multi-class classification.
- `Adam` → Adaptive optimizer. `lr=0.01` is the learning rate.
- `weight_decay=5e-4` → L2 regularization (adds a penalty for large weights to prevent overfitting).

```python
        self.train()
```
- Puts the model in training mode (enables dropout, batch norm training behavior if used).

```python
        for epoch in range(epochs+1):
            optimizer.zero_grad()
```
- Loops over `epochs+1` iterations (0 to `epochs` inclusive).
- `zero_grad()` → Clears accumulated gradients from the previous step.

```python
            out = self(data.x, data.edge_index)
```
- Forward pass: runs the full graph through the GCN. `out` shape: (2708, 7) — scores for all nodes.

```python
            loss = criterion(out[data.train_mask], data.y[data.train_mask])
```
- `data.train_mask` → Boolean mask selecting only the training nodes (140 nodes in Cora).
- Loss is computed **only on training nodes** — transductive learning.

```python
            acc = accuracy(out[data.train_mask].argmax(dim=1), data.y[data.train_mask])
```
- `argmax(dim=1)` → Gets the predicted class (index of max score) for each training node.
- Computes training accuracy.

```python
            loss.backward()
            optimizer.step()
```
- `loss.backward()` → Computes gradients via backpropagation through the graph.
- `optimizer.step()` → Updates model weights.

```python
            if(epoch % 20 == 0):
                val_loss = criterion(out[data.val_mask], data.y[data.val_mask])
                val_acc = accuracy(out[data.val_mask].argmax(dim=1), data.y[data.val_mask])
                print(f'Epoch {epoch:>3} | Train Loss:{loss:.3f} | Train Acc: {acc*100:>5.2f}% | Val Loss:{val_loss:.2f} | Val Acc: {val_acc*100:.2f}%')
```
- Every 20 epochs, computes and prints validation loss and accuracy.
- `data.val_mask` → Boolean mask for validation nodes (500 nodes in Cora).

---

### 🧪 Test Function (Lines 60–65)

```python
@torch.no_grad()
def test(self, data):
    self.eval()
    out = self(data.x, data.edge_index)
    acc = accuracy(out.argmax(dim=1)[data.test_mask], data.y[data.test_mask])
    return acc
```
- `@torch.no_grad()` → Decorator that disables gradient computation (faster, less memory during inference).
- `self.eval()` → Puts model in evaluation mode (disables dropout, etc.).
- Forward pass on the full graph, then selects test nodes using `data.test_mask` (1000 nodes in Cora).
- Returns test accuracy.

> ⚠️ **Note**: The `test` function is defined outside the class (indentation issue in source code), so it is a standalone function rather than a method of `GCN`.

---

### 🚀 Training Execution (Lines 67–69)

```python
gcn = GCN(dataset.num_features, 16, dataset.num_classes)
```
- Creates the GCN model:
  - `dataset.num_features` → 1433 (input feature size for Cora).
  - `16` → Hidden layer size.
  - `dataset.num_classes` → 7 (number of output classes).

```python
print(gcn)
```
- Prints the model structure showing both GCN layers.

```python
gcn.fit(data, epochs=100)
```
- Trains the model for 100 epochs on the Cora dataset.
- Prints training/validation metrics every 20 epochs.

---

## 🏛️ GCN Architecture Summary

```
Input Node Features: (2708, 1433)
         ↓
GCNConv(1433 → 16) + ReLU  →  (2708, 16)
         ↓
GCNConv(16 → 7)             →  (2708, 7)
         ↓
LogSoftmax                  →  (2708, 7) log-probabilities
```

---

## 🔑 Key Concepts

| Concept | Explanation |
|---|---|
| **Graph Convolution** | Each node aggregates feature information from its neighbors |
| **Transductive Learning** | All nodes (train + val + test) are used in forward pass, but loss only on train nodes |
| **Node Classification** | Each node gets a predicted class label |
| **Train/Val/Test Masks** | Boolean arrays that select subsets of nodes for each phase |

---

## ▶️ How to Run

```bash
pip install torch torch-geometric matplotlib
python gnn2.py
```

---

## 📈 Expected Output

- Degree distribution bar chart displayed
- Training progress every 20 epochs:
  ```
  Epoch   0 | Train Loss:1.946 | Train Acc: 14.29% | Val Loss:1.95 | Val Acc: 14.20%
  Epoch  20 | Train Loss:0.432 | Train Acc: 94.29% | Val Loss:1.08 | Val Acc: 73.20%
  ...
  Epoch 100 | Train Loss:0.012 | Train Acc:100.00% | Val Loss:0.74 | Val Acc: 78.40%
  ```
- Typical test accuracy: **~81%** on Cora
