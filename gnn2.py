# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 12:15:46 2024

@author: mithun
"""

# Import Planetoid benchmark citation network datasets (Cora, CiteSeer, PubMed) from PyTorch Geometric
from torch_geometric.datasets import Planetoid

# Import degree utility function to compute degree (number of connected edges) for each node
from torch_geometric.utils import degree

# Import Counter from Python's standard library to count frequencies of each node degree
from collections import Counter

# Import matplotlib's pyplot for creating visualization plots and bar charts
import matplotlib.pyplot as plt

# Download and load the Cora citation graph dataset into the current directory
dataset = Planetoid(root=".", name="Cora")

# Cora contains a single graph, extract it at index 0
data = dataset[0]

# Calculate node degrees using source node indices (edge_index[0]) and convert to numpy array
degrees = degree(data.edge_index[0]).numpy()

# Count how many nodes have each degree value (e.g., degree 1, degree 2, etc.)
numbers = Counter(degrees)

# Create a figure and axis object for plotting the degree distribution
fig, ax = plt.subplots()

# Label the x-axis as 'Node degree'
ax.set_xlabel('Node degree')

# Label the y-axis as 'Number of nodes'
ax.set_ylabel('Number of nodes')

# Draw a bar chart showing degree values on x-axis and their frequency count on y-axis
plt.bar(numbers.keys(), numbers.values())


# Import core PyTorch library for tensor calculations and deep learning
import torch

# Import functional interface of PyTorch (for activation functions, log_softmax, etc.)
import torch.nn.functional as F

# Import Graph Convolutional Network (GCN) layer from PyTorch Geometric
from torch_geometric.nn import GCNConv

# Define a helper function to calculate classification accuracy (correct predictions / total)
def accuracy(pred_y, y):
    # Compare predictions with true labels, sum correct matches, divide by count, and return as float
    return ((pred_y == y).sum() / len(y)).item()


# Define a 2-layer Graph Convolutional Network (GCN) model class
class GCN(torch.nn.Module):
      # Initialize model with input feature dimension, hidden layer dimension, and output classes
      def __init__(self, dim_in, dim_h, dim_out):
          # Call the parent torch.nn.Module initializer
          super().__init__()
          # 1st GCN layer: transforms node features from dim_in to hidden dimension dim_h
          self.gcn1 = GCNConv(dim_in, dim_h)
          # 2nd GCN layer: transforms hidden features from dim_h to output classes dim_out
          self.gcn2 = GCNConv(dim_h, dim_out)

      # Define the forward pass taking node feature matrix x and graph edge connections edge_index
      def forward(self, x, edge_index):
          # Pass node features and graph structure through the first GCN layer
          h = self.gcn1(x, edge_index)
          # Apply ReLU activation function to add non-linearity
          h = torch.relu(h)
          # Pass features through the second GCN layer
          h = self.gcn2(h, edge_index)
          # Apply Log-Softmax across class dimension (dim=1) to get log probabilities for classification
          return F.log_softmax(h, dim=1)

      # Define the training routine (fit method) for the specified number of epochs
      def fit(self, data, epochs):
          # Use CrossEntropyLoss (equivalent to NLLLoss when using log_softmax)
          criterion = torch.nn.CrossEntropyLoss()
          # Initialize Adam optimizer with learning rate 0.01 and L2 regularization weight decay
          optimizer = torch.optim.Adam(self.parameters(),lr=0.01,weight_decay=5e-4)
          # Set the model to training mode
          self.train()
          # Loop through each training epoch from 0 to epochs
          for epoch in range(epochs+1):
              # Clear gradients from previous iteration
              optimizer.zero_grad()
              # Forward pass: compute predictions for all nodes in the graph
              out = self(data.x, data.edge_index)
              # Calculate loss only on training nodes using train_mask boolean filter
              loss = criterion(out[data.train_mask],data.y[data.train_mask])
              # Compute training accuracy on training nodes
              acc = accuracy(out[data.train_mask].argmax(dim=1), data.y[data.train_mask])
              # Backward pass: compute gradients of loss with respect to model weights
              loss.backward()
              # Update model parameters
              optimizer.step()
              # Check and display progress every 20 epochs
              if(epoch % 20 == 0):
                 # Compute validation loss on validation nodes using val_mask
                 val_loss = criterion(out[data.val_mask],data.y[data.val_mask])
                 # Compute validation accuracy on validation nodes
                 val_acc = accuracy(out[data.val_mask].argmax(dim=1), data.y[data.val_mask])
                 # Print formatted training and validation loss and accuracy
                 print(f'Epoch {epoch:>3} | Train Loss:{loss:.3f} | Train Acc: {acc*100:>5.2f}% | Val Loss:{val_loss:.2f} | Val Acc: {val_acc*100:.2f}%')

# Decorator to disable gradient computation during testing/evaluation
@torch.no_grad()
# Define test function to evaluate model performance on the test split
def test(self, data):
    # Set model to evaluation mode
    self.eval()
    # Forward pass: get predictions for all nodes
    out = self(data.x, data.edge_index)
    # Compute accuracy on test nodes using test_mask boolean filter
    acc = accuracy(out.argmax(dim=1)[data.test_mask],data.y[data.test_mask])
    # Return test accuracy
    return acc

# Instantiate the GCN model with Cora features count (1433), 16 hidden units, and Cora classes (7)
gcn = GCN(dataset.num_features, 16, dataset.num_classes)

# Print the GCN model structure to console
print(gcn)

# Train the GCN model on the graph data for 100 epochs
gcn.fit(data, epochs=100)
