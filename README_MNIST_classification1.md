# 🧠 MNIST Classification using Convolutional Neural Network (CNN)

## 📌 Overview

This script trains a **Convolutional Neural Network (CNN)** to classify handwritten digits from the [MNIST dataset](http://yann.lecun.com/exdb/mnist/). The model learns to recognize digits 0–9 from 28×28 grayscale images.

---

## 🗂️ File

`MNIST_classification1.py`

---

## 📦 Libraries Used

| Library | Purpose |
|---|---|
| `numpy` | Numerical computations |
| `torch` | PyTorch deep learning framework |
| `torchvision` | Datasets and image transforms |
| `matplotlib` | Visualizing images |
| `time` | Timing (imported but not actively used) |
| `torchsummary` | Displaying model architecture summary |

---

## 🔢 Code Walkthrough — Line by Line

### 📥 Imports (Lines 3–9)

```python
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch import nn, optim
```
- `numpy` → Used for array operations.
- `torch` → Core PyTorch library for tensors and neural networks.
- `torchvision` → Provides access to MNIST dataset and image transformation utilities.
- `matplotlib.pyplot` → Used to visualize sample images.
- `time` → Imported for timing (available but unused in this script).
- `datasets, transforms` → `datasets` loads MNIST data; `transforms` preprocesses images.
- `nn, optim` → `nn` defines neural network layers; `optim` provides optimization algorithms.

```python
print(torch.__version__)
```
- Prints the installed PyTorch version for verification.

---

### 🔄 Data Preprocessing — Transforms (Lines 13–16)

```python
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])
```
- `transforms.Compose([...])` → Chains multiple transformations together.
- `transforms.ToTensor()` → Converts a PIL image (pixel values 0–255) into a PyTorch tensor with values in [0.0, 1.0].
- `transforms.Normalize((0.5,), (0.5,))` → Normalizes the tensor: subtracts mean 0.5 and divides by std 0.5. This maps pixel values from [0, 1] to [–1, 1], centering the data for better training stability.

---

### 📂 Loading Datasets (Lines 17–23)

```python
trainset = datasets.MNIST('./data', download=True, train=True, transform=transform)
testset  = datasets.MNIST('./',     download=True, train=False, transform=transform)
```
- `datasets.MNIST(...)` → Downloads and loads the MNIST dataset.
- `train=True` → Loads the 60,000 training images.
- `train=False` → Loads the 10,000 test images.
- `download=True` → Downloads the dataset if not already present locally.
- `transform=transform` → Applies the preprocessing pipeline defined above.

```python
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
testloader  = torch.utils.data.DataLoader(testset,  batch_size=64, shuffle=True)
```
- `DataLoader` → Wraps the dataset in an iterable that feeds mini-batches during training.
- `batch_size=64` → 64 images are processed at once per training step.
- `shuffle=True` → Shuffles the data every epoch to reduce overfitting.

---

### 🔍 Inspecting Data Shapes (Lines 25–38)

```python
dataiter = iter(trainloader)
images, labels = next(dataiter)
print(images.shape)   # torch.Size([64, 1, 28, 28])
print(labels.shape)   # torch.Size([64])
```
- `iter(trainloader)` → Creates an iterator over the training data loader.
- `next(dataiter)` → Fetches the first batch (64 images + 64 labels).
- `images.shape` → (batch=64, channels=1 for grayscale, height=28, width=28).
- `labels.shape` → 64 integer labels (0–9).

```python
plt.imshow(images[0].numpy().squeeze(), cmap='gray')
```
- `images[0]` → Selects the first image in the batch (shape: [1, 28, 28]).
- `.numpy()` → Converts PyTorch tensor to NumPy array.
- `.squeeze()` → Removes the channel dimension → shape becomes (28, 28).
- `cmap='gray'` → Displays the image in grayscale colormap.

---

### 🏗️ CNN Model Architecture (Lines 40–66)

```python
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
```
- Defines a class `Net` that inherits from `nn.Module` (all PyTorch models must do this).
- `super().__init__()` → Initializes the parent class, enabling all PyTorch model functionalities.

#### CNN Layers

```python
self.cnn_layers = nn.Sequential(
    nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(4),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(kernel_size=2, stride=2),

    nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(4),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(kernel_size=2, stride=2),
)
```
- `nn.Sequential(...)` → Groups layers that run one after another.
- **First Conv Block:**
  - `nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1)` → 2D convolution: takes 1 input channel (grayscale), outputs 4 feature maps, using a 3×3 filter with padding=1 (keeps spatial size: 28×28 → 28×28).
  - `nn.BatchNorm2d(4)` → Batch Normalization across 4 feature maps. Normalizes activations to stabilize and speed up training.
  - `nn.ReLU(inplace=True)` → Rectified Linear Unit activation. Sets negative values to 0. `inplace=True` saves memory by modifying the tensor directly.
  - `nn.MaxPool2d(kernel_size=2, stride=2)` → Reduces spatial dimensions by half (28×28 → 14×14) by taking the max value in each 2×2 window.
- **Second Conv Block:** Same structure as above.
  - `nn.Conv2d(4, 4, ...)` → Takes 4 feature maps, outputs 4 feature maps.
  - After this MaxPool: 14×14 → 7×7.
  - Final feature map size: (4 channels × 7 × 7 = 196 values).

#### Fully Connected Layers

```python
self.linear_layers = nn.Sequential(
    nn.Linear(4 * 7 * 7, 10)
)
```
- `nn.Linear(196, 10)` → Fully connected layer that maps 196 flattened features to 10 output classes (digits 0–9).

#### Forward Pass

```python
def forward(self, x):
    x = self.cnn_layers(x)
    x = x.view(x.size(0), -1)
    x = self.linear_layers(x)
    return x
```
- `self.cnn_layers(x)` → Passes input through all convolutional layers. Output shape: (batch, 4, 7, 7).
- `x.view(x.size(0), -1)` → Flattens each sample: (batch, 4, 7, 7) → (batch, 196). `-1` lets PyTorch infer this dimension automatically.
- `self.linear_layers(x)` → Passes through the linear layer to get 10 raw class scores (logits).

---

### ⚙️ Model Setup (Lines 67–84)

```python
model = Net()
```
- Creates an instance of the CNN.

```python
optimizer = optim.Adam(model.parameters(), lr=0.01)
```
- `optim.Adam(...)` → Adam optimizer — an adaptive gradient descent algorithm that adjusts learning rates per parameter.
- `model.parameters()` → All trainable weights and biases of the model.
- `lr=0.01` → Learning rate: how big each weight update step is.

```python
criterion = nn.CrossEntropyLoss()
```
- Loss function for multi-class classification.
- Internally combines `LogSoftmax` + `NLLLoss`, so the model's raw output (logits) is directly passed to it.

```python
if torch.cuda.is_available():
    model = model.cuda()
    criterion = criterion.cuda()
```
- Checks if a NVIDIA GPU is available.
- If yes, moves both model weights and loss function to the GPU for faster computation.

```python
print(model)
input = torch.randn(1, 1, 28, 28)
out = model(input)
print(out)
```
- `print(model)` → Displays the full model architecture.
- `torch.randn(1, 1, 28, 28)` → Creates a random test tensor (1 image, 1 channel, 28×28) to do a quick forward-pass test.
- `model(input)` → Runs the test image through the model; output shape: (1, 10).

```python
from torchsummary import summary
summary(model, input_size=(1, 28, 28))
```
- Prints a detailed layer-by-layer summary: input/output shapes and parameter counts for each layer.

---

### 🏋️ Training Loop (Lines 88–110)

```python
for i in range(10):
    running_loss = 0
```
- Loops over 10 training epochs (one epoch = one full pass through all training data).
- `running_loss` accumulates total loss for the epoch.

```python
    for images, labels in trainloader:
        if torch.cuda.is_available():
            images = images.cuda()
            labels = labels.cuda()
```
- Iterates batch-by-batch through the training data.
- Moves images and labels to GPU if available.

```python
        optimizer.zero_grad()
```
- Clears gradients from the previous step. If not done, gradients accumulate (PyTorch default behavior).

```python
        output = model(images)
```
- Forward pass: sends a batch of 64 images through the network. Output shape: (64, 10).

```python
        loss = criterion(output, labels)
```
- Computes cross-entropy loss between predicted class scores and true labels.

```python
        loss.backward()
```
- **Backpropagation**: computes gradients of the loss with respect to every model parameter using the chain rule.

```python
        optimizer.step()
```
- **Weight Update**: updates all model parameters in the direction that reduces the loss (gradient descent step).

```python
        running_loss += loss.item()
    else:
        print("Epoch {} - Training loss: {}".format(i+1, running_loss/len(trainloader)))
```
- `loss.item()` → Converts the scalar loss tensor to a Python float.
- `running_loss / len(trainloader)` → Average loss per batch for the epoch.
- `else` block after `for` → Executes when the for loop completes normally (not via `break`).

---

### 📊 Evaluation (Lines 112–133)

```python
correct_count, all_count = 0, 0
```
- Counters for correct predictions and total predictions.

```python
for images, labels in testloader:
    for i in range(len(labels)):
        if torch.cuda.is_available():
            images = images.cuda()
            labels = labels.cuda()
```
- Iterates through every test batch, then through every image inside the batch.

```python
        img = images[i].view(1, 1, 28, 28)
```
- Extracts the i-th image and reshapes it to (1, 1, 28, 28) — single sample in batch format.

```python
        with torch.no_grad():
            logps = model(img)
```
- `torch.no_grad()` → Disables gradient computation during inference (saves memory and speeds up evaluation).
- `logps` → Raw output (logits) from the model, shape: (1, 10).

```python
        ps = torch.exp(logps)
```
- Applies exponential to convert logits to unnormalized probabilities.

```python
        probab = list(ps.cpu()[0])
```
- Moves tensor to CPU and converts to a Python list of 10 probability values.

```python
        pred_label = probab.index(max(probab))
```
- Finds the index of the highest probability → the predicted digit class.

```python
        true_label = labels.cpu()[i]
```
- Gets the true label for this image, moved to CPU.

```python
        if(true_label == pred_label):
            correct_count += 1
        all_count += 1
```
- If prediction matches the ground truth, increments `correct_count`.
- Always increments `all_count`.

```python
print("Number Of Images Tested =", all_count)
print("\nModel Accuracy =", (correct_count/all_count))
```
- Prints the total number of test images and the final accuracy.

---

## 🏛️ Model Architecture Summary

```
Input: (1, 28, 28)
    ↓
Conv2d(1→4, 3×3, pad=1)  →  (4, 28, 28)
BatchNorm2d + ReLU
MaxPool2d(2×2)             →  (4, 14, 14)
    ↓
Conv2d(4→4, 3×3, pad=1)  →  (4, 14, 14)
BatchNorm2d + ReLU
MaxPool2d(2×2)             →  (4, 7, 7)
    ↓
Flatten                    →  196
    ↓
Linear(196 → 10)           →  10 class scores
```

---

## ▶️ How to Run

```bash
pip install torch torchvision torchsummary matplotlib numpy
python MNIST_classification1.py
```

---

## 📈 Expected Output

- Model architecture printed to console
- Training loss printed every epoch (10 epochs total)
- Test accuracy printed at the end (typically ~98–99%)
