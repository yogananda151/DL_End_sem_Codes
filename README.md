# Deep Learning End Semester Lab & Theory Study Guide

This guide maps each code and notebook file in this directory to the official **Deep Learning Course Syllabus** in the recommended sequence to study for the end-semester exam.

---

## 📚 Official Syllabus Overview

### **Unit 1**
- Deep Neural Networks (DNN)
- Convolutional Neural Network (CNN)
- Recurrent Neural Network (RNN): Long-Short-Term-Memory (LSTM)
- Graph based Neural Network (GNN)

### **Unit 2**
- Pre-processing: Noise Removal using deep learning algorithms
- Feature Extraction
- Signal Analysis: Time Series Analysis, CNNs, Auto encoders

### **Unit 3**
- Image Analysis: Transfer Learning, Attention models
- Ensemble Methods for Signal and Image Analysis (Bagging, Boosting, Stacking)

---

## 🧭 Recommended Study Sequence

| Study Order | File Name | Unit | Syllabus Topic | Primary Concept / Architecture |
| :---: | :--- | :---: | :--- | :--- |
| **01** | [`perceptron_2_class_epoch.ipynb`](perceptron_2_class_epoch.ipynb) | **Unit 1** | Deep Neural Networks (DNN) - Foundations | Perceptron learning rule, hard limit activation, decision boundary |
| **02** | [`MNIST_classification1.py`](MNIST_classification1.py) | **Unit 1** | Convolutional Neural Network (CNN) | PyTorch 2D CNN (Conv2d, MaxPool2d, Linear) on MNIST digit classification |
| **03** | [`RNN_Numbers_2.ipynb`](RNN_Numbers_2.ipynb) | **Unit 1 & 2** | Recurrent Neural Network (RNN) & Time Series | Sequential numbers regression, hidden states, time series forecasting |
| **04** | [`rnn_text2.py`](rnn_text2.py) | **Unit 1** | RNN, GRU, and Long-Short-Term-Memory (LSTM) | Word embeddings, Vanilla RNN vs GRU vs LSTM text generation |
| **05** | [`gnn2.py`](gnn2.py) | **Unit 1** | Graph based Neural Network (GNN) | PyTorch Geometric GCNConv, node degree analysis, Cora citation graph |
| **06** | [`UNet.ipynb`](UNet.ipynb) | **Unit 2** | Noise Removal / Autoencoders / Segmentation | U-Net Encoder-Decoder, skip connections, UpConvBlock, Dice Loss |
| **07** | [`Glaucoma_classification_1.ipynb`](Glaucoma_classification_1.ipynb) | **Unit 2 & 3** | Feature Extraction, CNNs & Image Analysis | Custom multi-layer CNN, spatial dimension tracking, LR scheduler |
| **08** | [`bagging_student.ipynb`](bagging_student.ipynb) | **Unit 3** | Transfer Learning & Ensemble: Bagging | Pretrained ResNet18, ResNet34, DenseNet121 + Bootstrap Aggregating |
| **09** | [`Boosting_student_new.ipynb`](Boosting_student_new.ipynb) | **Unit 3** | Transfer Learning & Ensemble: Boosting | Pretrained ResNet18 + AdaBoost sample weight updates & weighted voting |
| **10** | [`stacking_2 - Copy.ipynb`](stacking_2%20-%20Copy.ipynb) | **Unit 3** | Transfer Learning & Ensemble: Stacking | Out-Of-Fold (OOF) cross-validation, CNN base learners + Logistic Regression meta-classifier |

---

## 📝 Detailed File-by-File Breakdown

### 1. `perceptron_2_class_epoch.ipynb`
* **Syllabus Topic:** Unit 1 — Deep Neural Networks (DNN) Foundations
* **Description:** Implements a single-layer perceptron from scratch in NumPy for binary classification.
* **Key Concepts to Study:**
  - Hard limit (Heaviside step) activation function.
  - Perceptron weight and bias update rule: $w \leftarrow w + \eta \cdot (y - \hat{y}) \cdot x$.
  - Tracking classification error across epochs.
  - Plotting the linear decision boundary separating two classes.

---

### 2. `MNIST_classification1.py`
* **Syllabus Topic:** Unit 1 — Convolutional Neural Network (CNN)
* **Description:** A complete PyTorch script demonstrating CNN architecture design, training, and evaluation on the MNIST handwritten digits dataset.
* **Key Concepts to Study:**
  - `torchvision.datasets.MNIST` and `DataLoader` with normalization transforms.
  - Convolution block: `nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)`.
  - Batch normalization (`nn.BatchNorm2d`) and spatial reduction with `nn.MaxPool2d(2, 2)`.
  - Calculating flattened feature map dimensions (`4 * 7 * 7 = 196`) into `nn.Linear`.
  - Loss function (`nn.CrossEntropyLoss`) and optimization with `optim.Adam`.

---

### 3. `RNN_Numbers_2.ipynb`
* **Syllabus Topic:** Unit 1 & Unit 2 — Recurrent Neural Network (RNN) & Time Series Analysis
* **Description:** Builds a recurrent network to model sequential numerical patterns and predict the next continuous number in a series.
* **Key Concepts to Study:**
  - Constructing sequential input tensors with a sliding window (`sequence_length = 10`, 2 input features).
  - Recurrent layer instantiation: `torch.nn.RNN(input_size=2, hidden_size=64, batch_first=True)`.
  - Initializing hidden state tensor $h_0$.
  - Extracting the last time step output `out[:, -1, :]` for linear regression with `nn.MSELoss`.

---

### 4. `rnn_text2.py`
* **Syllabus Topic:** Unit 1 — RNN, GRU, and Long-Short-Term-Memory (LSTM)
* **Description:** Comprehensive NLP pipeline comparing Vanilla RNN, Gated Recurrent Unit (GRU), and LSTM architectures trained on `Alice1.txt.txt`.
* **Key Concepts to Study:**
  - Text preprocessing: punctuation stripping, word tokenization with NLTK, building `word_to_int` and `int_to_word` vocabularies.
  - Word embedding layer: `nn.Embedding(vocab_size, embedding_dim)`.
  - Architectural differences between:
    - **Vanilla RNN:** Single hidden state vector $h_t$.
    - **GRU:** Hidden state with reset and update gates.
    - **LSTM:** Dual state consisting of hidden state $h_t$ and cell state $c_t$ (handles vanishing gradients).
  - Autoregressive text generation loop by predicting the next token and sliding the sequence window.

---

### 5. `gnn2.py`
* **Syllabus Topic:** Unit 1 — Graph based Neural Network (GNN)
* **Description:** Demonstrates Graph Neural Networks on graph-structured data using PyTorch Geometric (`torch_geometric`).
* **Key Concepts to Study:**
  - Loading the citation benchmark dataset `Planetoid(root=".", name="Cora")`.
  - Graph structural properties: Node degrees using `torch_geometric.utils.degree` and degree distribution plotting.
  - Graph Convolutional layer: `GCNConv(dim_in, dim_h)`.
  - Message passing forward pass: `h = gcn(x, edge_index)` combining node features and adjacency graph connectivity.
  - Mask-based semi-supervised training (`train_mask`, `val_mask`, `test_mask`).

---

### 6. `UNet.ipynb`
* **Syllabus Topic:** Unit 2 — Noise Removal, Autoencoders, and Pre-processing
* **Description:** Implements the iconic U-Net Encoder-Decoder convolutional network with skip connections for biomedical image analysis and segmentation.
* **Key Concepts to Study:**
  - **Encoder (Contracting path):** Consecutive `ConvBlock` modules with downsampling to extract latent feature representations.
  - **Bottleneck:** Lowest resolution latent space capturing high-level context.
  - **Decoder (Expansive path):** `UpConvBlock` with `nn.Upsample` to restore spatial resolution.
  - **Skip Connections:** Concatenating encoder feature maps to decoder layers to preserve high-resolution spatial boundaries.
  - **Evaluation Metric & Loss:** Dice Coefficient and `dice_loss = 1.0 - dice`.

---

### 7. `Glaucoma_classification_1.ipynb`
* **Syllabus Topic:** Unit 2 & Unit 3 — Feature Extraction, CNNs & Image Analysis
* **Description:** Medical retinal image classification pipeline with a deep custom CNN architecture and robust training dynamics.
* **Key Concepts to Study:**
  - Custom dataset class `AMD_dataset` with custom image transformations.
  - Mathematical formula function `findConv2dOutShape(H_in, W_in, conv, pool)` to track layer dimensions dynamically.
  - Preventing overfitting: `nn.Dropout(p=0.25)` and `ReduceLROnPlateau` adaptive learning rate scheduler.
  - Checkpointing best model weights based on minimum validation loss (`best_model_wts`).

---

### 8. `bagging_student.ipynb`
* **Syllabus Topic:** Unit 3 — Transfer Learning & Ensemble Methods: Bagging
* **Description:** Trains an ensemble of heterogeneous pretrained deep convolutional networks using Bootstrap Aggregating (Bagging) on glaucoma retinal images.
* **Key Concepts to Study:**
  - Transfer learning with pretrained backbones:
    - `models.resnet18(weights=ResNet18_Weights.DEFAULT)`
    - `models.resnet34(weights=ResNet34_Weights.DEFAULT)`
    - `models.densenet121(weights=DenseNet121_Weights.DEFAULT)`
  - Modifying the classification head: `model.fc = nn.Linear(num_features, 2)`.
  - Bagging concept: Training independent models on bootstrap samples or varied architectures.
  - Soft voting / probability averaging: $\bar{P} = \frac{1}{M}\sum_{m=1}^{M} P_m$.
  - Evaluation metrics: Accuracy, Sensitivity (Recall), Specificity, Precision, F1-Score, and ROC-AUC.

---

### 9. `Boosting_student_new.ipynb`
* **Syllabus Topic:** Unit 3 — Transfer Learning & Ensemble Methods: Boosting
* **Description:** Implements sequential AdaBoost-style boosting with deep ResNet18 learners for binary medical image classification.
* **Key Concepts to Study:**
  - Sequential ensemble mechanism: Learners are trained iteratively where subsequent models focus on hard/misclassified examples.
  - Sample weight initialization: $w_i = \frac{1}{N}$.
  - Custom `WeightedGlaucomaDataset` with weighted sampling.
  - Calculating model importance weight $\alpha_m = \frac{1}{2} \ln\left(\frac{1 - \epsilon_m}{\epsilon_m}\right)$.
  - Updating sample weights: $w_i \leftarrow w_i \cdot \exp(\alpha_m)$ for misclassified samples.
  - Weighted voting aggregation for final predictions: $\hat{Y} = \text{sign}\left(\sum \alpha_m h_m(x)\right)$.

---

### 10. `stacking_2 - Copy.ipynb`
* **Syllabus Topic:** Unit 3 — Transfer Learning & Ensemble Methods: Stacking
* **Description:** Implements a meta-learning Stacking ensemble combining ResNet18, ResNet34, and DenseNet121 using out-of-fold (OOF) cross-validation and Logistic Regression.
* **Key Concepts to Study:**
  - **Level-0 Base Learners:** Pretrained ResNet18, ResNet34, and DenseNet121.
  - **Out-Of-Fold (OOF) Strategy:** Using `StratifiedKFold` to generate unbiased validation predictions across the dataset without data leakage.
  - **Meta-Features Matrix:** Stacking individual predictions:
    $$\mathbf{Z} = [\hat{y}_{\text{ResNet18}}, \hat{y}_{\text{ResNet34}}, \hat{y}_{\text{DenseNet121}}]$$
  - **Level-1 Meta-Learner:** Training a `LogisticRegression` classifier on $\mathbf{Z}$ to produce final ensemble decisions.
  - Comprehensive performance comparison table: Individual CNN baselines vs Final Stacking Ensemble.

---

## 📦 Data Files Reference

| File | Purpose | Used In |
| :--- | :--- | :--- |
| `Alice1.txt.txt` | Text corpus (Alice in Wonderland excerpt) for language modeling | [`rnn_text2.py`](rnn_text2.py) |
| `drion_training.xlsx` | Training labels & metadata for DRION retinal glaucoma dataset | [`stacking_2 - Copy.ipynb`](stacking_2%20-%20Copy.ipynb), [`bagging_student.ipynb`](bagging_student.ipynb), [`Boosting_student_new.ipynb`](Boosting_student_new.ipynb) |
| `drion_testing.xlsx` | Testing labels & metadata for DRION retinal glaucoma dataset | Ensemble notebooks |
| `training_drion.zip` | Compressed archive of retinal fundus images | Glaucoma / DRION classification tasks |

---

## 🎯 Quick Exam Revision Checklist

1. **Unit 1: Architectures & Mechanics**
   - [ ] Explain the difference between FC layers, Conv2D, RNN, LSTM, and GNN.
   - [ ] Draw and explain the gating mechanism of LSTM (Forget, Input, Output gates) vs GRU (Reset, Update gates).
   - [ ] Understand why GNNs need `edge_index` and how message passing updates node embeddings.

2. **Unit 2: Preprocessing & Autoencoders**
   - [ ] Understand the U-Net architecture: contracting path, bottleneck, expansive path, and skip connections.
   - [ ] Contrast standard autoencoders with U-Net (skip connections preserve fine spatial details).
   - [ ] Know how to compute output feature map dimensions after convolution and pooling.

3. **Unit 3: Transfer Learning & Ensembles**
   - [ ] Explain Transfer Learning (fine-tuning vs feature extraction, replacing the final classification layer).
   - [ ] Compare the 3 Ensemble techniques:
     - **Bagging:** Parallel, independent models, reduces variance (e.g., averaging probabilities).
     - **Boosting:** Sequential, focus on misclassified samples, reduces bias (e.g., AdaBoost weights).
     - **Stacking:** Heterogeneous models, meta-learner learns optimal weighting of base predictions.
