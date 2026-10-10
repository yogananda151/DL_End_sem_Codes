# 📖 Text Generation with RNN, GRU, and LSTM

## 📌 Overview

This script trains three different sequence models — **RNN**, **GRU**, and **LSTM** — to generate text character-by-character (word-by-word) based on Lewis Carroll's *"Alice in Wonderland"*. Each model learns to predict the next word in a sequence and can then generate new text from a starting phrase.

---

## 🗂️ File

`rnn_text2.py`

---

## 📦 Libraries Used

| Library | Purpose |
|---|---|
| `os` | File path checks and reading |
| `torch`, `torch.nn`, `torch.optim` | PyTorch deep learning |
| `numpy` | Array operations |
| `torch.utils.data` | Dataset and DataLoader for batching |
| `string` | Punctuation filtering |
| `nltk.tokenize` | Tokenizing text into words |

---

## 📂 Script Structure

The script has **4 major sections**:

1. **Data Loading & Preprocessing** (Lines 1–71)
2. **RNN Model** — Training & Text Generation (Lines 73–176)
3. **GRU Model** — Training & Text Generation (Lines 178–281)
4. **LSTM Model** — Training & Text Generation (Lines 283–388)

---

## 🔢 Code Walkthrough — Line by Line

---

## SECTION 1: Data Loading & Preprocessing

### 📥 Imports (Lines 8–15)

```python
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader, TensorDataset
import string
```
- `os` → For checking if the file exists and reading it.
- `torch` / `nn` / `optim` → PyTorch model building, layers, and optimizers.
- `numpy` → Array manipulation.
- `Dataset, DataLoader, TensorDataset` → Tools for batching data during training.
- `string` → Provides `string.punctuation` — a collection of all punctuation characters.

---

### 📄 Reading the Text File (Lines 19–34)

```python
input_file = r"f:/Amrita_2026/22AIE304(Deep Learning)/Codes/Endsem_codes/Alice1.txt.txt"
```
- Defines the path to the Alice in Wonderland text file.

```python
if not os.path.exists(input_file):
    print("ERROR: File does not exist!")
else:
    print("File exists.")
    with open(input_file, "r", encoding="utf-8") as infile:
        text_data = infile.read()
    print("Number of characters:", len(text_data))
    print("First 500 characters:")
    print(text_data[:500])
```
- `os.path.exists(input_file)` → Checks if the file path is valid.
- `open(..., encoding="utf-8")` → Opens the file in read mode using UTF-8 encoding.
- `infile.read()` → Reads the entire file content into a single string.
- Prints the character count and first 500 characters as a preview.

---

### 🧹 Text Cleaning (Lines 36–49)

```python
punctuation = string.punctuation.replace('.', '').replace('?', '')
```
- `string.punctuation` → All punctuation: `!"#$%&'()*+,-./:;<=>?@[\]^_...`
- `.replace('.', '')` → Keeps periods (`.`) in the text (sentence endings).
- `.replace('?', '')` → Keeps question marks (`?`) in the text.
- Result: all other punctuation characters will be removed.

```python
text_data = ""
if os.path.exists(input_file):
    with open(input_file, 'r') as infile:
        for line in infile:
            if line.strip():
                for char in punctuation:
                    line = line.replace(char, '')
                text_data += line
```
- Re-reads the file line by line (no UTF-8 this time).
- `line.strip()` → Skips empty lines.
- Inner loop removes all unwanted punctuation characters from each line.
- Cleaned lines are concatenated into `text_data`.

```python
print("Processed Text:")
print(text_data)
```
- Displays the cleaned text.

---

### 🔤 Tokenization & Vocabulary (Lines 51–59)

```python
from nltk.tokenize import TreebankWordTokenizer
tokenizer = TreebankWordTokenizer()
words = tokenizer.tokenize(text_data.lower())
```
- `TreebankWordTokenizer` → NLTK tokenizer using Penn Treebank rules. Splits text into words intelligently (handles contractions, punctuation properly).
- `text_data.lower()` → Converts all text to lowercase for consistency.
- `words` → A list of all word tokens in the text.

```python
word_to_int = {w: i for i, w in enumerate(set(words))}
int_to_word = {i: w for w, i in word_to_int.items()}
len(word_to_int)
```
- `set(words)` → Unique vocabulary (set removes duplicates).
- `word_to_int` → Dictionary mapping each unique word → unique integer index. Example: `{'alice': 0, 'was': 1, ...}`
- `int_to_word` → Reverse mapping: integer index → word. Used to decode predictions.
- `len(word_to_int)` → Total vocabulary size (number of unique words).

---

### 🔗 Creating Sequences (Lines 61–71)

```python
def create_sequences(input_data, seq_length):
    sequences = []
    for i in range(0, len(input_data) - seq_length):
        sequence_in  = input_data[i : i + seq_length]
        sequence_out = input_data[i + seq_length]
        sequences.append((sequence_in, sequence_out))
    return sequences
```
- Creates training pairs for next-word prediction using a **sliding window** approach.
- `seq_length` → Number of input words in each sequence (context window).
- `sequence_in` → A window of `seq_length` consecutive encoded words (the input).
- `sequence_out` → The very next word after the window (the target to predict).
- Example: if words are `[1, 4, 7, 2, 5, 9]` and `seq_length=4`:
  - Input: `[1, 4, 7, 2]`, Output: `5`
  - Input: `[4, 7, 2, 5]`, Output: `9`

```python
encoded_text = np.array([word_to_int[word] for word in words])
```
- Converts every word in the text to its integer index using `word_to_int`.
- Result: a NumPy array of integers representing the entire book.

```python
seq_length = 5
sequences = create_sequences(encoded_text, seq_length)
```
- Uses a context window of 5 words to predict the 6th word.
- `sequences` → List of (input_sequence, target_word) tuples.

---

---

## SECTION 2: RNN Model (Lines 73–176)

### 🏗️ RNN Model Architecture (Lines 80–100)

```python
class RNNModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(RNNModel, self).__init__()
```
- Defines the RNN-based language model.
- `vocab_size` → Number of unique words (vocabulary size).
- `embedding_dim` → Dimension of word embedding vectors (256).
- `hidden_dim` → Size of the hidden state (512).
- `output_dim` → Number of output classes (same as vocab_size for next-word prediction).

```python
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
```
- Lookup table that converts each word index into a dense 256-dimensional vector.
- Learns meaningful word representations during training.

```python
        self.rnn1 = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.rnn2 = nn.RNN(hidden_dim, hidden_dim, batch_first=True)
```
- Two stacked RNN layers.
- `batch_first=True` → Expects input as (batch, sequence_length, features).
- **RNN1**: Takes embedded words (256-dim) → produces hidden states (512-dim).
- **RNN2**: Takes RNN1 outputs (512-dim) → produces hidden states (512-dim).
- Stacking helps the model learn more complex patterns.

```python
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.hidden_dim = hidden_dim
```
- `nn.Linear(512, vocab_size)` → Maps the final hidden state to vocabulary scores.
- `hidden_dim` stored for use in `init_hidden`.

#### Forward Pass

```python
    def forward(self, x, hidden1, hidden2):
        embedded = self.embedding(x)
```
- `x` → Batch of integer sequences, shape: (batch, seq_length).
- `self.embedding(x)` → Converts each word index to its embedding: shape (batch, seq_length, 256).

```python
        out1, hidden1 = self.rnn1(embedded, hidden1)
        out2, hidden2 = self.rnn2(out1, hidden2)
```
- `out1` → All hidden states from RNN1: shape (batch, seq_length, 512).
- `hidden1` → Final hidden state of RNN1 (passed to the next batch for continuity).
- `out2`, `hidden2` → Same for RNN2.

```python
        out = self.fc(out2[:, -1, :])
        return out, hidden1, hidden2
```
- `out2[:, -1, :]` → Takes only the **last time step's** hidden state from RNN2: shape (batch, 512).
- This represents the model's understanding after reading all 5 input words.
- `self.fc(...)` → Projects to vocabulary size: shape (batch, vocab_size).
- Returns predictions and updated hidden states.

#### Hidden State Initialization

```python
    def init_hidden(self, batch_size, device):
        return (
            torch.zeros(1, batch_size, self.hidden_dim).to(device),
            torch.zeros(1, batch_size, self.hidden_dim).to(device),
        )
```
- Initializes hidden states to zeros at the start of each sequence batch.
- Shape: (num_layers=1, batch_size, hidden_dim=512).
- `.to(device)` → Moves tensors to GPU if available.

---

### ⚙️ RNN Hyperparameters & Setup (Lines 102–121)

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```
- Automatically selects GPU if available, else uses CPU.

```python
input_size   = len(word_to_int)   # vocab size
output_size  = len(word_to_int)   # same — predicting a word from vocabulary
embedding_dim = 256
hidden_dim    = 512
batch_size    = 32
epochs        = 100
```
- All hyperparameters defined before model creation.

```python
model = RNNModel(input_size, embedding_dim, hidden_dim, output_size).to(device)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
```
- Creates the model and moves it to the target device.
- `CrossEntropyLoss` → Measures how far the predicted word distribution is from the true next word.
- `Adam` optimizer with a small learning rate of 0.0001.

```python
inputs  = np.array([item[0] for item in sequences])
targets = np.array([item[1] for item in sequences])
```
- Extracts all input sequences and their corresponding target words into NumPy arrays.

```python
inputs_tensor  = torch.tensor(inputs,  dtype=torch.long)
targets_tensor = torch.tensor(targets, dtype=torch.long)
```
- Converts to PyTorch Long tensors (required for embedding lookups and loss computation).

```python
dataset     = TensorDataset(inputs_tensor, targets_tensor)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
```
- Wraps tensors into a Dataset and creates a DataLoader for mini-batch training.
- `shuffle=True` → Randomizes batch order each epoch to improve training.

---

### 🏋️ RNN Training Loop (Lines 123–138)

```python
for epoch in range(epochs):
    epoch_loss = 0
    for batch_idx, (inputs, targets) in enumerate(data_loader):
        inputs, targets = inputs.to(device), targets.to(device)
```
- Outer loop: iterates over 100 epochs.
- Inner loop: iterates over mini-batches.
- Moves each batch to GPU/CPU.

```python
        hidden1, hidden2 = model.init_hidden(inputs.size(0), device)
```
- Reinitializes hidden states for each new batch (fresh start per batch).
- `inputs.size(0)` → Current batch size (may differ from `batch_size` for the last batch).

```python
        optimizer.zero_grad()
        output, hidden1, hidden2 = model(inputs, hidden1, hidden2)
        loss = loss_function(output, targets)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
```
- Standard training step: zero grads → forward → loss → backprop → update.

```python
    print(f'Epoch {epoch + 1}, Loss: {epoch_loss / len(data_loader)}')
```
- Prints average loss per batch for the epoch.

```python
torch.save(model, 'RNN_61.pth')
```
- Saves the **entire model** (architecture + weights) to a file.

---

### 🔮 RNN Text Generation (Lines 142–176)

```python
model = torch.load('RNN_61.pth', map_location=torch.device('cpu'), weights_only=False)
model.eval()
```
- Loads the saved model from disk.
- `map_location='cpu'` → Loads to CPU even if saved on GPU.
- `model.eval()` → Switches to evaluation mode (disables dropout, etc.).

```python
def words_to_tensor(words, word_to_int):
    indices = [word_to_int[word] for word in words]
    tensor  = torch.tensor(np.array([indices]), dtype=torch.long)
    return tensor
```
- Converts a list of word strings to a Long tensor of their integer indices.
- Shape: (1, len(words)) — a single sequence.

```python
def generate_sequence(start_words, model, word_to_int, int_to_word, seq_length=5, length=2):
    start_words = start_words.replace('\n', ' ')
    words = tokenizer.tokenize(start_words.lower())
    input_tensor = words_to_tensor(words[-seq_length:], word_to_int)
```
- `start_words` → Initial seed phrase for text generation.
- Newlines replaced with spaces for clean tokenization.
- Tokenizes and lowercases the seed phrase.
- `words[-seq_length:]` → Takes the last 5 words (matches training context size).

```python
    hidden1, hidden2 = model.init_hidden(batch_size=1, device=torch.device('cpu'))
```
- Initializes hidden states for a single sequence (batch_size=1).

```python
    for _ in range(length):
        with torch.no_grad():
            output, hidden1, hidden2 = model(input_tensor, hidden1, hidden2)
        predicted_index = torch.argmax(output, dim=1).item()
        predicted_word  = int_to_word[predicted_index]
        words.append(predicted_word)
        input_tensor = words_to_tensor(words[-seq_length:], word_to_int)
```
- Generates `length` new words autoregressively:
  - Forward pass → get output scores → find highest score index → decode to word.
  - Appends predicted word to the word list.
  - Shifts the input window forward by 1 (sliding window).
- `torch.no_grad()` → Saves memory; no gradient needed for inference.

```python
    return ' '.join(words)
```
- Joins all words (original seed + generated words) into a readable sentence.

```python
start_words = "Alice was beginning to get very tired of "
predicted_sequence = generate_sequence(start_words, model, word_to_int, int_to_word)
print(f"Generated sequence: {predicted_sequence}")
```
- Runs text generation from the seed phrase and prints the result.

---

## SECTION 3: GRU Model (Lines 178–281)

The GRU section mirrors the RNN section but uses **Gated Recurrent Units** instead.

### Key Differences from RNN

```python
class GRUModel(nn.Module):
    ...
    self.gru1 = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
    self.gru2 = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
```
- **`nn.GRU`** replaces `nn.RNN`.
- GRU has **Update Gate** and **Reset Gate** which control how much of the previous hidden state to keep vs. reset.
- This helps GRU better handle **long-term dependencies** compared to vanilla RNN.
- Hidden state structure is the same: shape (1, batch, hidden_dim).

Everything else (forward pass, training loop, generation) is identical to the RNN section.

The trained GRU model is saved as:
```python
torch.save(model, 'my_gru_entire_model.pth')
```

Generation seed:
```python
start_words = "There was nothing so VERY remarkable"
```

---

## SECTION 4: LSTM Model (Lines 283–388)

The LSTM section mirrors the RNN/GRU sections but uses **Long Short-Term Memory** cells.

### Key Differences from RNN/GRU

```python
class LSTMModel(nn.Module):
    ...
    self.lstm1 = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
    self.lstm2 = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
```
- **`nn.LSTM`** replaces `nn.RNN`/`nn.GRU`.
- LSTM has **two hidden states**: `h` (hidden state) and `c` (cell state).
- The cell state acts like a memory conveyor belt, preserving long-term information more effectively.

```python
    def init_hidden(self, batch_size, device):
        return (
            (torch.zeros(1, batch_size, self.hidden_dim).to(device),   # h1, c1 for LSTM1
             torch.zeros(1, batch_size, self.hidden_dim).to(device)),
            (torch.zeros(1, batch_size, self.hidden_dim).to(device),   # h2, c2 for LSTM2
             torch.zeros(1, batch_size, self.hidden_dim).to(device))
        )
```
- Returns **two tuples** (one per LSTM layer), each containing `(h, c)`.
- Each LSTM layer needs both a hidden state `h` AND a cell state `c`.

The trained LSTM model is saved as:
```python
torch.save(model, 'my_lstm_entire_model.pth')
```

Generation seed:
```python
start_words = "Alice was beginning to get very tired of"
```

---

## 🏛️ All Model Architectures

```
Input Words (5 words)
      ↓
Embedding (vocab_size → 256)
      ↓
Layer 1: RNN / GRU / LSTM  (256 → 512)
      ↓
Layer 2: RNN / GRU / LSTM  (512 → 512)
      ↓
Take Last Timestep Output  (batch, 512)
      ↓
Linear (512 → vocab_size)
      ↓
Predicted Next Word Index
```

---

## 🔑 RNN vs. GRU vs. LSTM Comparison

| Feature | RNN | GRU | LSTM |
|---|---|---|---|
| Hidden States | 1 (h) | 1 (h) | 2 (h + c) |
| Gates | None | Update + Reset | Input + Forget + Output |
| Long-term memory | Poor | Good | Best |
| Parameters | Fewest | Medium | Most |
| Training Speed | Fastest | Medium | Slowest |

---

## ▶️ How to Run

```bash
pip install torch numpy nltk
python -c "import nltk; nltk.download('punkt')"
python rnn_text2.py
```

> ⚠️ Update `input_file` path on Line 19 to point to your local copy of `Alice1.txt.txt`.

---

## 📈 Expected Output

```
Processed Text: [cleaned alice text...]
['alice', 'was', 'beginning', ...]
Epoch 1, Loss: 7.432
Epoch 2, Loss: 6.891
...
Epoch 100, Loss: 2.341
Generated sequence: alice was beginning to get very tired of sitting
...
```
