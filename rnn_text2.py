# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 15:54:35 2025

@author: mithun
"""

# Import operating system module for file paths and filesystem checks
import os
# Import core PyTorch library for tensor manipulation and neural networks
import torch
# Import neural network building blocks (RNN, GRU, LSTM, Linear, Embedding)
import torch.nn as nn
# Import optimizers (such as Adam) to update network weights
import torch.optim as optim
# Import NumPy library for array processing and indexing
import numpy as np
# Import dataset utilities to package data and batch them into loaders
from torch.utils.data import Dataset, DataLoader,TensorDataset
# Redundant import of os kept intact without modifying code
import os
# Import string module containing ASCII punctuation constants
import string

# Redundant import of os kept intact without modifying code
import os

# Define the absolute file path to the text file (Alice in Wonderland)
input_file = r"f:/Amrita_2026/22AIE304(Deep Learning)/Codes/Endsem_codes/Alice1.txt.txt"

# Print status message indicating file path check
print("Checking file:")
# Print the path being verified
print(input_file)

# Check if the text file exists at the specified path
if not os.path.exists(input_file):
    # Print error message if the file is not found
    print("ERROR: File does not exist!")
else:
    # Print success message if the file is found
    print("File exists.")

    # Open the text file in read mode with UTF-8 encoding
    with open(input_file, "r", encoding="utf-8") as infile:
        # Read the entire text content into memory as a string
        text_data = infile.read()

    # Print the total count of characters in the file
    print("Number of characters:", len(text_data))
    # Print label for previewing the first 500 characters
    print("First 500 characters:")
    # Display the first 500 characters of the raw text
    print(text_data[:500])

# Define the input file path again
input_file = "f:/Amrita_2026/22AIE304(Deep Learning)/Codes/Endsem_codes/Alice1.txt.txt"
# Get punctuation characters while preserving '.' and '?' to keep sentence boundaries
punctuation = string.punctuation.replace('.', '').replace('?', '')

# Initialize an empty string to accumulate cleaned text
text_data=""
# Check if input file exists before opening
if os.path.exists(input_file) :
    # Open the file for reading
    with open(input_file, 'r') as infile:
        # Loop over each line in the file
        for line in infile:
            # Check if line is not empty (ignoring whitespace)
            if line.strip():
                # Loop through each unwanted punctuation character
                for char in punctuation:
                    # Remove the punctuation character by replacing it with empty string
                    line = line.replace(char, '')
                # Append cleaned line to text_data
                text_data += line

# Print header label for the cleaned text
print("Processed Text:")
# Print the full processed text
print(text_data)

# Import TreebankWordTokenizer from NLTK for splitting text into words
from nltk.tokenize import TreebankWordTokenizer

# Instantiate word tokenizer
tokenizer = TreebankWordTokenizer()
# Convert text to lowercase and tokenize into a list of word tokens
words = tokenizer.tokenize(text_data.lower())
# Print tokenized words list
print(words)

# Create vocabulary mapping: map each unique word to a unique integer index
word_to_int = {w: i for i, w in enumerate(set(words))}
# Create reverse mapping: map integer index back to its corresponding word
int_to_word = {i: w for w, i in word_to_int.items()}
# Show the vocabulary size (number of unique words)
len(word_to_int)

# Define function to construct sliding window input-output sequence pairs
def create_sequences(input_data, seq_length):
    # Initialize empty list to store sequence pairs
    sequences = []
    # Loop over indices where a full sequence window plus target fits
    for i in range(0, len(input_data) - seq_length):
        # Slice 'seq_length' consecutive words as input sequence
        sequence_in = input_data[i:i + seq_length]
        # Next immediate word is the target word to predict
        sequence_out = input_data[i + seq_length]
        # Append tuple (input_sequence, target_word)
        sequences.append((sequence_in, sequence_out))
    # Return all generated sequence pairs
    return sequences

# Encode word tokens into integer indices using word_to_int mapping
encoded_text = np.array([word_to_int[word] for word in words])
# Set sequence length (number of context words used for prediction) to 5
seq_length = 5
# Create dataset of 5-word sequences and next-word targets
sequences = create_sequences(encoded_text, seq_length)

########################################### RNN ####################################################
# Re-import core PyTorch library
import torch
# Re-import neural network module
import torch.nn as nn
# Re-import optimizers
import torch.optim as optim
# Re-import DataLoader and TensorDataset utilities
from torch.utils.data import DataLoader, TensorDataset
# Re-import numpy
import numpy as np

# Define Recurrent Neural Network (RNN) architecture for word generation
class RNNModel(nn.Module):
    # Initialize model layers: embedding, 2 RNN layers, linear layer
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        # Call parent constructor
        super(RNNModel, self).__init__()
        # Embedding layer to convert word index integers into dense vectors of size embedding_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # 1st RNN layer accepting embeddings and outputting hidden representations
        self.rnn1 = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        # 2nd RNN layer stacked on top of the first RNN layer
        self.rnn2 = nn.RNN(hidden_dim, hidden_dim, batch_first=True)
        # Fully connected layer projecting hidden state to vocabulary size for next-word prediction
        self.fc = nn.Linear(hidden_dim, output_dim)
        # Store hidden dimension size
        self.hidden_dim = hidden_dim

    # Define forward pass with inputs x and two hidden state tensors
    def forward(self, x, hidden1, hidden2):
        # Pass input word indices through embedding layer: (batch, seq_len) -> (batch, seq_len, embed_dim)
        embedded = self.embedding(x)
        # Pass embeddings through 1st RNN layer
        out1, hidden1 = self.rnn1(embedded, hidden1)
        # Pass 1st RNN output through 2nd RNN layer
        out2, hidden2 = self.rnn2(out1, hidden2)
        # Take hidden state of last time step (-1) and pass through linear layer
        out = self.fc(out2[:, -1, :])
        # Return output logits and updated hidden states
        return out, hidden1, hidden2

    # Helper method to initialize hidden states with zeros
    def init_hidden(self, batch_size, device):
        # Return a tuple of two zero tensors for hidden states of rnn1 and rnn2
        return (
            torch.zeros(1, batch_size, self.hidden_dim).to(device),
            torch.zeros(1, batch_size, self.hidden_dim).to(device),
        )

# Select GPU if available, otherwise fallback to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Input vocabulary size
input_size = len(word_to_int)
# Output vocabulary size (predicting next word from same vocabulary)
output_size = len(word_to_int)
# Dimensionality of embedding vector for each word
embedding_dim = 256
# Number of hidden units in each RNN layer
hidden_dim = 512
# Number of samples per training batch
batch_size = 32
# Total training epochs
epochs = 100

# Instantiate RNNModel and transfer its weights to the selected device (GPU/CPU)
model = RNNModel(input_size, embedding_dim, hidden_dim, output_size).to(device)

# Define CrossEntropyLoss function for word classification
loss_function = nn.CrossEntropyLoss()
# Define Adam optimizer with learning rate 0.0001
optimizer = optim.Adam(model.parameters(), lr=0.0001)
# Extract all input sequences into a numpy array
inputs = np.array([item[0] for item in sequences])
# Extract all target words into a numpy array
targets = np.array([item[1] for item in sequences])

# Convert input sequences into a PyTorch LongTensor
inputs_tensor = torch.tensor(inputs, dtype=torch.long)
# Convert target words into a PyTorch LongTensor
targets_tensor = torch.tensor(targets, dtype=torch.long)

# Wrap tensors into a TensorDataset
dataset = TensorDataset(inputs_tensor, targets_tensor)
# Create DataLoader with batch size 32 and shuffling enabled
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Loop over training epochs
for epoch in range(epochs):
    # Initialize epoch loss accumulator
    epoch_loss = 0

    # Iterate over each mini-batch in DataLoader
    for batch_idx, (inputs, targets) in enumerate(data_loader):
        # Move inputs and targets to GPU/CPU
        inputs, targets = inputs.to(device), targets.to(device)
        # Initialize zero hidden states for current batch
        hidden1, hidden2 = model.init_hidden(inputs.size(0), device)

        # Zero out gradients from previous batch
        optimizer.zero_grad()
        # Forward pass: get model prediction logits and updated hidden states
        output, hidden1, hidden2 = model(inputs, hidden1, hidden2)
        # Calculate loss between predictions and actual targets
        loss = loss_function(output, targets)
        # Backward pass: compute parameter gradients
        loss.backward()
        # Update model parameters
        optimizer.step()

        # Accumulate scalar batch loss
        epoch_loss += loss.item()

    # Print epoch number and average training loss
    print(f'Epoch {epoch + 1}, Loss: {epoch_loss / len(data_loader)}')

# Save the trained model checkpoint to disk
torch.save(model, 'RNN_61.pth')

# Re-import torch
import torch
# Re-import numpy
import numpy as np

# Load the saved RNN model checkpoint onto CPU with weights_only=False
model = torch.load('RNN_61.pth', map_location=torch.device('cpu'),weights_only=False)
# Set model to evaluation mode for inference
model.eval()

# Helper function to convert a list of words into a PyTorch LongTensor
def words_to_tensor(words, word_to_int):
    # Map each word to its integer index
    indices = [word_to_int[word] for word in words]
    # Wrap index list into a 2D tensor of shape (1, len(words))
    tensor = torch.tensor(np.array([indices]), dtype=torch.long)
    # Return the tensor
    return tensor

# Function to generate text continuation using the trained model
def generate_sequence(start_words, model, word_to_int, int_to_word, seq_length=5, length=2):
    # Replace newline characters with spaces
    start_words= start_words.replace('\n', ' ')
     
    # Tokenize input seed prompt into lowercase words
    words = tokenizer.tokenize(start_words.lower())
    # Convert last seq_length words of prompt into input tensor
    input_tensor = words_to_tensor(words[-seq_length:], word_to_int)

    # Initialize hidden states with batch size 1 on CPU
    hidden1, hidden2 = model.init_hidden(batch_size=1, device=torch.device('cpu'))

    # Loop to generate 'length' number of new words sequentially
    for _ in range(length):
        # Disable gradient calculations during generation
        with torch.no_grad():
            # Pass current input tensor and hidden states through model
            output, hidden1, hidden2 = model(input_tensor, hidden1, hidden2)
        # Find index of word with highest predicted score
        predicted_index = torch.argmax(output, dim=1).item()
        # Convert predicted index back to word string
        predicted_word = int_to_word[predicted_index]  
        # Append predicted word to text words list
        words.append(predicted_word)
        # Update input tensor with latest seq_length words
        input_tensor = words_to_tensor(words[-seq_length:], word_to_int)

    # Join word tokens back into a readable string
    return ' '.join(words)

# Define seed phrase to prompt text generation
start_words = (
    "Alice was beginning to get very tired of "
)

# Generate continuation of the sentence using the trained RNN model
predicted_sequence = generate_sequence(start_words, model, word_to_int, int_to_word)
# Print the generated sentence
print(f"Generated sequence: {predicted_sequence}")

###################################################   GRU    ###################################

# Re-import torch
import torch
# Re-import neural network module
import torch.nn as nn
# Re-import optimizers
import torch.optim as optim
# Re-import DataLoader and TensorDataset
from torch.utils.data import DataLoader, TensorDataset
# Re-import numpy
import numpy as np

# Define Gated Recurrent Unit (GRU) model architecture
class GRUModel(nn.Module):
    # Initialize GRU layers and embedding
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        # Call parent constructor
        super(GRUModel, self).__init__()
        # Embedding layer mapping word indices to vectors
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # 1st GRU layer
        self.gru1 = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        # 2nd GRU layer
        self.gru2 = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        # Linear layer mapping hidden states to vocabulary logits
        self.fc = nn.Linear(hidden_dim, output_dim)
        # Hidden dimension size
        self.hidden_dim = hidden_dim

    # Forward pass through GRU model
    def forward(self, x, hidden1, hidden2):
        # Embed input word indices
        embedded = self.embedding(x)
        # Pass through 1st GRU layer
        out1, hidden1 = self.gru1(embedded, hidden1)
        # Pass through 2nd GRU layer
        out2, hidden2 = self.gru2(out1, hidden2)
        # Linear layer on last time step output
        out = self.fc(out2[:, -1, :])
        # Return output logits and hidden states
        return out, hidden1, hidden2

    # Initialize GRU hidden states with zeros
    def init_hidden(self, batch_size, device):
        # Return zero tensors for gru1 and gru2 hidden states
        return (
            torch.zeros(1, batch_size, self.hidden_dim).to(device),
            torch.zeros(1, batch_size, self.hidden_dim).to(device),
        )

# Select GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Input vocabulary size
input_size = len(word_to_int)
# Output vocabulary size
output_size = len(word_to_int)
# Word embedding dimension
embedding_dim = 256
# GRU hidden units
hidden_dim = 512
# Training batch size
batch_size = 32
# Total training epochs
epochs = 100

# Instantiate GRUModel and move to GPU/CPU
model = GRUModel(input_size, embedding_dim, hidden_dim, output_size).to(device)

# Define CrossEntropyLoss function
loss_function = nn.CrossEntropyLoss()
# Define Adam optimizer with learning rate 0.0001
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Extract inputs from sequence tuples
inputs = np.array([item[0] for item in sequences])
# Extract targets from sequence tuples
targets = np.array([item[1] for item in sequences])

# Convert inputs array to LongTensor
inputs_tensor = torch.tensor(inputs, dtype=torch.long)
# Convert targets array to LongTensor
targets_tensor = torch.tensor(targets, dtype=torch.long)

# Combine inputs and targets into TensorDataset
dataset = TensorDataset(inputs_tensor, targets_tensor)
# Create DataLoader for mini-batching and shuffling
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Loop over training epochs for GRU
for epoch in range(epochs):
    # Initialize epoch loss
    epoch_loss = 0

    # Iterate over each batch
    for batch_idx, (inputs, targets) in enumerate(data_loader):
        # Move inputs and targets to device
        inputs, targets = inputs.to(device), targets.to(device)
        # Initialize hidden states
        hidden1, hidden2 = model.init_hidden(inputs.size(0), device)

        # Reset gradients
        optimizer.zero_grad()
        # Forward pass through GRU
        output, hidden1, hidden2 = model(inputs, hidden1, hidden2)
        # Compute loss
        loss = loss_function(output, targets)
        # Backward pass
        loss.backward()
        # Update weights
        optimizer.step()

        # Add batch loss
        epoch_loss += loss.item()

    # Print epoch training progress
    print(f'Epoch {epoch + 1}, Loss: {epoch_loss / len(data_loader)}')
# Save trained GRU model to disk
torch.save(model, 'my_gru_entire_model.pth')
# Re-import torch
import torch
# Re-import numpy
import numpy as np

# Load trained GRU model
model = torch.load('my_gru_entire_model.pth', map_location=torch.device('cpu'),weights_only=False)
# Set model to evaluation mode
model.eval()

# Helper function to convert words to tensor
def words_to_tensor(words, word_to_int):
    # Map words to indices
    indices = [word_to_int[word] for word in words]
    # Wrap into PyTorch LongTensor
    tensor = torch.tensor(np.array([indices]), dtype=torch.long)
    # Return tensor
    return tensor

# Function to generate sequence with trained GRU model
def generate_sequence(start_words, model, word_to_int, int_to_word, seq_length=5, length=2):
    # Replace newlines with spaces
    start_words= start_words.replace('\n', ' ')
    # Tokenize input seed prompt
    words = tokenizer.tokenize(start_words.lower())  
    # Create input tensor from last seq_length words
    input_tensor = words_to_tensor(words[-seq_length:], word_to_int)

    # Initialize GRU hidden states
    hidden1, hidden2 = model.init_hidden(batch_size=1, device=torch.device('cpu'))

    # Loop to generate 'length' words
    for _ in range(length):
        # Disable gradients during generation
        with torch.no_grad():
            # Pass through GRU model
            output, hidden1, hidden2 = model(input_tensor, hidden1, hidden2)
        # Get highest probability word index
        predicted_index = torch.argmax(output, dim=1).item()
        # Map index to word string
        predicted_word = int_to_word[predicted_index]  
        # Append predicted word
        words.append(predicted_word)
        # Update input tensor with latest words
        input_tensor = words_to_tensor(words[-seq_length:], word_to_int)

    # Return generated sentence string
    return ' '.join(words)

# Define seed words for GRU generation
start_words = (
    "There was nothing so VERY remarkable"
)

# Generate continuation using GRU model
predicted_sequence = generate_sequence(start_words, model, word_to_int, int_to_word)
# Print generated sequence
print(f"Generated sequence: {predicted_sequence}")

##################################################### LSTM ###################################
# Re-import torch
import torch
# Re-import neural network module
import torch.nn as nn
# Re-import optimizers
import torch.optim as optim
# Re-import DataLoader and TensorDataset
from torch.utils.data import DataLoader, TensorDataset
# Re-import numpy
import numpy as np

# Define Long Short-Term Memory (LSTM) network architecture
class LSTMModel(nn.Module):
    # Initialize LSTM layers, embedding, and linear layer
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        # Call parent constructor
        super(LSTMModel, self).__init__()
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # 1st LSTM layer
        self.lstm1 = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        # 2nd LSTM layer
        self.lstm2 = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        # Final classification linear layer
        self.fc = nn.Linear(hidden_dim, output_dim)
        # Hidden dimension size
        self.hidden_dim = hidden_dim

    # Forward pass through LSTM
    def forward(self, x, hidden1, hidden2):
        # Convert word indices to dense vectors
        embedded = self.embedding(x)
        # Pass through 1st LSTM layer
        out1, hidden1 = self.lstm1(embedded, hidden1)
        # Pass through 2nd LSTM layer
        out2, hidden2 = self.lstm2(out1, hidden2)
        # Linear layer on hidden state of last time step
        out = self.fc(out2[:, -1, :])
        # Return logits and updated (hidden_state, cell_state) tuples
        return out, hidden1, hidden2

    # Initialize LSTM hidden states and cell states with zeros
    def init_hidden(self, batch_size, device):
        # LSTM requires both hidden state (h) and cell state (c) for each layer
        return (
            (torch.zeros(1, batch_size, self.hidden_dim).to(device),  
             torch.zeros(1, batch_size, self.hidden_dim).to(device)),  
            (torch.zeros(1, batch_size, self.hidden_dim).to(device),  
             torch.zeros(1, batch_size, self.hidden_dim).to(device))   
        )

# Select GPU if available, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Input vocabulary size
input_size = len(word_to_int)
# Output vocabulary size
output_size = len(word_to_int)
# Embedding dimension
embedding_dim = 256
# LSTM hidden dimension
hidden_dim = 512
# Batch size
batch_size = 32
# Training epochs
epochs = 100

# Instantiate LSTMModel and send to GPU/CPU
model = LSTMModel(input_size, embedding_dim, hidden_dim, output_size).to(device)

# Define CrossEntropyLoss
loss_function = nn.CrossEntropyLoss()
# Define Adam optimizer
optimizer = optim.Adam(model.parameters(), lr=0.0001)
# Extract inputs array
inputs = np.array([item[0] for item in sequences])
# Extract targets array
targets = np.array([item[1] for item in sequences])

# Convert inputs to LongTensor
inputs_tensor = torch.tensor(inputs, dtype=torch.long)
# Convert targets to LongTensor
targets_tensor = torch.tensor(targets, dtype=torch.long)

# Wrap in TensorDataset
dataset = TensorDataset(inputs_tensor, targets_tensor)
# Create DataLoader
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Loop over epochs for LSTM training
for epoch in range(epochs):
    # Initialize epoch loss
    epoch_loss = 0

    # Iterate over batches
    for batch_idx, (inputs, targets) in enumerate(data_loader):
        # Move inputs and targets to device
        inputs, targets = inputs.to(device), targets.to(device)
        # Initialize hidden and cell states for LSTM
        hidden1, hidden2 = model.init_hidden(inputs.size(0), device)

        # Clear gradients
        optimizer.zero_grad()
        # Forward pass through LSTM
        output, hidden1, hidden2 = model(inputs, hidden1, hidden2)

        # Compute loss
        loss = loss_function(output, targets)
        # Backward pass
        loss.backward()
        # Update weights
        optimizer.step()

        # Accumulate loss
        epoch_loss += loss.item()

    # Print epoch progress
    print(f'Epoch {epoch + 1}, Loss: {epoch_loss / len(data_loader)}')
# Save trained LSTM model to disk
torch.save(model, 'my_lstm_entire_model.pth')

# Re-import torch
import torch
# Re-import numpy
import numpy as np

# Load trained LSTM model checkpoint
model = torch.load('my_lstm_entire_model.pth', map_location=torch.device('cpu'),weights_only=False)
# Set model to evaluation mode
model.eval()

# Helper function to convert words to tensor
def words_to_tensor(words, word_to_int):
    # Map words to indices
    indices = [word_to_int[word] for word in words]
    # Create tensor
    tensor = torch.tensor(np.array([indices]), dtype=torch.long)
    # Return tensor
    return tensor

# Function to generate text continuation using LSTM
def generate_sequence(start_words, model, word_to_int, int_to_word, seq_length=5, length=1):
    # Replace newlines with spaces
    start_words= start_words.replace('\n', ' ')
    # Tokenize input seed prompt
    words = tokenizer.tokenize(start_words.lower())  
    # Create input tensor from last seq_length words
    input_tensor = words_to_tensor(words[-seq_length:], word_to_int)

    # Initialize LSTM hidden and cell states
    hidden1, hidden2 = model.init_hidden(batch_size=1, device=torch.device('cpu'))

    # Loop to generate 'length' words
    for _ in range(length):
        # Disable gradients during inference
        with torch.no_grad():
            # Forward pass
            output, hidden1, hidden2 = model(input_tensor, hidden1, hidden2)
        # Get predicted word index
        predicted_index = torch.argmax(output, dim=1).item()
        # Map to word string
        predicted_word = int_to_word[predicted_index]  
        # Append predicted word
        words.append(predicted_word)
        # Update input tensor with latest words
        input_tensor = words_to_tensor(words[-seq_length:], word_to_int)

    # Return generated sequence string
    return ' '.join(words)

# Define seed words for LSTM generation
start_words = (
    "Alice was beginning to get very tired of"
)

# Generate continuation with LSTM model
predicted_sequence = generate_sequence(start_words, model, word_to_int, int_to_word)
# Print generated sequence
print(f"Generated sequence: {predicted_sequence}")
