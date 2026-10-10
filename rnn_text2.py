# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 15:54:35 2025

@author: mithun
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader,TensorDataset
import os
import string

import os

input_file = r"f:/Amrita_2026/22AIE304(Deep Learning)/Codes/Endsem_codes/Alice1.txt.txt"

print("Checking file:")
print(input_file)

if not os.path.exists(input_file):
    print("ERROR: File does not exist!")
else:
    print("File exists.")

    with open(input_file, "r", encoding="utf-8") as infile:
        text_data = infile.read()

    print("Number of characters:", len(text_data))
    print("First 500 characters:")
    print(text_data[:500])

input_file = "f:/Amrita_2026/22AIE304(Deep Learning)/Codes/Endsem_codes/Alice1.txt.txt"
punctuation = string.punctuation.replace('.', '').replace('?', '')

text_data=""
if os.path.exists(input_file) :
    with open(input_file, 'r') as infile:
        for line in infile:
            if line.strip():
                for char in punctuation:
                    line = line.replace(char, '')
                text_data += line

print("Processed Text:")
print(text_data)

from nltk.tokenize import TreebankWordTokenizer

tokenizer = TreebankWordTokenizer()
words = tokenizer.tokenize(text_data.lower())
print(words)

word_to_int = {w: i for i, w in enumerate(set(words))}
int_to_word = {i: w for w, i in word_to_int.items()}
len(word_to_int)

def create_sequences(input_data, seq_length):
    sequences = []
    for i in range(0, len(input_data) - seq_length):
        sequence_in = input_data[i:i + seq_length]
        sequence_out = input_data[i + seq_length]
        sequences.append((sequence_in, sequence_out))
    return sequences

encoded_text = np.array([word_to_int[word] for word in words])
seq_length = 5
sequences = create_sequences(encoded_text, seq_length)

########################################### RNN  ####################################################
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

class RNNModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(RNNModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn1 = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.rnn2 = nn.RNN(hidden_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.hidden_dim = hidden_dim

    def forward(self, x, hidden1, hidden2):
        embedded = self.embedding(x)
        out1, hidden1 = self.rnn1(embedded, hidden1)
        out2, hidden2 = self.rnn2(out1, hidden2)
        out = self.fc(out2[:, -1, :])
        return out, hidden1, hidden2

    def init_hidden(self, batch_size, device):
        return (
            torch.zeros(1, batch_size, self.hidden_dim).to(device),
            torch.zeros(1, batch_size, self.hidden_dim).to(device),
        )

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_size = len(word_to_int)
output_size = len(word_to_int)
embedding_dim = 256
hidden_dim = 512
batch_size = 32
epochs = 100

model = RNNModel(input_size, embedding_dim, hidden_dim, output_size).to(device)

loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
inputs = np.array([item[0] for item in sequences])
targets = np.array([item[1] for item in sequences])

inputs_tensor = torch.tensor(inputs, dtype=torch.long)
targets_tensor = torch.tensor(targets, dtype=torch.long)

dataset = TensorDataset(inputs_tensor, targets_tensor)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

for epoch in range(epochs):
    epoch_loss = 0

    for batch_idx, (inputs, targets) in enumerate(data_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        hidden1, hidden2 = model.init_hidden(inputs.size(0), device)

        optimizer.zero_grad()
        output, hidden1, hidden2 = model(inputs, hidden1, hidden2)
        loss = loss_function(output, targets)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f'Epoch {epoch + 1}, Loss: {epoch_loss / len(data_loader)}')

torch.save(model, 'RNN_61.pth')

import torch
import numpy as np

model = torch.load('RNN_61.pth', map_location=torch.device('cpu'),weights_only=False)
model.eval()

def words_to_tensor(words, word_to_int):
    indices = [word_to_int[word] for word in words]
    tensor = torch.tensor(np.array([indices]), dtype=torch.long)
    return tensor

def generate_sequence(start_words, model, word_to_int, int_to_word, seq_length=5, length=2):
    start_words= start_words.replace('\n', ' ')
     
    words = tokenizer.tokenize(start_words.lower())
    input_tensor = words_to_tensor(words[-seq_length:], word_to_int)

    hidden1, hidden2 = model.init_hidden(batch_size=1, device=torch.device('cpu'))

    for _ in range(length):
        with torch.no_grad():
            output, hidden1, hidden2 = model(input_tensor, hidden1, hidden2)
        predicted_index = torch.argmax(output, dim=1).item()
        predicted_word = int_to_word[predicted_index]  
        words.append(predicted_word)
        input_tensor = words_to_tensor(words[-seq_length:], word_to_int)

    return ' '.join(words)

start_words = (
    "Alice was beginning to get very tired of "
)

predicted_sequence = generate_sequence(start_words, model, word_to_int, int_to_word)
print(f"Generated sequence: {predicted_sequence}")

###################################################   GRU    ###################################

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

class GRUModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(GRUModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru1 = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.gru2 = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.hidden_dim = hidden_dim

    def forward(self, x, hidden1, hidden2):
        embedded = self.embedding(x)
        out1, hidden1 = self.gru1(embedded, hidden1)
        out2, hidden2 = self.gru2(out1, hidden2)
        out = self.fc(out2[:, -1, :])
        return out, hidden1, hidden2

    def init_hidden(self, batch_size, device):
        return (
            torch.zeros(1, batch_size, self.hidden_dim).to(device),
            torch.zeros(1, batch_size, self.hidden_dim).to(device),
        )

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_size = len(word_to_int)
output_size = len(word_to_int)
embedding_dim = 256
hidden_dim = 512
batch_size = 32
epochs = 100

model = GRUModel(input_size, embedding_dim, hidden_dim, output_size).to(device)

loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

inputs = np.array([item[0] for item in sequences])
targets = np.array([item[1] for item in sequences])

inputs_tensor = torch.tensor(inputs, dtype=torch.long)
targets_tensor = torch.tensor(targets, dtype=torch.long)

dataset = TensorDataset(inputs_tensor, targets_tensor)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

for epoch in range(epochs):
    epoch_loss = 0

    for batch_idx, (inputs, targets) in enumerate(data_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        hidden1, hidden2 = model.init_hidden(inputs.size(0), device)

        optimizer.zero_grad()
        output, hidden1, hidden2 = model(inputs, hidden1, hidden2)
        loss = loss_function(output, targets)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f'Epoch {epoch + 1}, Loss: {epoch_loss / len(data_loader)}')
torch.save(model, 'my_gru_entire_model.pth')
import torch
import numpy as np

model = torch.load('my_gru_entire_model.pth', map_location=torch.device('cpu'),weights_only=False)
model.eval()

def words_to_tensor(words, word_to_int):
    indices = [word_to_int[word] for word in words]
    tensor = torch.tensor(np.array([indices]), dtype=torch.long)
    return tensor

def generate_sequence(start_words, model, word_to_int, int_to_word, seq_length=5, length=2):
    start_words= start_words.replace('\n', ' ')
    words = tokenizer.tokenize(start_words.lower())  
    input_tensor = words_to_tensor(words[-seq_length:], word_to_int)

    hidden1, hidden2 = model.init_hidden(batch_size=1, device=torch.device('cpu'))

    for _ in range(length):
        with torch.no_grad():
            output, hidden1, hidden2 = model(input_tensor, hidden1, hidden2)
        predicted_index = torch.argmax(output, dim=1).item()
        predicted_word = int_to_word[predicted_index]  
        words.append(predicted_word)
        input_tensor = words_to_tensor(words[-seq_length:], word_to_int)

    return ' '.join(words)

start_words = (
    "There was nothing so VERY remarkable"
)

predicted_sequence = generate_sequence(start_words, model, word_to_int, int_to_word)
print(f"Generated sequence: {predicted_sequence}")

##################################################### LSTM  ###################################
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm1 = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.hidden_dim = hidden_dim

    def forward(self, x, hidden1, hidden2):
        embedded = self.embedding(x)
        out1, hidden1 = self.lstm1(embedded, hidden1)
        out2, hidden2 = self.lstm2(out1, hidden2)
        out = self.fc(out2[:, -1, :])
        return out, hidden1, hidden2

    def init_hidden(self, batch_size, device):
        return (
            (torch.zeros(1, batch_size, self.hidden_dim).to(device),  
             torch.zeros(1, batch_size, self.hidden_dim).to(device)),  
            (torch.zeros(1, batch_size, self.hidden_dim).to(device),  
             torch.zeros(1, batch_size, self.hidden_dim).to(device))   
        )

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_size = len(word_to_int)
output_size = len(word_to_int)
embedding_dim = 256
hidden_dim = 512
batch_size = 32
epochs = 100

model = LSTMModel(input_size, embedding_dim, hidden_dim, output_size).to(device)

loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
inputs = np.array([item[0] for item in sequences])
targets = np.array([item[1] for item in sequences])

inputs_tensor = torch.tensor(inputs, dtype=torch.long)
targets_tensor = torch.tensor(targets, dtype=torch.long)

dataset = TensorDataset(inputs_tensor, targets_tensor)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

for epoch in range(epochs):
    epoch_loss = 0

    for batch_idx, (inputs, targets) in enumerate(data_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        hidden1, hidden2 = model.init_hidden(inputs.size(0), device)

        optimizer.zero_grad()
        output, hidden1, hidden2 = model(inputs, hidden1, hidden2)

        loss = loss_function(output, targets)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f'Epoch {epoch + 1}, Loss: {epoch_loss / len(data_loader)}')
torch.save(model, 'my_lstm_entire_model.pth')

import torch
import numpy as np

model = torch.load('my_lstm_entire_model.pth', map_location=torch.device('cpu'),weights_only=False)
model.eval()

def words_to_tensor(words, word_to_int):
    indices = [word_to_int[word] for word in words]
    tensor = torch.tensor(np.array([indices]), dtype=torch.long)
    return tensor

def generate_sequence(start_words, model, word_to_int, int_to_word, seq_length=5, length=1):
    start_words= start_words.replace('\n', ' ')
    words = tokenizer.tokenize(start_words.lower())  
    input_tensor = words_to_tensor(words[-seq_length:], word_to_int)

    hidden1, hidden2 = model.init_hidden(batch_size=1, device=torch.device('cpu'))

    for _ in range(length):
        with torch.no_grad():
            output, hidden1, hidden2 = model(input_tensor, hidden1, hidden2)
        predicted_index = torch.argmax(output, dim=1).item()
        predicted_word = int_to_word[predicted_index]  
        words.append(predicted_word)
        input_tensor = words_to_tensor(words[-seq_length:], word_to_int)

    return ' '.join(words)

start_words = (
    "Alice was beginning to get very tired of"
)

predicted_sequence = generate_sequence(start_words, model, word_to_int, int_to_word)
print(f"Generated sequence: {predicted_sequence}")


