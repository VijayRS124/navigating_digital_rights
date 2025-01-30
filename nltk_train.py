import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import nltk
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from collections import Counter

# Load GDPR dataset
df = pd.read_csv('preprocessed_gdpr.csv')

# Select Features and Target
X_text = df['clean_text'].astype(str)  # Convert to string (in case of NaN values)
y_label = df['chapter_title'].astype(str)

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_label)

# Tokenization
nltk.download('punkt')
tokenized_texts = [nltk.word_tokenize(text) for text in X_text]

# Build Vocabulary
word_counts = Counter(word for text in tokenized_texts for word in text)
vocab = {word: i+2 for i, (word, _) in enumerate(word_counts.most_common(5000))}  # Limit vocab to 5000
vocab["<PAD>"] = 0
vocab["<UNK>"] = 1

# Convert words to indices
def text_to_sequence(text):
    return [vocab.get(word, vocab["<UNK>"]) for word in text]

X_sequences = [text_to_sequence(text) for text in tokenized_texts]

# Padding sequences
MAX_SEQUENCE_LENGTH = 100
X_padded = [seq[:MAX_SEQUENCE_LENGTH] + [0] * (MAX_SEQUENCE_LENGTH - len(seq)) for seq in X_sequences]

# Convert to tensors
X_tensor = torch.tensor(X_padded, dtype=torch.long)
y_tensor = torch.tensor(y, dtype=torch.long)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42, stratify=y)

# Create Dataset & DataLoader
class GDPRDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = GDPRDataset(X_train, y_train)
test_dataset = GDPRDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define LSTM Model
class GDPRLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super(GDPRLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x)
        x = self.dropout(lstm_out[:, -1, :])  # Take last hidden state
        x = self.fc(x)
        return x

# Model Parameters
VOCAB_SIZE = len(vocab)
EMBED_DIM = 128
HIDDEN_DIM = 128
NUM_CLASSES = len(label_encoder.classes_)

# Initialize Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GDPRLSTM(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, NUM_CLASSES).to(device)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Function
def train(model, train_loader, criterion, optimizer, epochs=10):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct += (outputs.argmax(1) == y_batch).sum().item()
        
        accuracy = correct / len(train_loader.dataset)
        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}, Accuracy: {accuracy:.4f}")

# Training Model
train(model, train_loader, criterion, optimizer, epochs=10)

# Evaluation Function
def evaluate(model, test_loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            correct += (outputs.argmax(1) == y_batch).sum().item()
    
    accuracy = correct / len(test_loader.dataset)
    print(f"Test Accuracy: {accuracy:.4f}")

# Evaluate Model
evaluate(model, test_loader)

# Save Model
torch.save(model.state_dict(), "gdpr_lstm_model.pth")
