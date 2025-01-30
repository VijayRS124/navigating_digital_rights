import torch
import torch.nn as nn
import nltk
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

# Load vocabulary and label encoder
df = pd.read_csv(r'C:\Users\kishore l\navigating_digital_rights\preprocessed_gdpr.csv')
nltk.download('punkt')

# Tokenization
tokenized_texts = [nltk.word_tokenize(text) for text in df['clean_text'].astype(str)]

# Build Vocabulary
from collections import Counter
word_counts = Counter(word for text in tokenized_texts for word in text)
vocab = {word: i+2 for i, (word, _) in enumerate(word_counts.most_common(5000))}
vocab["<PAD>"] = 0
vocab["<UNK>"] = 1

# Load label encoder
label_encoder = LabelEncoder()
label_encoder.fit(df['chapter_title'].astype(str))

# Load Model
VOCAB_SIZE = len(vocab)
EMBED_DIM = 128
HIDDEN_DIM = 128
NUM_CLASSES = len(label_encoder.classes_)

model = GDPRLSTM(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, NUM_CLASSES)
model.load_state_dict(torch.load(r'C:\Users\kishore l\navigating_digital_rights\gdpr_lstm_model.pth', map_location=device))
model.to(device)
model.eval()

import torch.nn.functional as F

def predict_chapter(text):
    # Preprocess input text
    tokens = nltk.word_tokenize(text.lower())
    text_seq = [vocab.get(word, vocab["<UNK>"]) for word in tokens]

    # Padding
    MAX_SEQUENCE_LENGTH = 100
    text_seq = text_seq[:MAX_SEQUENCE_LENGTH] + [0] * (MAX_SEQUENCE_LENGTH - len(text_seq))
    
    # Convert to tensor
    text_tensor = torch.tensor([text_seq], dtype=torch.long).to(device)
    
    # Make prediction
    with torch.no_grad():
        output = model(text_tensor)
        probabilities = F.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
    
    # Get chapter title
    predicted_label = label_encoder.inverse_transform([predicted_class])[0]
    return predicted_label, probabilities[0][predicted_class].item()
sample_text = "This regulation ensures the protection of personal data and specifies rules for its lawful processing."
predicted_chapter, confidence = predict_chapter(sample_text)

print(f"Predicted Chapter: {predicted_chapter}")
print(f"Confidence Score: {confidence:.4f}")
