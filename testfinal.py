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
