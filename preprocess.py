import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Load the dataset
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

df = pd.read_csv("dpdp_clauses.csv")

# Preprocessing function
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Tokenize text
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words("english"))
    tokens = [word for word in tokens if word not in stop_words]
    # Lemmatize tokens
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    # Join tokens back to text
    return " ".join(tokens)

# Apply preprocessing to the "Clause Text" column
df["Processed Clause Text"] = df["Clause Text"].apply(preprocess_text)

# Save the processed dataset
processed_csv = "dpdp_clauses_processed.csv"
df.to_csv(processed_csv, index=False)

print(f"Processed dataset saved to {processed_csv}")
