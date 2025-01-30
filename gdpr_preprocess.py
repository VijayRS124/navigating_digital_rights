import pandas as pd
import nltk
import re
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('wordnet')
df = pd.read_csv(r"C:\Users\kishore l\gdg\gdpr_text.csv")




# Download NLTK resources


from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

from nltk.tokenize import word_tokenize, sent_tokenize

def preprocess_text(text):
    text = text.lower()  
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = word_tokenize(text, language="english")  # Specify language
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

df['clean_text'] = df['gdpr_text'].apply(preprocess_text)
df.to_csv('preprocessed_gdpr.csv', index=False)

