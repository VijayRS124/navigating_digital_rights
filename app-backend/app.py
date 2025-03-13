from fastapi import FastAPI, HTTPException, Header, Depends, File, UploadFile
from fastapi.responses import JSONResponse
import pandas as pd
import re
import logging
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for servers
import matplotlib.pyplot as plt
from collections import Counter

# Scikit-learn & imblearn imports
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.decomposition import TruncatedSVD
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline  # imblearn pipeline to work with SMOTE
from dotenv import load_dotenv
import os



load_dotenv()
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- API KEY Setup ---
API_KEY = os.getenv("API_KEY")
API_KEY_NAME = "X-API-Key"

def get_api_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return x_api_key

# Initialize FastAPI
app = FastAPI(title="GDPR Compliance Prediction API", version="1.0")

# --------------------------------------
# Custom Preprocess Function
# --------------------------------------
def preprocess_text(text: str) -> str:
    """
    Preprocess a single document: lowercases text and removes non-alphabetic characters.
    """
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

# --------------------------------------
# Load and Prepare Dataset
# --------------------------------------
df1 = pd.read_csv("gdpr_violations_noisy.csv")
df2 = pd.read_csv("gdpr_text_noisy.csv")
dataset = pd.concat([df1, df2]).drop_duplicates()
dataset['summary'] = dataset['summary'].fillna('')

X_text = dataset['summary']
y = dataset['condition']  # 0 = violation, 1 = compliance

# Split into training and test sets (stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X_text, y, test_size=0.2, random_state=42, stratify=y
)

# --------------------------------------
# Define Pipeline for XGBoost Model
# --------------------------------------
def create_pipeline(classifier):
    return Pipeline([
        ('tfidf', TfidfVectorizer(preprocessor=preprocess_text, stop_words='english', max_features=10000)),
        ('to_dense', FunctionTransformer(lambda x: x.toarray(), accept_sparse=True)),
        ('smote', SMOTE(random_state=42)),
        ('clf', classifier)
    ])

xgboost_model = XGBClassifier(eval_metric='logloss', random_state=42, use_label_encoder=False)
pipeline = create_pipeline(xgboost_model)

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
y_probs = pipeline.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, y_probs)

# --------------------------------------
# Define Prediction Endpoint
# --------------------------------------
@app.post("/predict", response_class=JSONResponse)
def predict(file: UploadFile = File(...), api_key: str = Depends(get_api_key)):
    """
    Predict GDPR compliance based on input text file.
    Request body should be a .txt file.
    Output will be either "violated" or "compliance".
    """
    try:
        content = file.file.read().decode("utf-8")
    except Exception as e:
        logger.error("Error reading file: %s", e)
        raise HTTPException(status_code=400, detail="Invalid file format. Please upload a valid .txt file.")
    
    prediction = pipeline.predict([content])[0]
    label_map = {0: "violated", 1: "compliance"}
    predicted_label = label_map.get(prediction, "Unknown")
    
    response = {
        "file_name": content,
        "prediction": predicted_label,
    }
    return JSONResponse(content=response)

@app.get("/")
def read_root(api_key: str = Depends(get_api_key)):
    return {"message": "Welcome to the GDPR Compliance Prediction API! Use the /predict endpoint to upload a .txt file."}

# To run the app, use a command like: uvicorn your_filename:app --reload
if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)