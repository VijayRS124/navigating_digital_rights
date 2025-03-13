from fastapi import FastAPI, HTTPException, Header, Depends, File, UploadFile
from fastapi.responses import JSONResponse
import pandas as pd
import re
import logging
import os

# Machine Learning Libraries
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for servers
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import accuracy_score, roc_auc_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Gemini API Integration
import google.generativeai as genai

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- API KEY Setup ---
API_KEY = os.getenv("API_KEY")
API_KEY_NAME = "X-API-Key"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY is None:
    raise ValueError("Gemini API key not set")

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-1.5-pro')

# API Key Dependency
def get_api_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return x_api_key

# Initialize FastAPI
app = FastAPI(title="GDPR Compliance Prediction API", version="1.2")

# --------------------------------------
# Custom Preprocess Function
# --------------------------------------
def preprocess_text(text: str) -> str:
    """Lowercases text and removes non-alphabetic characters."""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

# --------------------------------------
# Load and Prepare Dataset
# --------------------------------------
try:
    df1 = pd.read_csv("gdpr_violations_noisy.csv")
    df2 = pd.read_csv("gdpr_text_noisy.csv")
    dataset = pd.concat([df1, df2]).drop_duplicates()
    dataset['summary'] = dataset['summary'].fillna('')

    X_text = dataset['summary']
    y = dataset['condition']  # 0 = violation, 1 = compliance

    # Stratified train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_text, y, test_size=0.2, random_state=42, stratify=y
    )

    logger.info("Dataset successfully loaded and split.")

except Exception as e:
    logger.error(f"Error loading dataset: {e}")
    raise RuntimeError("Failed to load dataset. Check file paths.")

# --------------------------------------
# Define Pipeline for XGBoost Model
# --------------------------------------
def create_pipeline():
    return Pipeline([
        ('tfidf', TfidfVectorizer(preprocessor=preprocess_text, stop_words='english', max_features=10000)),
        ('to_dense', FunctionTransformer(lambda x: x.toarray(), accept_sparse=True)),
        ('smote', SMOTE(random_state=42)),
        ('clf', XGBClassifier(eval_metric='logloss', random_state=42))  # Removed use_label_encoder
    ])

pipeline = create_pipeline()

try:
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_probs = pipeline.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_probs)
    
    logger.info(f"Model trained successfully. Accuracy: {accuracy:.4f}, ROC-AUC: {roc_auc:.4f}")

except Exception as e:
    logger.error(f"Model training failed: {e}")
    raise RuntimeError("Failed to train the model.")

# --------------------------------------
# Define Prediction Endpoint (Multi-line Support)
# --------------------------------------
@app.post("/predict", response_class=JSONResponse)
def predict(file: UploadFile = File(...), api_key: str = Depends(get_api_key)):
    """
    Predict GDPR compliance for multiple lines in a text file.
    Each line is treated as a separate rule and checked individually.
    """
    try:
        content = file.file.read().decode("utf-8").strip()
        if not content:
            raise ValueError("File is empty")
        lines = [line.strip() for line in content.split("\n") if line.strip()]  # Process multiple lines
    except Exception as e:
        logger.error(f"Error reading file: {e}")
        raise HTTPException(status_code=400, detail="Invalid file format. Please upload a valid .txt file.")

    # Ensure model is trained before prediction
    try:
        check_is_fitted(pipeline)
    except Exception as e:
        logger.error(f"Model not trained: {e}")
        raise HTTPException(status_code=500, detail="Model is not trained. Try restarting the server.")

    # Make predictions for each line
    predictions = pipeline.predict(lines)
    label_map = {0: "violated", 1: "compliance"}

    # Generate Gemini API explanations
    results = []
    for rule, prediction in zip(lines, predictions):
        predicted_label = label_map.get(prediction, "Unknown")

        prompt = (
            f"Analyze the following text and explain why it is considered in 2 lines '{predicted_label}' "
            f"under GDPR guidelines. Provide a summary and any suggestions for improvement in 2 another 2 lines:\n\n{rule}"
        )

        try:
            gemini_response = model.generate_content(prompt)
            gemini_summary = gemini_response.text if gemini_response else "No response from Gemini API."
        except Exception as e:
            logger.error(f"Error with Gemini API: {e}")
            gemini_summary = "Gemini API unavailable."

        results.append({
            "rule": rule,
            "prediction": predicted_label,
            "gemini_summary": gemini_summary
        })

    return JSONResponse(content={"file_name": file.filename, "predictions": results})

@app.get("/")
def read_root(api_key: str = Depends(get_api_key)):
    return {"message": "Welcome to the GDPR Compliance Prediction API! Use the /predict endpoint to upload a .txt file."}

# To run the app: uvicorn app:app --reload
if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
