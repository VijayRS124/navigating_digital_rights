import pandas as pd
import numpy as np
import torch
import os
import logging
import re
from fastapi import FastAPI, HTTPException, Header, Depends, File, UploadFile
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sentence_transformers import SentenceTransformer
from sklearn.metrics import accuracy_score, log_loss
from pdfminer.high_level import extract_text
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- API & Gemini Keys ---
API_KEY = os.getenv("API_KEY")
API_KEY_NAME = "X-API-Key"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY is None:
    raise ValueError("Gemini API key not set")

genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel('gemini-1.5-pro')

# --- API Key Dependency ---
def get_api_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return x_api_key

# --- FastAPI App ---
app = FastAPI(title="GDPR Compliance API", version="2.0")

# --------------------------------------
# Load & Preprocess Data
# --------------------------------------
def load_and_preprocess_data():
    df1 = pd.read_csv("gdpr_violations_noisy.csv")
    df2 = pd.read_csv("gdpr_text_noisy.csv")
    dataset = pd.concat([df1, df2]).drop_duplicates()
    dataset['summary'] = dataset['summary'].fillna('')
    
    le = LabelEncoder()
    dataset['condition'] = le.fit_transform(dataset['condition'])
    
    return dataset, le

# --------------------------------------
# Train-Test Split
# --------------------------------------
def train_test_split_data(dataset):
    return train_test_split(
        dataset['summary'], dataset['condition'], test_size=0.2, random_state=42, stratify=dataset['condition']
    )

# --------------------------------------
# Load LegalBERT Model
# --------------------------------------
def load_legalbert_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer("nlpaueb/legal-bert-base-uncased", device=device)
    return model

# --------------------------------------
# Train XGBoost Model
# --------------------------------------
def train_xgb_model(X_train_embeddings, y_train):
    model = XGBClassifier(eval_metric='logloss', random_state=42)
    model.fit(X_train_embeddings, y_train)
    return model

# --------------------------------------
# Evaluate Model
# --------------------------------------
def evaluate_model(model, X_test_embeddings, y_test):
    y_pred = model.predict(X_test_embeddings)
    y_pred_proba = model.predict_proba(X_test_embeddings)
    acc = accuracy_score(y_test, y_pred)
    loss = log_loss(y_test, y_pred_proba)
    logger.info(f"Model Accuracy: {acc:.4f}, Log Loss: {loss:.4f}")

# --------------------------------------
# Initialize Models
# --------------------------------------
logger.info("Initializing models...")
try:
    dataset, le = load_and_preprocess_data()
    X_train, X_test, y_train, y_test = train_test_split_data(dataset)
    
    logger.info("Loading LegalBERT model...")
    legalbert_model = load_legalbert_model()
    logger.info("LegalBERT model loaded successfully!")

    logger.info("Encoding training and test data...")
    X_train_embeddings = np.array(legalbert_model.encode(X_train.tolist(), batch_size=16, convert_to_numpy=True))
    X_test_embeddings = np.array(legalbert_model.encode(X_test.tolist(), batch_size=16, convert_to_numpy=True))

    logger.info("Training XGBoost model...")
    xgb_model = train_xgb_model(X_train_embeddings, y_train)
    logger.info("XGBoost model trained successfully!")

    logger.info("Evaluating model...")
    evaluate_model(xgb_model, X_test_embeddings, y_test)

    app.state.legalbert_model = legalbert_model
    app.state.xgb_model = xgb_model
except Exception as e:
    logger.error(f"❌ Error during model initialization: {e}")
    raise RuntimeError("Model initialization failed.")

# --------------------------------------
# Process Uploaded File (.txt or .pdf)
# --------------------------------------
def extract_text_from_file(file: UploadFile) -> list:
    try:
        content = file.file.read()
        filename = file.filename.lower()

        if filename.endswith('.txt'):
            text = content.decode("utf-8").strip()
        elif filename.endswith('.pdf'):
            text = extract_text(file.file)
        else:
            raise ValueError("Unsupported file format. Only .txt and .pdf are allowed.")

        if not text.strip():
            raise ValueError("File is empty")
        
        return [line.strip() for line in text.split("\n") if line.strip()]
    except Exception as e:
        logger.error(f"Error processing file: {e}")
        raise HTTPException(status_code=400, detail="Invalid file format.")

# --------------------------------------
# Prediction Endpoint
# --------------------------------------
@app.post("/predict", response_class=JSONResponse)
def predict(file: UploadFile = File(...), api_key: str = Depends(get_api_key)):
    """
    Predict GDPR compliance for multiple rules in a text or PDF file.
    """
    lines = extract_text_from_file(file)

    # Ensure model is trained before prediction
    try:
        legalbert_model = app.state.legalbert_model
        xgb_model = app.state.xgb_model
    except AttributeError:
        raise HTTPException(status_code=500, detail="Model not initialized.")

    # Make predictions
    text_embeddings = np.array(legalbert_model.encode(lines, batch_size=16, convert_to_numpy=True))
    predictions = xgb_model.predict(text_embeddings)

    label_map = {0: "violated", 1: "compliance"}
    results = []

    # Generate Gemini explanations
# Generate Gemini explanations
    for rule, prediction in zip(lines, predictions):
        predicted_label = label_map.get(prediction, "Unknown")
        
        prompt = (
            f"Analyze the following text and explain why it is considered '{predicted_label}' under GDPR."
            f"\nProvide a concise explanation in no more than 5-6 lines:\n\n{rule}"
        )

        try:
            gemini_response = gemini_model.generate_content(prompt)
            gemini_summary = gemini_response.text if gemini_response else "No response from Gemini API."
            
            # Limit response to 5-6 lines
            gemini_summary = "\n".join(gemini_summary.split("\n")[:6])
        except Exception as e:
            logger.error(f"Gemini API Error: {e}")
            gemini_summary = "Gemini API unavailable."

        results.append({
            "rule": rule,
            "prediction": predicted_label,
            "gemini_summary": gemini_summary
        })


    return JSONResponse(content={"file_name": file.filename, "predictions": results})

# --------------------------------------
# Root Endpoint
# --------------------------------------
@app.get("/")
def read_root(api_key: str = Depends(get_api_key)):
    return {"message": "Welcome to the GDPR Compliance API! Upload a .txt or .pdf file to /predict."}

# Run the app: uvicorn app:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)