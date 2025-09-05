# api/app.py
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
import sys
import pandas as pd
from io import BytesIO
from pathlib import Path

# Add the model folder to sys.path so we can import predict_fever
sys.path.append(str(Path(__file__).parent.parent / "model"))
import predict_fever as pred_module  # import your existing functions

app = FastAPI(title="Fake News Detector API")

# Request schema for single text
class NewsRequest(BaseModel):
    text: str

# Routes
@app.get("/")
def root():
    return {"message": "Fake News Detection API running"}

@app.post("/predict_text")
def predict_text_endpoint(request: NewsRequest):
    return pred_module.predict_text(request.text)

@app.post("/predict_csv")
async def predict_csv_endpoint(file: UploadFile = File(...), text_column: str = "text"):
    # Read uploaded CSV
    df = pd.read_csv(file.file)
    out_csv = "temp_predictions.csv"

    # Since your predict_csv expects file path, we save the uploaded CSV temporarily
    temp_input = Path("temp_input.csv")
    df.to_csv(temp_input, index=False)

    # Call the original prediction function
    pred_module.predict_csv(input_csv=str(temp_input), out_csv=out_csv, text_column=text_column)

    # Load predictions
    result_df = pd.read_csv(out_csv)

    # Extract FAKE / UNKNOWN news
    fake_unknown = []
    for _, row in result_df.iterrows():
        if row["label"] in ["FALSE", "UNKNOWN"]:
            fake_unknown.append({
                "source": row.get("source", ""),
                "title": row.get("title", ""),
                "date": row.get("created_utc", ""),
                "text": row.get(text_column, ""),
                "label": row["label"],
                "score": row["score"] * 100  # convert to percentage
            })

    # Return CSV as string + list of FAKE/UNKNOWN
    csv_bytes = result_df.to_csv(index=False).encode("utf-8")
    return {"csv": csv_bytes, "fake_unknown": fake_unknown}
