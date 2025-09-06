# predict_combined.py

import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import argparse
import torch.nn.functional as F
from pathlib import Path


MODEL_DIR = Path(__file__).parent / "deberta-combined"   # <- put your combined model folder here
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.to(device)
model.eval()

# Labels (adapt if your combined model uses different ones)
FEVER_LABELS = ["REFUTES", "SUPPORTS", "NOT ENOUGH INFO"]

BINARY_LABELS = {
    "SUPPORTS": "TRUE",
    "REFUTES": "FALSE",
    "NOT ENOUGH INFO": "UNKNOWN"
}


def predict_text(text: str):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = F.softmax(logits, dim=-1)
        score, pred_idx = torch.max(probs, dim=-1)
        fever_label = FEVER_LABELS[pred_idx.item()]
        binary_label = BINARY_LABELS[fever_label]
        return {"text": text, "label": binary_label, "score": score.item()}


def predict_csv(input_csv: str, out_csv: str, text_column: str = "text"):
    df = pd.read_csv(input_csv)
    predictions = []

    for _, row in df.iterrows():
        text = str(row.get(text_column, "")).strip()
        if not text:
            predictions.append({"label": None, "score": None})
            continue
        pred = predict_text(text)
        predictions.append({"label": pred["label"], "score": pred["score"]})

    pred_df = pd.DataFrame(predictions)
    result_df = pd.concat([df.reset_index(drop=True), pred_df], axis=1)
    result_df.to_csv(out_csv, index=False)
    print(f"✅ Predictions saved to {out_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict with Combined Model (FEVER + LIAR + ...).")
    parser.add_argument("--text", type=str, help="Single text to classify")
    parser.add_argument("--csv", type=str, help="CSV file to classify")
    parser.add_argument("--out", type=str, default="predictions_combined.csv", help="Output CSV file")
    parser.add_argument("--text_column", type=str, default="text", help="Column name in CSV to classify")
    args = parser.parse_args()

    if args.text:
        pred = predict_text(args.text)
        print(f"Text: {pred['text']}")
        print(f"Label: {pred['label']}")
        print(f"Score: {pred['score']*100:.2f}%")
    elif args.csv:
        predict_csv(args.csv, args.out, text_column=args.text_column)
    else:
        print("⚠️ Please provide either --text or --csv")
