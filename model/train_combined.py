# train_combined.py
import pandas as pd
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from pathlib import Path

# --------------------------
# CONFIG
# --------------------------
MODEL_NAME = "microsoft/deberta-v3-base"
OUTPUT_DIR = "./deberta-combined"

FINAL_LABELS = {"TRUE": 0, "FALSE": 1, "UNKNOWN": 2}

# --------------------------
# Map FEVER labels
# --------------------------
def map_fever(label):
    if label == "SUPPORTS":
        return FINAL_LABELS["TRUE"]
    elif label == "REFUTES":
        return FINAL_LABELS["FALSE"]
    else:
        return FINAL_LABELS["UNKNOWN"]

# --------------------------
# Map LIAR labels
# --------------------------
def map_liar(label):
    liar_map = {
        "true": "TRUE",
        "mostly-true": "TRUE",
        "false": "FALSE",
        "pants-fire": "FALSE",
        "barely-true": "FALSE",
        "half-true": "UNKNOWN"
    }
    return FINAL_LABELS[liar_map.get(label, "UNKNOWN")]

# --------------------------
# Load FEVER
# --------------------------
try:
    fever = load_dataset("fever", "v1.0")
    fever_train = fever["train"].to_pandas()
    fever_train = fever_train.rename(columns={"claim": "text", "label": "fever_label"})
    fever_train["label"] = fever_train["fever_label"].map(map_fever)
except Exception as e:
    print("Could not load FEVER:", e)
    fever_train = pd.DataFrame(columns=["text", "label"])

# --------------------------
# Load LIAR
# --------------------------
try:
    liar = load_dataset("liar")
    liar_train = liar["train"].to_pandas()
    liar_train = liar_train.rename(columns={"statement": "text", "label": "liar_label"})
    liar_train["label"] = liar_train["liar_label"].map(map_liar)
except Exception as e:
    print("⚠️ Could not load LIAR:", e)
    liar_train = pd.DataFrame(columns=["text", "label"])

# --------------------------
# Load multiple NewsAPI CSVs
# --------------------------
news_dir = Path("collectors/csv/newsapi")
news_dfs = []
if news_dir.exists():
    for csv_file in news_dir.glob("*.csv"):
        df = pd.read_csv(csv_file)
        if "label" in df.columns:  # only include if pseudo-labeled
            df = df.rename(columns={"text": "text", "label": "label"})
            df["label"] = df["label"].map(lambda x: FINAL_LABELS.get(x, FINAL_LABELS["UNKNOWN"]))
            news_dfs.append(df)
if news_dfs:
    news_df = pd.concat(news_dfs, ignore_index=True)
else:
    news_df = pd.DataFrame(columns=["text", "label"])

# --------------------------
# Merge datasets
# --------------------------
df = pd.concat(
    [fever_train[["text", "label"]], liar_train[["text", "label"]], news_df[["text", "label"]]],
    ignore_index=True
)
df = df.dropna(subset=["text", "label"])
print(f"✅ Combined dataset size: {len(df)}")

# --------------------------
# Train/Val split
# --------------------------
train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)

# --------------------------
# Hugging Face Dataset
# --------------------------
train_ds = Dataset.from_pandas(train_df)
val_ds = Dataset.from_pandas(val_df)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def preprocess(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=128)

train_ds = train_ds.map(preprocess, batched=True)
val_ds = val_ds.map(preprocess, batched=True)

train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
val_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# --------------------------
# Model + Trainer
# --------------------------
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)

args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,   # DeBERTa is heavy → smaller batch size
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tokenizer
)

trainer.train()
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("Training finished! Model saved at", OUTPUT_DIR)
