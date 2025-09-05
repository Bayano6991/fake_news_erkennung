# train_distil.py
import torch
from torch import nn
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from datasets import load_dataset, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# ---------------------------
# Label mapping (binary)
# ---------------------------
label_map = {
    "pants-fire": "FAKE",
    "false": "FAKE",
    "barely-true": "FAKE",
    "half-true": "FAKE",
    "mostly-true": "TRUE",
    "true": "TRUE"
}
labels = ["FAKE", "TRUE"]
label2id = {l: i for i, l in enumerate(labels)}
id2label = {i: l for l, i in label2id.items()}

# ---------------------------
# Load dataset
# ---------------------------
print("Loading LIAR dataset...")
ds = load_dataset("liar", split="train")  # only train split, will split manually

# Map labels
def preprocess(example):
    original = example["label"].strip().lower()
    mapped = label_map.get(original, "FAKE")
    return {"labels": label2id[mapped], "text": example["statement"]}

ds = ds.map(preprocess)

# Convert to pandas for train_test_split
df = ds.to_pandas()
train_val_df, test_df = train_test_split(df, test_size=0.1, random_state=42)
train_df, val_df = train_test_split(train_val_df, test_size=0.1111, random_state=42)

# Convert back to HF Dataset
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)
test_dataset = Dataset.from_pandas(test_df)

print(f"Dataset sizes -> Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

# ---------------------------
# Tokenizer
# ---------------------------
tokenizer = DistilBertTokenizerFast.from_pretrained("distilroberta-base")

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=256)

train_dataset = train_dataset.map(tokenize, batched=True)
val_dataset = val_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.map(tokenize, batched=True)

columns_to_return = ["input_ids", "attention_mask", "labels"]
train_dataset.set_format(type="torch", columns=columns_to_return)
val_dataset.set_format(type="torch", columns=columns_to_return)
test_dataset.set_format(type="torch", columns=columns_to_return)

# ---------------------------
# Metrics
# ---------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro")
    }

# ---------------------------
# Model
# ---------------------------
model = DistilBertForSequenceClassification.from_pretrained(
    "distilroberta-base",
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id
)

# ---------------------------
# Trainer
# ---------------------------
training_args = TrainingArguments(
    output_dir="./distil-liar-model",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=100,
    report_to="none",  # no wandb
    load_best_model_at_end=True,
    fp16=torch.cuda.is_available(),
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# ---------------------------
# Train
# ---------------------------
trainer.train()

# Evaluate on test
results = trainer.evaluate(test_dataset)
print("Final Test Results:", results)

# Save model
trainer.save_model("./distil-liar-model")
tokenizer.save_pretrained("./distil-liar-model")
print("✅ Model and tokenizer saved to ./distil-liar-model")
