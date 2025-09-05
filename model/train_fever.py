# train_fever_fast.py

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score

# -----------------
# Load FEVER v1.0
# -----------------
dataset = load_dataset("fever", "v1.0")

valid_labels = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]
label_map = {"SUPPORTS": 1, "REFUTES": 0, "NOT ENOUGH INFO": 2}

# -----------------
# Create train/validation splits
# -----------------
split = dataset["train"].train_test_split(test_size=0.1, seed=42)
train_dataset = split["train"].shuffle(seed=42).select(range(10000))   # subset
val_dataset = split["test"].shuffle(seed=42).select(range(2000))       # subset

# -----------------
# Preprocess: filter labels + keep claim
# -----------------
def preprocess(example):
    if example["label"] not in valid_labels:
        return None
    return {
        "labels": label_map[example["label"]],
        "input_text": example["claim"]
    }

train_dataset = train_dataset.map(preprocess, remove_columns=train_dataset.column_names)
train_dataset = train_dataset.filter(lambda x: x is not None)

val_dataset = val_dataset.map(preprocess, remove_columns=val_dataset.column_names)
val_dataset = val_dataset.filter(lambda x: x is not None)

# -----------------
# Tokenization
# -----------------
model_name = "microsoft/deberta-v3-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize(batch):
    return tokenizer(batch["input_text"], padding="max_length", truncation=True, max_length=128)

train_dataset = train_dataset.map(tokenize, batched=True)
val_dataset = val_dataset.map(tokenize, batched=True)

train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# -----------------
# Model
# -----------------
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

# -----------------
# Metrics
# -----------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro")
    }

# -----------------
# Training arguments
# -----------------
training_args = TrainingArguments(
    output_dir="./deberta-fever-fast",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    num_train_epochs=1,                     # just 1 epoch for speed
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=1,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=100,
    load_best_model_at_end=True,
    fp16=torch.cuda.is_available(),
    report_to="none"
)

# -----------------
# Trainer
# -----------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# -----------------
# Train and save
# -----------------
trainer.train()
trainer.save_model("./deberta-fever-fast")
tokenizer.save_pretrained("./deberta-fever-fast")

print("✅ Fast training finished. Model saved to ./deberta-fever-fast")
