# train_binary.py

import torch
from datasets import load_dataset
from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from sklearn.metrics import accuracy_score, f1_score
import torch.nn as nn
from pathlib import Path

# -----------------
# Binary label mapping
# -----------------
label_map = {
    "pants-fire": "fake",
    "false": "fake",
    "barely-true": "fake",
    "half-true": "fake",
    "mostly-true": "true",
    "true": "true"
}

labels = ["fake", "true"]
label2id = {l: i for i, l in enumerate(labels)}
id2label = {i: l for l, i in label2id.items()}

columns = [
    'label', 'statement', 'subject', 'speaker', 'speaker_job', 'state_info',
    'party_affiliation', 'barely_true_counts', 'false_counts',
    'half_true_counts', 'mostly_true_counts', 'pants_on_fire_counts', 'context'
]

# -----------------
# Dataset loader
# -----------------
dataset = load_dataset(
    'csv',
    data_files={"train": "train.tsv", "validation": "valid.tsv", "test": "test.tsv"},
    delimiter="\t",
    column_names=columns
)

def preprocess_labels(example):
    original = example["label"].strip().lower()
    if original in label_map:
        mapped = label_map[original]
        example["labels"] = label2id[mapped]
    else:
        example["labels"] = -1
    return example

dataset = dataset.map(preprocess_labels)
dataset = dataset.filter(lambda x: x["labels"] != -1)

# -----------------
# Tokenizer + text building
# -----------------
tokenizer = RobertaTokenizer.from_pretrained("roberta-large")

def build_text(example):
    extra = f"[SUBJECT] {example['subject']} [SPEAKER] {example['speaker']} [PARTY] {example['party_affiliation']} [CONTEXT] {example['context']} [JOB] {example['speaker_job']}"
    return {"text": example["statement"] + " " + extra}

dataset = dataset.map(build_text)

# Tokenize without padding (padding handled dynamically)
dataset = dataset.map(
    lambda batch: tokenizer(batch["text"], truncation=True, max_length=256),
    batched=True
)

# Use DataCollatorWithPadding to dynamically pad batches
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# -----------------
# Custom Trainer with Focal Loss
# -----------------
class TrainerWithFocalLoss(Trainer):
    def __init__(self, gamma=2.0, *args, **kwargs):
        kwargs.pop("tokenizer", None)  # remove deprecated tokenizer
        super().__init__(*args, **kwargs)
        self.gamma = gamma

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        device = next(model.parameters()).device  # ✅ works with DataParallel
        labels = inputs.pop("labels").to(device)
        outputs = model(**inputs)
        logits = outputs.logits
        ce_loss = nn.CrossEntropyLoss(reduction="none")(logits, labels)
        pt = torch.exp(-ce_loss)
        loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return (loss, outputs) if return_outputs else loss

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
# Training
# -----------------
def train_model():
    model = RobertaForSequenceClassification.from_pretrained(
        "roberta-large",
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id
    )

    output_dir = Path("./roberta-fake-news-binary")
    output_dir.mkdir(exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-6,
        num_train_epochs=10,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        logging_dir="./logs-binary",
        logging_steps=50,
        report_to="none",
        load_best_model_at_end=True,
        fp16=torch.cuda.is_available(),
    )

    trainer = TrainerWithFocalLoss(
        gamma=2.0,
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=data_collator,  # ✅ dynamic padding
        compute_metrics=compute_metrics,
    )

    # Train
    trainer.train()

    # Evaluate
    results = trainer.evaluate(dataset["test"])
    print("Final Test Results (Binary):", results)

    # Save final model (pipeline-ready)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"✅ Model ready for Hugging Face pipeline at {output_dir}")

if __name__ == "__main__":
    train_model()
