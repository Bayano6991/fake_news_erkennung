import torch
from torch import nn
from datasets import load_dataset
from transformers import RobertaTokenizer, RobertaModel, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score

# -----------------
# Label Mapping
# -----------------
label2id = {'fake': 0, 'misleading': 1, 'partially true': 2, 'mostly true': 3, 'true': 4}
id2label = {v: k for k, v in label2id.items()}

original_to_custom = {
    "pants-fire": "fake",
    "false": "fake",
    "barely-true": "misleading",
    "half-true": "partially true",
    "mostly-true": "mostly true",
    "true": "true"
}

# -----------------
# Load Dataset
# -----------------
columns = [
    'label', 'statement', 'subject', 'speaker', 'speaker_job', 'state_info',
    'party_affiliation', 'barely_true_counts', 'false_counts',
    'half_true_counts', 'mostly_true_counts', 'pants_on_fire_counts', 'context'
]

dataset = load_dataset(
    'csv',
    data_files={
        "train": "train.tsv",
        "validation": "valid.tsv",
        "test": "test.tsv"
    },
    delimiter='\t',
    column_names=columns
)

def fix_labels(example):
    original = example["label"].strip().lower()
    mapped = original_to_custom.get(original)
    example["labels"] = label2id[mapped] if mapped else -1
    return example

dataset = dataset.map(fix_labels)
dataset = dataset.filter(lambda x: x["labels"] != -1)

# Build text field
def build_text(example):
    extra = f"[SUBJECT] {example['subject']} [SPEAKER] {example['speaker']} [PARTY] {example['party_affiliation']} [CONTEXT] {example['context']} [JOB] {example['speaker_job']}"
    return {"text": example["statement"] + " " + extra}

dataset = dataset.map(build_text)

# -----------------
# Tokenizer
# -----------------
tokenizer = RobertaTokenizer.from_pretrained("roberta-large")

def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True, max_length=256)

dataset = dataset.map(tokenize, batched=True)
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

train_dataset = dataset["train"]
val_dataset = dataset["validation"]
test_dataset = dataset["test"]

# -----------------
# Model with Focal Loss
# -----------------
class RobertaWithFocalLoss(nn.Module):
    def __init__(self, num_labels, gamma=2.0):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained("roberta-large")
        self.classifier = nn.Linear(self.roberta.config.hidden_size, num_labels)
        self.gamma = gamma

    def focal_loss(self, logits, labels):
        ce_loss = nn.CrossEntropyLoss(reduction="none")(logits, labels)
        pt = torch.exp(-ce_loss)
        return ((1 - pt) ** self.gamma * ce_loss).mean()

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0]  # CLS token
        logits = self.classifier(pooled)
        loss = None
        if labels is not None:
            loss = self.focal_loss(logits, labels)
        return {"loss": loss, "logits": logits}

model = RobertaWithFocalLoss(num_labels=len(label2id))

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
training_args = TrainingArguments(
    output_dir="./roberta-fake-news",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-6,
    num_train_epochs=10,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    logging_dir="./logs",
    logging_steps=50,
    report_to="none",
    load_best_model_at_end=True,
    fp16=torch.cuda.is_available()
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()

print("Evaluating on Test set...")
results = trainer.evaluate(test_dataset)
print("Final Test Results:", results)
