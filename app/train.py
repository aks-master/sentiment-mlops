import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from transformers import DataCollatorWithPadding
import mlflow
import torch

# Set up MLFlow
mlflow.set_tracking_uri("http://127.0.0.1:5000")  # Local MLflow server
mlflow.set_experiment("sentiment-analysis")

# Load the dataset
def load_dataset(file_path):
    df = pd.read_csv(file_path)
    return df["text"].tolist(), df["label"].tolist()

# Tokenize the dataset
def tokenize_data(texts, labels, tokenizer):
    tokenized_data = tokenizer(texts, truncation=True, padding=True, max_length=128)
    tokenized_data["labels"] = labels
    return tokenized_data

# Dataset class for PyTorch
class SentimentDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        return item

# Training function
def train_model():
    # Load pre-trained model and tokenizer
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    # Load and split dataset
    dataset_path = os.path.join("data", "sample.csv")
    texts, labels = load_dataset(dataset_path)
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )

    # Tokenize datasets
    train_encodings = tokenize_data(train_texts, train_labels, tokenizer)
    val_encodings = tokenize_data(val_texts, val_labels, tokenizer)

    train_dataset = SentimentDataset(train_encodings)
    val_dataset = SentimentDataset(val_encodings)

    # Data collator for padding
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="app/models",      # Directory to save model
        eval_strategy="epoch", # Evaluate at the end of each epoch
        save_strategy="epoch",       # Save model at the end of each epoch
        logging_dir="logs",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=2,
        weight_decay=0.01,
        save_total_limit=1,          # Keep only the last checkpoint
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Train and evaluate model
    trainer.train()
    results = trainer.evaluate()

    # Log model and metrics to MLflow
    with mlflow.start_run():
        mlflow.log_metric("accuracy", results["eval_accuracy"])
        mlflow.log_metric("f1_score", results["eval_f1"])
        mlflow.log_artifacts("app/models")
        mlflow.pytorch.log_model(model, "model")

    print("Training complete. Model saved and logged to MLflow.")

# Main entry point
if __name__ == "__main__":
    train_model()