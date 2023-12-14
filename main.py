import os
import logging
import torch
import numpy as np
import optuna
from typing import Tuple
from transformers import BertForSequenceClassification, BertTokenizer, AdamW
from datasets import load_dataset
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BertClassifier:
    """
    BERT Classifier for sequence classification tasks.
    """
    def __init__(self, model_name: str = 'bert-base-uncased', device: str = 'mps'):
        self.model, self.tokenizer = self.load_model_and_tokenizer(model_name)
        self.device = self.set_device(device)
        self.model.to(self.device)

    @staticmethod
    def load_model_and_tokenizer(model_name: str) -> Tuple[BertForSequenceClassification, BertTokenizer]:
        """
        Load the model and tokenizer from the pretrained model name.
        """
        try:
            model = BertForSequenceClassification.from_pretrained(model_name)
            tokenizer = BertTokenizer.from_pretrained(model_name)
            return model, tokenizer
        except Exception as e:
            logger.error(f"Error loading model and tokenizer: {e}")
            raise

    @staticmethod
    def set_device(device: str):
        """
        Set the device for model training and inference.
        """
        if device == "cuda" and torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info("Running on the GPU")
        elif device == "mps" and torch.backends.mps.is_available():
            device = torch.device("mps")
            logger.info("Running on the MPS device")
        else:
            device = torch.device("cpu")
            logger.info("Running on the CPU")
        return device

    def load_and_preprocess_data(self, dataset_name: str = "glue", task_name: str = "sst2"):
        """
        Load and preprocess the data from the given dataset name and task name.
        """
        try:
            raw_datasets = load_dataset(dataset_name, task_name)
            def tokenize_function(examples):
                return self.tokenizer(examples["sentence"], padding="max_length", truncation=True)
            tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
            tokenized_datasets.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'], device=self.device)
            return tokenized_datasets
        except Exception as e:
            logger.error(f"Error loading and preprocessing data: {e}")
            raise

    def train_model(self, tokenized_datasets, trial: optuna.Trial):
        """
        Train the model with the given tokenized datasets and trial.
        """
        try:
            batch_size = trial.suggest_categorical('batch_size', [8, 16, 32])
            learning_rate = trial.suggest_loguniform('lr', 1e-5, 1e-3)
            epochs = trial.suggest_int('epochs', 1, 5)
            gradient_accumulation_steps = trial.suggest_int('gradient_accumulation_steps', 1, 5)

            train_dataloader = DataLoader(tokenized_datasets['train'], batch_size=batch_size, shuffle=True)
            optimizer = AdamW(self.model.parameters(), lr=learning_rate)

            for epoch in range(epochs):
                logger.info(f"Starting epoch {epoch+1}/{epochs}")
                for step, batch in enumerate(train_dataloader):
                    self.model.train()
                    inputs = {key: value for key, value in batch.items() if key != 'label'}
                    labels = batch['label']
                    outputs = self.model(**inputs, labels=labels)
                    loss = outputs.loss
                    loss = loss / gradient_accumulation_steps
                    loss.backward()
                    if (step + 1) % gradient_accumulation_steps == 0:
                        optimizer.step()
                        optimizer.zero_grad()
                logger.info(f"Finished epoch {epoch+1}/{epochs}")
                self.save_model(f'./model_epoch_{epoch+1}')

            return self.evaluate_model(tokenized_datasets, batch_size)
        except Exception as e:
            logger.error(f"Error during training: {e}")
            raise

    def save_model(self, path: str = './model'):
        """
        Save the model and tokenizer to the given path.
        """
        try:
            os.makedirs(path, exist_ok=True)
            self.model.save_pretrained(path)
            self.tokenizer.save_pretrained(path)
            logger.info(f"Model saved in {path}")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise

    def load_model(self, path: str = './model'):
        """
        Load the model and tokenizer from the given path.
        """
        try:
            self.model = BertForSequenceClassification.from_pretrained(path)
            self.tokenizer = BertTokenizer.from_pretrained(path)
            logger.info(f"Model loaded from {path}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def evaluate_model(self, tokenized_datasets, batch_size: int = 8):
        """
        Evaluate the model with the given tokenized datasets and batch size.
        """
        test_dataloader = DataLoader(tokenized_datasets['test'], batch_size=batch_size)
        predictions, true_labels = [], []
        self.model.eval()
        for batch in test_dataloader:
            with torch.no_grad():
                inputs = {key: value for key, value in batch.items() if key != 'labels'}
                labels = batch['label']
                outputs = self.model(**inputs, labels=labels)
            logits = outputs.logits.detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()
            predictions.extend(np.argmax(logits, axis=1))
            true_labels.extend(labels)
        accuracy = accuracy_score(true_labels, predictions)
        precision = precision_score(true_labels, predictions, average='weighted')
        recall = recall_score(true_labels, predictions, average='weighted')
        f1 = f1_score(true_labels, predictions, average='weighted')
        logger.info(f'Test Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1: {f1}')
        logger.info(classification_report(true_labels, predictions))
        return accuracy, precision, recall, f1

def main():
    """
    Main function to run the BERT Classifier.
    """
    classifier = BertClassifier()
    tokenized_datasets = classifier.load_and_preprocess_data()

    def objective(trial):
        return classifier.train_model(tokenized_datasets, trial)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50)

    classifier.evaluate_model(tokenized_datasets)

if __name__ == "__main__":
    main()