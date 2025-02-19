import os
import csv
import pandas as pd
import torch
import numpy as np
from torch.nn import CrossEntropyLoss
import torch.nn as nn
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments
)
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from typing import Dict, List, Optional, Tuple, Any
from Logging.logger import model_log


class WebAttackDataset(torch.utils.data.Dataset):
    def __init__(self, encodings: Dict[str, List], labels: Optional[List[int]] = None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self) -> int:
        return len(self.encodings["input_ids"])


class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        self.class_weights = kwargs.pop('class_weights', None)
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        model_inputs = {k: v for k, v in inputs.items() if k != "labels"}

        outputs = model(**model_inputs)
        logits = outputs.logits

        if self.class_weights is not None:
            loss_fct = CrossEntropyLoss(weight=self.class_weights)
        else:
            loss_fct = CrossEntropyLoss()

        loss = loss_fct(logits.view(-1, logits.shape[-1]), labels.view(-1))

        return (loss, outputs) if return_outputs else loss


class WebAttackDetector:
    def __init__(self, data_frame: Optional[pd.DataFrame] = None):
        model_log.info("Initializing WebAttackDetector")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.data = data_frame
        self.model = None
        self.tokenizer = None
        self.weights = None
        self.category_mappings = {}
        self.test_results = {}

        if data_frame is not None:
            self._initialize_mappings()
            self._initialize_model()

    def _initialize_mappings(self) -> None:
        model_log.debug("Initializing category mappings")
        unique_categories = self.data['category'].unique()
        self.category_mappings = {
            'id2label': {id: label for id, label in enumerate(unique_categories)},
            'label2id': {label: id for id, label in enumerate(unique_categories)},
            'category_to_label': None,
            'label_to_category': None
        }
        model_log.info(f"label2id: {self.category_mappings['label2id']}")

    def _initialize_model(self) -> None:
        model_log.debug("Initializing tokenizer and model")
        self.tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
        num_labels = len(self.category_mappings['id2label'])
        self.model = DistilBertForSequenceClassification.from_pretrained(
            'distilbert-base-uncased',
            num_labels=num_labels
        ).to(self.device)

    def prepare_data(self) -> Tuple[WebAttackDataset, WebAttackDataset]:
        model_log.debug("Preparing training and validation datasets")

        self.data['categories_label'], category_mapping = pd.factorize(self.data.category)
        self.category_mappings['category_to_label'] = dict(enumerate(category_mapping))
        self.category_mappings['label_to_category'] = {v: k for k, v in
                                                       self.category_mappings['category_to_label'].items()}

        X_train, X_val, y_train, y_val = train_test_split(
            list(self.data["UriQuery"]),
            list(self.data["categories_label"]),
            test_size=0.2,
            stratify=self.data["categories_label"]
        )

        unique_classes = np.unique(y_train)
        class_weights = class_weight.compute_class_weight(
            'balanced',
            classes=unique_classes,
            y=y_train
        )

        self.weights = torch.FloatTensor(class_weights).to(self.device)

        train_encodings = self.tokenizer(
            X_train,
            truncation=True,
            padding=True,
            max_length=512
        )
        val_encodings = self.tokenizer(
            X_val,
            truncation=True,
            padding=True,
            max_length=512
        )

        return (
            WebAttackDataset(train_encodings, y_train),
            WebAttackDataset(val_encodings, y_val)
        )

    def train(self, output_dir: str = "results") -> None:
        model_log.debug("Starting model training")

        os.makedirs(output_dir, exist_ok=True)

        train_dataset, val_dataset = self.prepare_data()

        training_args = TrainingArguments(
            output_dir=output_dir,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=6,
            per_device_eval_batch_size=6,
            num_train_epochs=3,
            weight_decay=0.01,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            logging_dir=os.path.join(output_dir, "logs"),
            logging_steps=100,
            save_total_limit=2
        )

        trainer = CustomTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            class_weights=self.weights,
            compute_metrics=self._compute_metrics
        )

        try:
            model_log.debug("Training model")
            trainer.train()

            model_log.debug("Evaluating model")
            metrics = trainer.evaluate()
            model_log.info(f"Evaluation metrics: {metrics}")

            model_log.debug("Saving model")
            save_path = os.path.join(output_dir, "final_model")
            os.makedirs(save_path, exist_ok=True)
            trainer.save_model(save_path)

            self._evaluate_test_cases()

        except Exception as e:
            model_log.error(f"Error during training: {str(e)}")
            raise

    def get_test_cases(self) -> List[str]:
        model_log.debug("Loading test cases")
        test_cases = []
        test_dir = os.path.join("Repository", "test")

        try:
            for filename in os.listdir(test_dir):
                if filename.endswith(".csv"):
                    path = os.path.join(test_dir, filename)
                    category = os.path.splitext(filename)[0]

                    try:
                        with open(path, mode='r', encoding='latin-1') as file:
                            reader = csv.reader(file)
                            data = [item for row in reader for item in row]
                            test_cases.extend(data)
                            self.test_results[category] = len(data)
                            model_log.debug(f"{category}: {len(data)} samples")
                    except Exception as e:
                        model_log.error(f"Error reading file {filename}: {str(e)}")

        except Exception as e:
            model_log.error(f"Error accessing test directory: {str(e)}")

        return test_cases

    def _evaluate_test_cases(self):
        test_cases = self.get_test_cases()
        predictions = {}
        model_log.info("Evaluate Test Cases, Please Wait.....")
        for test in test_cases:
            category, _ = self.predict(test)
            predictions.setdefault(category, []).append(test)

        model_log.info("Test Results Summary:")
        for category, cases in predictions.items():
            model_log.info(f"{category}: {len(cases)} predictions")

    def load_model(self) -> None:
        model_log.debug("Initializing tokenizer and model")
        self.tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
        num_labels = len(self.category_mappings['id2label'])
        self.model = DistilBertForSequenceClassification.from_pretrained(
            'uri_queries_web_attack_detection-V2.0',
            num_labels=num_labels
        ).to(self.device)

    def predict(self, uri_query: str,categories=None) -> Tuple[str, np.ndarray]:
        if categories:
            self.category_mappings['category_to_label'] = categories
        self.model.eval()
        inputs = self.tokenizer(
            uri_query,
            padding=True,
            truncation=True,
            return_tensors='pt'
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predictions = predictions.cpu().numpy()

        category_idx = np.argmax(predictions[0])
        #print(self.category_mappings['category_to_label'])
        category = self.category_mappings['category_to_label'][category_idx]
        #print("Here")
        return category, predictions[0]

    @staticmethod
    def _compute_metrics(pred_obj: Any) -> Dict[str, float]:
        predictions, labels = pred_obj
        predictions = np.argmax(predictions, axis=1)

        return {
            "accuracy": accuracy_score(labels, predictions),
            "precision": precision_score(labels, predictions, average='weighted'),
            "recall": recall_score(labels, predictions, average='weighted'),
            "f1": f1_score(labels, predictions, average='weighted')
        }
