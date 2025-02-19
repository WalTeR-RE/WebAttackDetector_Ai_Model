import os
import csv
import pandas as pd
import torch
from transformers import BertTokenizerFast, BertForSequenceClassification, Trainer, TrainingArguments, BertTokenizer
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from Logging.logger import model_log
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import matplotlib.pyplot as plt
import numpy as np
from torch.nn import CrossEntropyLoss
import torch.nn as nn
import ktrain




class WebAttackDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels:
          item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])


class BERT:
    def __init__(self, data_frame=None):
        self.weights = None
        np.set_printoptions(suppress=True)
        self.data = data_frame
        self.model = None
        self.id2label, self.label2id = self.labelization()
        self.category_to_label = None
        self.label_to_category = None
        self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
        # model_log.info(f"id2label: {self.id2label}")
        model_log.info(f"label2id: {self.label2id}")
        self.represent()
        self.test_list = self.get_test_cases()
        #self.train_model()
        # self.create_model()

    def load_model(self):
        model_log.debug("Loading Model")
        self.model = BertForSequenceClassification.from_pretrained("uri_queries_web_attack_detection")
        self.model.to('cuda' if torch.cuda.is_available() else 'cpu')
        model_log.debug("Model Loaded!")

    def predict(self, uri_query, categories):
        self.model.eval()
        inputs = self.tokenizer(uri_query, padding=True, truncation=True, return_tensors='pt').to('cuda')
        outputs = self.model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predictions = predictions.cpu().detach().numpy()
        identification_indx = np.argmax(predictions)
        model_log.debug(
                f"Test: {uri_query}\nPrediction: {predictions}\nMax_Indx: {identification_indx}\nClassification: {categories[identification_indx]}")

    def custom_loss_fn(self,outputs, labels):
        loss_fct = CrossEntropyLoss(weight=self.weights)
        return loss_fct(outputs.logits, labels)
    def train_model(self):
        model_log.debug("Training Model")
        classes = list(set(self.data.category))
        X_train, X_val, y_train, y_val = train_test_split(list(self.data["UriQuery"]),
                                                          list(self.data["categories_label"]), test_size=0.2,
                                                          stratify=self.data["categories_label"])


        tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
        #tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
        class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        self.weights = torch.tensor(class_weights, dtype=torch.float).to('cuda')  # Move to GPU if available
        self.model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=len(classes))
        self.model.to('cuda')
        X_train_tokenized = tokenizer(X_train, truncation=True, padding=True, max_length=512)
        X_val_tokineized = tokenizer(X_val, truncation=True, padding=True, max_length=512)
        train_dataset = WebAttackDataset(X_train_tokenized, y_train)
        val_dataset = WebAttackDataset(X_val_tokineized, y_val)
        training_args = TrainingArguments(
            output_dir="./results",
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=6,
            per_device_eval_batch_size=6,
            num_train_epochs=3,
            weight_decay=0.01,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
        )

        class CustomTrainer(Trainer):
            def __init__(self, *args, class_weights=None, **kwargs):
                super().__init__(*args, **kwargs)
                self.criterion = nn.CrossEntropyLoss(weight=class_weights)

            def compute_loss(self, model, inputs, return_outputs=False):
                labels = inputs.pop("labels")  # Extract labels
                outputs = model(**inputs)  # Forward pass
                logits = outputs.logits  # Model predictions

                loss = self.criterion(logits, labels)  # Weighted loss
                return (loss, outputs) if return_outputs else loss

        trainer = CustomTrainer(
            model=self.model,
            args=training_args,  # Your TrainingArguments
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            class_weights=self.weights,  # Pass weights here
            compute_metrics=self.compute_metric  # Optional: Your evaluation metrics
        )
        trainer.train()
        """
        training_args = TrainingArguments(
        output_dir="./Results",
        num_train_epochs=2,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_dir="./Logs",
        logging_steps=200,
        load_best_model_at_end=True,
        metric_for_best_model="eval_Accuracy",
        # Optional: To improve checkpoint saving
        save_total_limit=3,

        weight_decay=0.01,
        warmup_steps=500,
        fp16=True,
        seed=42,
        )

        self.load_model_for_train()

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self.compute_metric

        )"""
        list_of_xss = []
        list_of_sqli = []
        list_of_rce = []
        list_of_rfi = []
        list_of_lfi = []
        list_of_ben = []
        model_log.debug("Training Model")
        trainer.train()
        model_log.debug("Evaluating Model")
        trainer.evaluate()
        model_log.debug("Saving Model")
        trainer.save_model('uri_queries_web_attack_detection')
        model_log.info("Testing Model")
        for test in self.test_list:
            inputs = tokenizer(test,padding=True,truncation=True,return_tensors='pt').to('cuda')
            outputs = self.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predictions = predictions.cpu().detach().numpy()
            identification_indx = np.argmax(predictions)
            if self.category_to_label[identification_indx] == "XSS":
                list_of_xss.append(test)
            elif self.category_to_label[identification_indx] == "SQLi":
                list_of_sqli.append(test)
            elif self.category_to_label[identification_indx] == "RCE":
                list_of_rce.append(test)
            elif self.category_to_label[identification_indx] == "RFI":
                list_of_rfi.append(test)
            elif self.category_to_label[identification_indx] == "LFI":
                list_of_lfi.append(test)
            elif self.category_to_label[identification_indx] == "Benign":
                list_of_ben.append(test)

            model_log.debug(f"Test: {test}\nPrediction: {predictions}\nMax_Indx: {identification_indx}\nClassification: {self.category_to_label[identification_indx]}")


        model_log.debug("Model Trained")
        model_log.debug(f"Final Res:\n",
                        f"XSS: {len(list_of_xss)} out of 3104\n",
                        f"SQLI:{len(list_of_sqli)} out of 2002\n",
                        f"RFI: {len(list_of_rfi)} out of 2001\n",
                        f"RCE: {len(list_of_rce)} out of 491\n",
                        f"LFI: {len(list_of_lfi)} out of 2220\n",
                        f"Benign: {len(list_of_ben)} out of 11051\n")

    def compute_metric(self, p):
        print(type(p))
        pred, labels = p
        pred = np.argmax(pred, axis=1)

        accuracy = accuracy_score(y_true=labels, y_pred=pred)
        recall = recall_score(y_true=labels, y_pred=pred,average='weighted')
        precision = precision_score(y_true=labels, y_pred=pred,average='weighted')
        f1 = f1_score(y_true=labels, y_pred=pred,average='weighted')

        return {"Accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

    def labelization(self):
        labels = self.data['category'].unique().tolist()
        labels = [s.strip() for s in labels]
        return {id: label for id, label in enumerate(labels)}, {label: id for id, label in enumerate(labels)}


    def get_test_cases(self):
        for filename in os.listdir("./Repository/test"):
            if filename.endswith(".csv"):
                path = os.path.join("./Repository/test",filename)
                array_name = os.path.splitext(filename)[0] + "_samples"
                array_name = array_name.lower()
                with open(path, mode='r', encoding='latin-1') as file:
                    reader = csv.reader(file)
                    flattened_data = [item for row in reader for item in row]
                    globals()[array_name] = flattened_data
                    print(f"{array_name}: {len(globals()[array_name])}")

        return globals()['lfi_samples']+globals()['benign_samples']+globals()['rfi_samples']+globals()['sqli_samples']+globals()['rce_samples']+globals()['xss_samples']


    def represent(self):
        self.data["categories_label"],category_mapping = pd.factorize(self.data.category)
        self.category_to_label = dict(enumerate(category_mapping))
        self.label_to_category = {v: k for k, v in self.category_to_label.items()}
        model_log.info(self.label_to_category)
        model_log.warning(f"Testing Values: {self.category_to_label[0]},{self.category_to_label[1]},{self.category_to_label[2]}")

    def load_model_for_train(self):
        self.model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(self.label2id))

    def viewGraph(self):
        self.data['category'].value_counts().plot(
            kind='pie',
            figsize=(8, 8),
            autopct=lambda p: f'{p:.2f}%'
        )

        plt.ylabel('')
        plt.title('Category Distribution')
        plt.show()
