# trainer.push_to_hub()


# Load model directly
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import Trainer, TrainingArguments
from sklearn.metrics import accuracy_score,f1_score
from torch.utils.data import Dataset
from typing import Dict , Optional

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


model_path = "google-bert/bert-base-uncased"

finetuned_model = "no"

class IMDbReview:
    def __init__(self):
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = None
        self.label_mapping = {
            "Good" : 0,
            "Bad" : 1
        }
        self.id2label = {v:k for k,v in self.label_mapping.item()}
    
    def load_model(self,model_path = finetuned_model):
        """ Loading Finetuned Model"""
        path = model_path if model_path else self.model_path
        
        self.model = AutoModelForSequenceClassification.from_pretrained(path)
        self.model.eval()

    def preprocess(self,text,max_length=512):
        tokenized_text = self.tokenizer(
            text,
            truncation = True,
            padding = True,
            max_length = max_length,
            return_tensors = "pt"
        ) 

        return tokenized_text
    
    def predict(self,text):
        """ Predicting the value """
        if self.model is None:
            raise ValueError("Model Not loaded")
        
        tokenized_text = self.preprocess(text)

        with torch.no_grad():
            output = self.model(** tokenized_text)

            prediction = torch.nn.functional.softmax(output.logits,dim=-1)

        prediction_class = torch.argmax(prediction,dim=-1).item()
        confidence = prediction[0][prediction_class].item()

        return {"Review" : self.id2label[prediction_class], "confidence" : confidence}
    
class IMDbReviewTrainer:

    def __init__(self):
        self.classifier = IMDbReview()
    
    def compute_metrics(self,eval_pred):
        """ Compute metrics for Evaluation """
        predictions,labels = eval_pred

        predictions = np.argmax(predictions,axis=1)

        accuracy = accuracy_score(labels,predictions)
        f1 = f1_score(labels,predictions,average='weighted')

        return {
            'eval_accuracy': accuracy, 
            'eval_f1': f1
        }
    def train_model(self,train_dataset,test_dataset,output_dir = "../trained_models/IMDb_model"):
        """ Training the model """

        model = AutoModelForSequenceClassification.from_pretrained(
            self.classifier.model_path,
            num_labels = len(self.classifier.label_mapping),
            id2label = self.classifier.id2label,
            label2id = self.classifier.label_mapping
        )

        training_args = TrainingArguments()


    



