import torch
from datasets import load_dataset

from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification


print(torch.cuda.is_available()) 


# Load the IMDB dataset
dataset = load_dataset("imdb")

# Peek at the dataset structure
print("This is Dataset: ",dataset)


# Initialize the BERT tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Tokenize a sample text
sample_text = "I absolutely loved this movie! Highly recommend it."
tokens = tokenizer(sample_text, padding="max_length", truncation=True, max_length=128)

print("Tokenized Text: ",tokens)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

# Apply the tokenizer to the dataset
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Inspect tokenized samples
print(tokenized_datasets["train"][0])


from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# Class imbalance: Use oversampling or weighted loss functions to balance your dataset. You can calculate class weights with scikit-learn like this:

labels = dataset["train"]["label"]
class_weights = compute_class_weight("balanced", classes=np.unique(labels), y=labels)
print("Class-Weights: ",class_weights)


# Initialize a BERT model for binary classification
model_name = "bert-base-uncased"
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

print(model.config)