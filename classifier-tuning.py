from datasets import load_dataset
from transformers import AutoTokenizer
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

dataset = load_dataset("imdb")


tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)


labels = dataset["train"]["label"]
class_weights = compute_class_weight("balanced", classes=np.unique(labels), y=labels)
print(class_weights)

