
# Load model directly
from transformers import AutoTokenizer, AutoModelForSequenceClassification


model_path = "google-bert/bert-base-uncased"

class IMDbReview:
    def __init__(self):
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
         