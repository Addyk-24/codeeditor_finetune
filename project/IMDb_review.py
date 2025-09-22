
# Load model directly
from transformers import AutoTokenizer, AutoModelForSequenceClassification


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
    
    


