from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
import accelerate
from datasets import load_dataset 


model_name = "distilbert/distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name,torch_dtype=torch.float16,device_map="auto")

df = load_dataset("mteb/tweet_sentiment_extraction")
df_path = "mteb/tweet_sentiment_extraction"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(df["train"][0])

use_hf = True
training_config = {
    "model": {"pretrained_model_name":model,"max_length":2048},
    "dataset":{"use_hf" : use_hf,"path":df_path},
    "verbose":True

}