from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
from datasets import load_dataset


model_name = "distilbert/distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name,torch_dtype=torch.float16,device_map="auto")

df = load_dataset("mteb/tweet_sentiment_extraction")

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


