from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
import accelerate
from datasets import load_dataset

import logging


model_name = "distilbert/distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name,torch_dtype=torch.float16,device_map="auto")

df = load_dataset("mteb/tweet_sentiment_extraction")
df_path = "mteb/tweet_sentiment_extraction"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(df["train"][0])

device_count = torch.cuda.device_count()
if device_count > 0:
	logger.debug("Select GPU device")
	device = torch.device("cuda")
else:
	logger.debug("Select CPU device")
	device = torch.device("cpu")

use_hf = True
training_config = {
    "model": {"pretrained_model_name":model,"max_length":2048},
    "dataset":{"use_hf" : use_hf,"path":df_path},
    "verbose":True
}

def inference(input_text, model,tokenizer,max_input_length = 2048,max_output_Length = 2048):
    # Tokenizer
    input_ids = tokenizer.encode(input_text, return_tensors="pt",max_length=max_input_length)

    model = model.device()

    # Generate
    generate_tokens_prompt = model.generate(input_ids = input_ids.to(device),max_length = max_output_Length)

    # Decode
    generated_text = tokenizer.batch_decode(generate_tokens_prompt, skip_special_tokens=True)

    # Strip the prompt
    generated_text = generated_text[0][len(input_text):]

    return generated_text
