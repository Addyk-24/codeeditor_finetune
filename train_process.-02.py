from utilities import *
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from transformers import TrainingArguments
from transformers import AutoModelForCausalLM
from llama import BasicModelRunner


import time
import torch
import transformers
import pandas as pd
import jsonlines

dataset_name = "lamini_docs.jsonl"
dataset_path = f"/content/{dataset_name}"
use_hf = False

model_name = "EleutherAI/pythia-70m" 

dataset_path = "lamini/lamini_docs"
use_hf = True

logger = logging.getLogger(__name__)
global_config = None

training_config = {
"model": {
"pretrained_name": model_name,
"max_length" : 2048
},
"datasets": {
"use_hf": use_hf,
"path": dataset_path
},
"verbose": True
}

base_model = AutoModelForCausalLM.from_pretrained(model_name)

device_count = torch.cuda.device_count()

if device_count > 0:
    logger.debug("Select GPU device")
    device = torch.device("cuda")
else:
    logger.debug("Select CPU device")
    device = torch.device("cpu")

# base_model.to(device)


# def inference(text, model, tokenizer, max_input_tokens=1000, max_output_tokens=100):
# 	#Tokenize
# 	input_ids = tokenizer.encode(
# 	text,
# 	return_tensors="pt",
# 	truncation=True,
# 	max_length=max_input_tokens
# 	)
#     # Generate
#     device = model.device
#     generated_tokens_with_prompt = model.generate(
#     input_ids=input_ids.to(device),
#     max_length=max_output_tokens
#   )
#    # Decode
#   generated_text_with_prompt = tokenizer.batch_decode(generated_tokens_with_prompt, skip_special_tokens=True)

#   # Strip the prompt
#   generated_text_answer = generated_text_with_prompt[0][len(text):]

#   return generated_text_answer