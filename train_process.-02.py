from utilities import *
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from transformers import TrainingArguments
from transformers import AutoModelForCausalLM
# from llama import BasicModelRunner


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


# TRAINING ARGUMENTS
# max_steps = 3
# training_args = TrainingArguments(

# # Learning rate
# learning_rate=1.0e-5,

# # Number of training epochs
# num_train_epochs=1,

# # Max steps to train for (each step is a batch of data)
# # Overrides num_train_epochs, if not -1
# max_steps=max_steps,

# # Batch size for training
# per_device_train_batch_size=1,

# # Directory to save model checkpoints
# output_dir=output_dir,

# # Other arguments
# overwrite_output_dir=False, # Overwrite the content of the output directory
# disable_tqdm=False, # Disable progress bars
# eval_steps=120, # Number of update steps between two evaluations
# save_steps=120, # After # steps model is saved
# warmup_steps=1, # Number of warmup steps for learning rate scheduler
# per_device_eval_batch_size=1, # Batch size for evaluation
# evaluation_strategy="steps",
# logging_strategy="steps",
# logging_steps=1,
# optim="adafactor",
# gradient_accumulation_steps = 4,
# gradient_checkpointing=False,

# # Parameters for early stopping

# load_best_model_at_end=True,
# save_total_limit=1,
# metric_for_best_model="eval_loss",
# greater_is_better=False
# )


# MAIN TRAINER

trainer = Trainer(
    model=base_model,
    model_flops=model_flops,
    total_steps=max_steps,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)
trainer.do_grad_scaling = False


# training_output = trainer.train()