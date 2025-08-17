from transformers import AutoTokenizer, AutoModelForMaskedLM, TrainingArguments,Trainer
import torch
import accelerate

from peft import LoraConfig, get_peft_model, PeftModel

from datasets import load_dataset

import logging

logger = logging.getLogger(__name__)
global_config = None


model_name = "distilbert/distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name,torch_dtype=torch.float16,device_map="auto")

df = load_dataset("mteb/tweet_sentiment_extraction")
df_path = "mteb/tweet_sentiment_extraction"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(df["train"][0])

train_dataset = df["train"]
test_dataset = df["test"]

device_count = torch.cuda.device_count()
if device_count > 0:
	logger.debug("Select GPU device")
	device = torch.device("cuda")
else:
	logger.debug("Select CPU device")
	device = torch.device("cpu")

use_hf = True
training_config = {
    "model": {"pretrained_model_name":model_name,"max_length":2048},
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

max_steps = 3
output_dir = "output"

training_args = TrainingArguments(
learning_rate=1.0e-5,
num_train_epochs=1,
max_steps=max_steps,
per_device_train_batch_size=1,
output_dir=output_dir,

)

model_flops = (
	model.floating_point_ops(
	{
	"input_ids": torch.zeros(
	(1, training_config["model"]["max_length"])
	)
	}
	)
	  * training_args.gradient_accumulation_steps
)

# print("Memory footprint", model.get_memory_footprint() / 1e9, "GB")
# print("Flops", model_flops / 1e9, "GFLOPs")

trainer= SFTTrainer(
    model=model,
    args=training_arguments,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    processing_class=tokenizer,
    peft_config=peft_config,
)
trainer.do_grad_scaling = False

training_output = trainer.train()

print("Training completed.")
trainer.save_model()