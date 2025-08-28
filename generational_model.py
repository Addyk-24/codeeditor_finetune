from transformers import AutoTokenizer,AutoModelForSequenceClassification, AutoModelForMaskedLM, TrainingArguments,Trainer, AutoProcessor
import torch
import accelerate


# from peft import LoraConfig, get_peft_model, PeftModel

from datasets import load_dataset

import logging

logger = logging.getLogger(__name__)
global_config = None


# model_name = "google-bert/bert-base-uncased"
# model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
model_name = "bert-base-uncased"


tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name,torch_dtype=torch.float16,device_map="auto")

# processor = AutoProcessor.from_pretrained(model_name)

df = load_dataset("mteb/tweet_sentiment_extraction")
df_path = "mteb/tweet_sentiment_extraction"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# print(df["train"][0])

train_dataset = df["train"]
test_dataset = df["test"]


# def tokenization(text):
#             train_tokenized_text = tokenizer(text["train"],padding=True)["input_ids"]
#             test_tokenized_text = tokenizer(text["test"],padding=True)["input_ids"]
#             return {"train":train_tokenized_text,"test":test_tokenized_text}
          



use_hf = True

training_config = {
    "model": {"pretrained_model_name":model_name,"max_length":512},
    "dataset":{"use_hf" : use_hf,"path":df_path},
    "verbose":True
}

def tokenization(batch):
    tokenized = tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=training_config["model"]["max_length"]
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

train_dataset = df["train"].map(tokenization, batched=True)
test_dataset = df["test"].map(tokenization, batched=True)  


# FOR GENERATION MODELS
def inference(input_text, model,tokenizer,max_input_length = 512,max_output_Length = 512):
    # Tokenizer
    input = tokenizer.encode(input_text, return_tensors="pt",max_length=max_input_length).to(model.device)
    
    # input_len = input["input_ids"].shape[-1]

    # Generate
    generate_tokens_prompt = model.generate(**input,max_length = max_output_Length)
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
                (1, training_config["model"]["max_length"]), dtype=torch.long
            )
        }
    )
    * training_args.gradient_accumulation_steps
)
# print("Memory footprint", model.get_memory_footprint() / 1e9, "GB")
# print("Flops", model_flops / 1e9, "GFLOPs")

# trainer= SFTTrainer(
#     model=model,
#     args=training_arguments,
#     train_dataset=dataset["train"],
#     eval_dataset=dataset["test"],
#     processing_class=tokenizer,
#     peft_config=peft_config,
# )

trainer = Trainer(
model=model,
args=training_args,
train_dataset=train_dataset,
eval_dataset=test_dataset,
)

trainer.do_grad_scaling = False

training_output = trainer.train()

save_dir = f'{output_dir}/final'

trainer.save_model(save_dir)
print("Saved model to:", save_dir)

finetuned_model = AutoModelForSequenceClassification.from_pretrained(save_dir, local_files_only=True)

# finetuned_model.to(device) 

test_question = df["test"]['text'][0]
print("Question input (test):", test_question)

# print("Finetuned slightly model's answer: ")
# print(inference(test_question, finetuned_model, tokenizer))
print("Finetuned slightly model's answer: ")
print(finetuned_model.predict(test_question))


test_answer = test_dataset["test"]['label_text'][0]
print("Target answer output (test):", test_answer)

evalution_mode = model.evaluate()

print("Evaluation mode:", evalution_mode)




# print("Training completed.")
# trainer.save_model()

# print("Model saved to", output_dir)
