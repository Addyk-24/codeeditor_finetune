from transformers import AutoTokenizer,AutoModelForCausalLM,TrainingArguments,AutoModelForCausalLM

from datasets import load_dataset

model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2") 

tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")

dataset = load_dataset("mteb/tweet_sentiment_extraction")

# for i in range(len(dataset["test"])):
#     print("Dataset:",dataset["test"]["text"][i])

def tokenize_func(text):
    return tokenizer(text["test"],truncation=True)["input_ids"]

tokenized_dataset = dataset.map(tokenize_func,batched=True)

print("Tokenized Dataset: \n", tokenized_dataset["test"]["input_ids"][0])