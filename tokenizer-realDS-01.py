taylor_swift_dataset = "lamini/taylor_swift"
bts_dataset = "lamini/bts"
open_llms = "HuggingFaceTB/SmolLM3-3B"

from transformers import AutoTokenizer
import pprint
from datasets import load_dataset

tokenizer = AutoTokenizer.from_pretrained(open_llms)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
taylore_dataset = load_dataset(taylor_swift_dataset)
# print("dataset: \n",taylore_dataset["train"][0])


def tokenization(text):
    if "question" in text or "answer" in text:

        tokenized_question = tokenizer(text["question"], padding=True)["input_ids"]
        tokenized_answer = tokenizer(text["answer"], padding=True)["input_ids"]
        return {"question": tokenized_question, "answer": tokenized_answer}

    else:
        tokenized_text = tokenizer(text["train"],padding=True)["input_ids"]
        return {"text": tokenized_text}
    

tokenzied_taylor = taylore_dataset.map(tokenization,batched=True)

print("Tokenized Dataset: \n",tokenzied_taylor["train"][0])