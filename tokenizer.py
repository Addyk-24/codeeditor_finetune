from transformers import AutoTokenizer
import pprint
tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")

# WITHOUT PADDING AND EOS
# def tokenize_text(text):
#     tokenized_text = tokenizer(text)["input_ids"]
#     return tokenized_text

def tokenize_text_with_padding(text):
    # tokenizer.pad_token = tokenizer.eos_token since bert model deos not have eos token so this line does not affect
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenized_text = tokenizer(text,padding=True)["input_ids"]
    return tokenized_text
def main():
    sample_text = "Hi, This is Aditya and I like to explore new technologies."
    # tokenized_output = tokenize_text(sample_text)
    tokenized_output = tokenize_text_with_padding(sample_text)
    print("Tokenized Output:", tokenized_output)

if __name__ == "__main__":
    main()
