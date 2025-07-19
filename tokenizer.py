from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")

def tokenize_text(text):
    tokenized_text = tokenizer(text)["input_ids"]
    return tokenized_text

def main():
    sample_text = "Hi, This is Aditya and I like to explore new technologies."
    tokenized_output = tokenize_text(sample_text)
    print("Tokenized Output:", tokenized_output)

if __name__ == "__main__":
    main()
