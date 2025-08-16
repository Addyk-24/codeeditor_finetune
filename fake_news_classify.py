from transformers import AutoTokenizer, AutoModelForMaskedLM

model_name = "distilbert/distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name,torch_dtype=torch.float16,device_map="auto")


