from datasets import load_dataset
import itertools
import jsonlines

# Login using e.g. `huggingface-cli login` to access this dataset
ds = load_dataset("microsoft/NextCoderDataset")

# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

# tokenizer = AutoTokenizer.from_pretrained("LiquidAI/LFM2-700M")
# model = AutoModelForCausalLM.from_pretrained("LiquidAI/LFM2-700M")
print("Below is dtataet info:")
n = 5
top_n = list(itertools.islice(ds["train"],n))
print(top_n)