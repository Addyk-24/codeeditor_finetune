# There is few steps to run the evaluation in GPUs using Lamini

# finetuned_model = BasicModelRunner(
#     "lamini/lamini_docs_finetuned"
# )
# finetuned_output = finetuned_model(
#     test_dataset_list # batched!
# ) 

# LETS THIS UNDER THE HOOD

import datasets
import tempfile
import logging
import random
import config
import os
import yaml
import logging
import difflib
import pandas as pd

import transformers
import datasets
import torch

from tqdm import tqdm
from utilities import *
from transformers import AutoTokenizer, AutoModelForCausalLM

logger = logging.getLogger(__name__)
global_config = None


# Try the ARC benchmark - for academic reading comprehension
# !python lm-evaluation-harness/main.py --model hf-causal --model_args pretrained=lamini/lamini_docs_finetuned --tasks arc_easy --device cpu

# There are many other benchmarks available, such as:
