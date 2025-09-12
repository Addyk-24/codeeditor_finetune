# Load model directly

# Step 1 : Installing Required Libraries
import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
import torch.nn as nn
import bitsandbytes as bnb
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM

from datasets import load_dataset

ds = load_dataset("Abirate/english_quotes")

# Step 2: Load the Pretrained Model


tokenizer = AutoTokenizer.from_pretrained("distilbert/distilgpt2")
model = AutoModelForCausalLM.from_pretrained("distilbert/distilgpt2")

# Step 3 : Freezing the Model’s Parameters
for param in model.parameters():
    param.requires_grad = False # Freeze the model parameters
    if param.ndim == 1:
        # It is done in order to have mixed-precision training, to have the main computations done in float16 (to speed up training and reduce memory consumption), but certain parameters—like biases—are kept in float32 to avoid numerical instability.
        param.data = param.data.to(torch.float32) 

    # The first line is for Gradient checkpointing and it is a memory-saving technique. When enabled, instead of storing all intermediate activations needed for backpropagation, the model recomputes some activations during the backward pass. This allows the model to train using less GPU memory, but it incurs a slight computational cost due to recomputation.
    model.gradient_checkpointing_enable()
    # The second line ensures that gradients are calculated for the model’s inputs, which can be useful when you need to compute gradients with respect to the input data (for example, in adversarial training or certain fine-tuning methods).
    model.enable_input_require_grads()

    # in order to have output in higher precision
    class CastOutputToFloat(nn.Sequential):
        def forward(Self,x):
            return super().forward(x).to(torch.float32)
    model.lm_head = CastOutputToFloat(model.lm_head)
    # In the above code, the forward method overrides the default behavior and ensures that the output of the model's forward pass is cast to torch.float32. This ensures that the output is always in 32-bit floating point precision.

    # The Above code is done because model.lm_head typically refers to the final output layer of the language model, which produces the logits or predictions. By wrapping model.lm_head with CastOutputToFloat, we ensures that the output from the language model head is always in float32 precision, even if the model was performing computations in another precision (like float16 or bfloat16), which is common in mixed-precision training.

# Step 4 : Checking Trainable Parameters

# The above code help’s us to know how much paraters were actually in the model, how much are frozen and how many are now trainable for LoRA as shown below :

def print_trainable_parameters(model):
    """ Printing the no of trainable parameters in the model"""
    trainable_params = 0
    all_param = 0
    for _ , param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(f"Trainable params: {trainable_params} || all params: {all_param} || trainable%: {trainable_params*100/all_param}")
