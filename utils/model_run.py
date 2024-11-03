# imports
import os
import json
import torch

from tqdm import tqdm 
from transformers import AutoTokenizer, AutoModelForCausalLM
from func import load_examples_from_json, generate_completion

# Disable parallelism in tokenizers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("bigcode/tiny_starcoder_py")
model = AutoModelForCausalLM.from_pretrained("bigcode/tiny_starcoder_py")

# check for cuda
if torch.cuda.is_available():
    device = "cuda"
else: 
    device = model.device
    
# set the pad token
tokenizer.add_special_tokens({"pad_token": "[PAD]"})

# Load the data
examples = load_examples_from_json("data/code/code.json")

# Generate completions
for example in tqdm(examples):
    example["generated_middle"] = generate_completion(model, tokenizer, device, example["prefix"], example["suffix"])

# Optionally, save examples to a JSON file
with open("data/code/code.json", "w") as f:
    json.dump(examples, f, indent=4)
    