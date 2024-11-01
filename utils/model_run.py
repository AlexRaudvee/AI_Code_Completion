from transformers import AutoTokenizer, AutoModelForCausalLM
from func import load_examples_from_json, generate_completion

# Load the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("bigcode/tiny_starcoder_py")
model = AutoModelForCausalLM.from_pretrained("bigcode/tiny_starcoder_py")

# loading the data
examples = load_examples_from_json("data/code/code.json")

print(examples[0])
# Generate completions
for example in examples:
    example["generated_middle"] = generate_completion(model, tokenizer, example["prefix"], example["suffix"])

print(examples[0])
# Save examples to a JSON file
# with open("data/code/code.json", "w") as f:
#     json.dump(examples, f, indent=4)