# imports
import os
import json 


# FUNCTIONS FOR GENERAL USE

def load_examples_from_json(file_path):
    """
    Loads examples from a JSON file and returns them as a list of dictionaries.
    
    Parameters
    ---
    - file_path : str
        Path to the JSON file containing examples.
    
    Returns
    ---
    - examples : list
        A list of dictionaries, each with 'prefix', 'middle', and 'suffix'.
    """
    with open(file_path, "r") as f:
        examples = json.load(f)
    return examples


# FUNCTIONS FOR DATASET CREATION

def load_code_files(directory_path, file_extension=".py"):
    """
    Loads code files from a specified directory and returns their content along with the count of files.

    Parameters
    ---
    - directory_path : str 
        Path to the directory containing code files.
    - file_extension : str
        File extension to filter code files, default is ".py".

    Returns
    ---
    - tuple : A tuple containing:
        - code_snippets (list of str): List of code content strings from each file.
        - files_num (int): Number of files processed and loaded.

    Example
    ---
    >>> code_snippets, files_num = load_code_files("path/to/directory")
    >>> print(files_num)
    3  # Assuming there were 3 files with .py extension
    """
    code_snippets = []
    files_num = 0
    for filename in os.listdir(directory_path):
        if filename.endswith(file_extension):
            with open(os.path.join(directory_path, filename), "r") as f:
                code_snippets.append(f.read())
            files_num += 1
    return code_snippets, files_num

def adjust_empty_sections(prefix, middle, suffix):
    """
    Adjusts the prefix, middle, and suffix sections of code to ensure none are empty.
    
    If `middle` is empty, it moves one line from `suffix` to `middle` and removes this line from `suffix`.
    If `suffix` is empty, it shifts one line up from `prefix` to `middle` and from `middle` to `suffix`,
    ensuring all three sections are populated if possible.

    Parameters
    ---
    - prefix : str 
        The prefix section of code, representing content before the cursor.
    - middle : str 
        The middle section, representing code where a user might expect completion.
    - suffix : str 
        The suffix section of code, representing content after the cursor.

    Returns
    ---
    - tuple : A tuple containing the adjusted `prefix`, `middle`, and `suffix` strings.

    Example
    ---
    >>> prefix, middle, suffix = adjust_empty_sections("line1\nline2", "", "line3\nline4")
    >>> print(middle)
    "line3"  # The first line from suffix moves to middle
    """
    # Split prefix, middle, and suffix into lines for easier manipulation
    prefix_lines = prefix.splitlines()
    suffix_lines = suffix.splitlines()

    # If middle is empty, add one line from suffix and remove it from suffix
    if not middle.strip() and suffix_lines:
        middle = suffix_lines.pop(0)
        suffix = "\n".join(suffix_lines)
    
    # If suffix is empty, shift lines up to fill suffix and middle from prefix
    if not suffix.strip():
        # If possible, add the last line of prefix to middle
        if prefix_lines:
            middle = prefix_lines.pop()
            prefix = "\n".join(prefix_lines)
    
    return prefix, middle, suffix


# FUNCTIONS FOR MODEL APPLICATION 


def generate_completion(model, tokenizer, device, prefix, suffix, max_new_tokens=1000):
    """
    Generate a code completion by filling in the missing middle part based on the given prefix and suffix.
    
    This function takes a prefix and suffix, formats them into a specific prompt with custom tokens, and uses a language model 
    to generate the intermediate text that fits between them. The function returns only the generated middle part, excluding 
    the prefix and suffix from the final output. 
    
    Parameters
    ---
    - model : transformers.PreTrainedModel
        The language model used to generate the completion.
    - tokenizer : transformers.PreTrainedTokenizer
        The tokenizer associated with the model for encoding and decoding the text.
    - device : torch.device
        The device (CPU or GPU) where the model and inputs are processed.
    - prefix : str
        The starting text provided as context for generating the middle part.
    - suffix : str
        The ending text that follows the middle part to be generated.
    - max_new_tokens : int, optional, default=1000
        The maximum number of new tokens to be generated for the middle part. This restricts the output length.

    Returns
    ---
    - middle_text : str
        The generated middle part text that completes the input prompt between the given prefix and suffix.

    Example
    ---
    >>> from transformers import AutoModelForCausalLM, AutoTokenizer
    >>> import torch

    >>> model_name = "your-model-name"
    >>> model = AutoModelForCausalLM.from_pretrained(model_name)
    >>> tokenizer = AutoTokenizer.from_pretrained(model_name)
    >>> device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    >>> model.to(device)

    >>> prefix = "def add_numbers(a, b):"
    >>> suffix = "return result"

    >>> middle = generate_completion(model, tokenizer, device, prefix, suffix, max_new_tokens=50)
    >>> print("Generated middle:", middle)
    """
    
    # Format the input using the <fim_prefix>, <fim_suffix>, and <fim_middle> tokens
    input_text = f"<fim_prefix>{prefix}<fim_suffix>{suffix}<fim_middle>"
    
    # Encode the input and move it to the device (if required by model setup)
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=False).to(device)
    
    # Generate the completion without specifying max_length, but limiting new tokens
    outputs = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=max_new_tokens,  # Limit only the generated tokens
        pad_token_id=tokenizer.pad_token_id
    )
    
    # Decode the output and return only the generated middle part
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
    # print(generated_text)
    # Remove prefix and suffix to get only the generated middle part
    middle_text = generated_text[generated_text.find("<fim_middle>") + len("<fim_middle>"): len(generated_text) - len(tokenizer.eos_token)]
    return middle_text


# FUNCTIONS FOR EVALUATIONS

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.chrf_score import sentence_chrf


def exact_match(predicted, actual):
    """Check if the predicted text matches the actual text exactly."""
    return int(predicted.strip() == actual.strip())

def compute_bleu(predicted, actual):
    """Compute BLEU score with smoothing."""

    smoothing_func = SmoothingFunction().method4

    return sentence_bleu(
        [actual.split()], 
        predicted.split(), 
        # weights=weights, 
        smoothing_function=smoothing_func
    )

def compute_chrf(predicted, actual):
    """Compute ChrF score."""
    return sentence_chrf([actual], predicted)