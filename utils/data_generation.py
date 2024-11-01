import os
import json
import random

from func import load_code_files, adjust_empty_sections


def generate_code_completion_examples(code_snippets, num_examples=10, output_file="examples.json"):
    """
    Generates prefix, middle, and suffix examples from code snippets, saves them to a JSON file
    inside a 'data/code' directory, and returns the examples list.
    
    Parameters:
    ---
    - code_snippets (list): A list of code strings to generate examples from.
    - num_examples (int): The number of examples to generate from each code snippet.
    - output_file (str): Name of the JSON file to save examples in the 'data/code' directory.
    
    Returns:
    
    - examples (list): A list of dictionaries, each containing 'prefix', 'middle', and 'suffix'.
    """
    # Ensure the 'data/code' directory exists
    output_dir = os.path.join("data", "code")
    os.makedirs(output_dir, exist_ok=True)
    
    total_lines = 0
    examples = []

    for code in code_snippets:
        lines = code.splitlines()
        if len(lines) < 3:
            continue  # Skip short code files
        
        # Generate random cursor position for each example
        for _ in range(num_examples):
            cursor_line = random.randint(1, len(lines) - 2)  # Avoid first and last line
            prefix = "\n".join(lines[:cursor_line])
            middle = lines[cursor_line]
            suffix = "\n".join(lines[cursor_line + 1:])

            # Adjust sections if any of them are empty
            prefix, middle, suffix = adjust_empty_sections(prefix, middle, suffix)
            
            # Append the adjusted example
            examples.append({"prefix": prefix, "middle": middle, "suffix": suffix})

        total_lines += len(lines)
    
    # Define the path to the output JSON file inside 'data/code'
    output_path = os.path.join(output_dir, output_file)
    
    # Save examples to a JSON file
    with open(output_path, "w") as f:
        json.dump(examples, f, indent=4)
    
    return examples, total_lines

# Load code snippets from your project directory
code_snippets, files_num = load_code_files("data/code_")
examples, total_lines = generate_code_completion_examples(code_snippets, output_file="code.json")

print(f"""\n
Statistics:
\t Number of code lines: {total_lines}
\t Number of files used: {files_num}
\t Number of pref-mid-suf trios: {len(examples)}
""")