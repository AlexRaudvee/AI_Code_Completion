# imports 
import os
import json

import pandas as pd

from sklearn.metrics import accuracy_score
from func import exact_match, compute_bleu, compute_chrf, load_examples_from_json


# load the data 
examples = load_examples_from_json("data/code/code.json")

# Annotate and evaluate
results = []
for example in examples:
    generated_middle = example["generated_middle"]
    actual_middle = example["middle"]
    
    # Calculate metrics
    em = exact_match(generated_middle, actual_middle)
    bleu = compute_bleu(generated_middle, actual_middle)
    chrf = compute_chrf(generated_middle, actual_middle)
    
    # Store results
    results.append({
        "prefix": example["prefix"],
        "generated_middle": generated_middle,
        "actual_middle": actual_middle,
        "suffix": example["suffix"],
        "exact_match": em,
        "bleu": bleu,
        "chrf": chrf
    })


# Convert results to a DataFrame for analysis
df = pd.DataFrame(results)

# Calculate the average score for each metric
average_em = df["exact_match"].mean()
average_bleu = df["bleu"].mean()
average_chrf = df["chrf"].mean()

# Summarize results
print("Average Exact Match:", average_em)
print("Average BLEU:", average_bleu)
print("Average ChrF:", average_chrf)

# Store the averages in a dictionary
summary_results = {
    "average_exact_match": average_em,
    "average_bleu": average_bleu,
    "average_chrf": average_chrf
}

# Save the summary results to a JSON file
output_file = "data/metric_summary_results.json"
with open(output_file, "w") as f:
    json.dump(summary_results, f, indent=4)

print(f"Summary results saved to {output_file}")



# CREATION OF FILES FOR MANUAL REVIEW


# Load JSON data from file
with open("data/code/code.json", "r") as file:
    data = json.load(file)

# Directory to save the .py files
output_dir = "data/generated_code_files"
os.makedirs(output_dir, exist_ok=True)

# Iterate over each entry in the JSON file and process every 5th entry
for i, entry in enumerate(data):
    # Process only every 5th entry (5th, 10th, 15th, etc.)
    if (i + 1) % 5 == 0:
        # Retrieve prefix, generated_middle, and suffix
        prefix = entry.get("prefix", "")
        middle = entry.get("generated_middle", "")
        suffix = entry.get("suffix", "")

        # Combine them to form the complete code
        complete_code = f"{prefix}#<PREDICTION>\n{middle}#</PREDICTION>\n{suffix}"

        # Define the output filename, e.g., "code_5.py", "code_10.py", etc.
        output_filename = os.path.join(output_dir, f"code_{i + 1}.py")

        # Write the combined code to the .py file
        with open(output_filename, "w") as output_file:
            output_file.write(complete_code)

        print(f"Generated {output_filename}")

print("Selected files have been generated successfully.")
