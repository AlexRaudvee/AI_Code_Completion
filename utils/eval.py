# imports 
import os
import json
import tqdm

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

import subprocess

def run_code_from_file(file_path):
    """
    Run a given Python file and return the output as a string.
    
    Parameters:
        file_path (str): The path to the Python file to execute.

    Returns:
        tuple: (output, error) - Output and any error messages from the execution.
    """
    # Use subprocess to run the code and capture the output
    result = subprocess.run(
        ['python', file_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Return output and any errors
    return result.stdout, result.stderr

def functional_accuracy(predicted_file_path, ground_truth_file_path):
    """
    Evaluate functional accuracy by running the predicted and ground truth Python files.
    
    Parameters:
        predicted_file_path (str): The path to the predicted code file.
        ground_truth_file_path (str): The path to the ground truth code file.

    Returns:
        bool: True if outputs are the same, False otherwise.
    """
    # Run the predicted code
    predicted_output, predicted_error = run_code_from_file(predicted_file_path)
    
    # Run the ground truth code
    ground_truth_output, ground_truth_error = run_code_from_file(ground_truth_file_path)

    # Print outputs for debugging
    # print("Predicted Output:")
    # print(predicted_output)
    # print("Predicted Error:")
    # print(predicted_error)

    # print("Ground Truth Output:")
    # print(ground_truth_output)
    # print("Ground Truth Error:")
    # print(ground_truth_error)

    # Compare outputs
    if predicted_output.strip() == ground_truth_output.strip():
        return True
    else:
        return False

# Example usage:
predicted_file_path = "data/code_files/code_5.py"  # Replace with the path to your predicted code file
ground_truth_file_path = "data/code_files/Net_RNN.py"  # Replace with the path to your ground truth code file

predicted_file_paths = ["data/code_files/code_5.py", "data/code_files/code_10.py", "data/code_files/code_15.py", "data/code_files/code_20.py",
                        "data/code_files/code_25.py", "data/code_files/code_30.py","data/code_files/code_35.py", "data/code_files/code_40.py", 
                        "data/code_files/code_45.py", "data/code_files/code_50.py", "data/code_files/code_55.py", "data/code_files/code_60.py",
                        "data/code_files/code_65.py", "data/code_files/code_70.py"]

ground_truth_file_paths = ["data/code_files/Net_RNN.py", "data/code_files/Net_RNN.py", "data/code_files/data_loading_Enc_Dec.py", "data/code_files/data_loading_Enc_Dec.py",
                           "data/code_files/Net_Enc_Dec.py", "data/code_files/Net_Enc_Dec.py", "data/code_files/main_RNN.py", "data/code_files/main_RNN.py",
                           "data/code_files/main_Enc_Dec.py", "data/code_files/main_Enc_Dec.py", "data/code_files/Net_LSTM.py", "data/code_files/Net_LSTM.py", 
                           "data/code_files/func.py", "data/code_files/func.py"]

count = 0
for prediction, truth in tqdm.tqdm(zip(predicted_file_paths, ground_truth_file_paths)):
    if functional_accuracy(prediction, truth):
        count +=1

result = count / len(predicted_file_paths)

# Evaluate functional accuracy
print("%\ of similarity between files with AI completion and truth files", result)


# Store the averages in a dictionary
summary_results = {
    "average_exact_match": average_em,
    "average_bleu": average_bleu,
    "average_chrf": average_chrf,
    "average_functional_accuracy": result
}

# Save the summary results to a JSON file
output_file = "data/metric_summary_results.json"
with open(output_file, "w") as f:
    json.dump(summary_results, f, indent=4)

print(f"Summary results saved to {output_file}")