import json

import pandas as pd

from sklearn.metrics import accuracy_score
from func import exact_match, compute_bleu, compute_chrf, load_examples_from_json


# load the data 
examples = load_examples_from_json("data/code/code.json")

# Annotate and evaluate
results = []
for example in examples:
    generated_middle = example["middle"]
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
output_file = "metric_summary_results.json"
with open(output_file, "w") as f:
    json.dump(summary_results, f, indent=4)

print(f"Summary results saved to {output_file}")
# Suggesting the best metric based on correlation with manual annotations (if available)
# For simplicity, this example only uses averages to judge performance. Further analysis might
# involve comparing correlations between manual judgments and these metric scores.
