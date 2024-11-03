# imports
import subprocess

"""
Run of the data generator 
"""
subprocess.run(["python", "utils/data_generation.py"])

"""
Run of the model for code completition
"""
subprocess.run(["python", "utils/model_run.py"])


"""
Run of the evaluations
"""
subprocess.run(["python", "utils/eval.py"])