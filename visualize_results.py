"""
Script to visualize and compare results from different models.
"""

import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob
from matplotlib.colors import LinearSegmentedColormap


def find_result_files(results_dir):
    """
    Find all result files in the given directory.
    
    Args:
        results_dir: Directory containing result files
        
    Returns:
        Dictionary mapping model names to their result files
    """
    models = {}
    
    # Find accuracy files
    accuracy_files = glob.glob(os.path.join(results_dir, "*_accuracy.png"))
    for file in accuracy_files:
        model_name = os.path.basename(file).split("_")[0]
        if model_name not in models:
            models[model_name] = {}
        models[model_name]["accuracy"] = file
    
    # Find loss files
    loss_files = glob.glob(os.path.join(results_dir, "*_loss.png"))
    for file in loss_files:
        model_name = os.path.basename(file).split("_")[0]
        if model_name not in models:
            models[model_name] = {}
        models[model_name]["loss"] = file
    
    # Find confusion matrix files
    cm_files = glob.glob(os.path.join(results_dir, "*_confusion_matrix.png"))
    for file in cm_files:
        model_name = os.path.basename(file).split("_")[0]
        if model_name not in models:
            models[model_name] = {}
        models[model_name]["confusion_matrix"] = file
    
    # Find ROC curve files
    roc_files = glob.glob(os.path.join(results_dir, "*_roc_curve.png"))
    for file in roc_files:
        model_name = os.path.basename(file).split("_")[0]
        if model_name not in models:
            models[model_name] = {}
        models[model_name]["roc_curve"] = file
    
    return models


def compare_results(models, output_dir):
    """
    Create comparison visualizations for the models.
    
    Args:
        models: Dictionary of model results
        output_dir: Directory to save the comparison visualizations
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a comparison plot
    plt.figure(figsize=(15, 10))
    
    # Compare accuracy plots (this would require parsing the plots or having the raw data)
    plt.subplot(2, 2, 1)
    plt.title("Model Accuracy Comparison")
    plt.xlabel("Model")
    plt.ylabel("Test Accuracy (%)")
    
    # Example data - you would need to extract actual metrics from results
    # This is just a placeholder for demonstration
    model_names = list(models.keys())
    accuracy_values = np.random.uniform(70, 95, size=len(model_names))
    
    plt.bar(model_names, accuracy_values)
    plt.ylim([0, 100])
    
    # Compare F1 scores (placeholder)
    plt.subplot(2, 2, 2)
    plt.title("Model F1 Score Comparison")
    plt.xlabel("Model")
    plt.ylabel("F1 Score")
    
    f1_values = np.random.uniform(0.7, 0.95, size=len(model_names))
    plt.bar(model_names, f1_values)
    plt.ylim([0, 1])
    
    # Compare training time (placeholder)
    plt.subplot(2, 2, 3)
    plt.title("Training Time Comparison")
    plt.xlabel("Model")
    plt.ylabel("Training Time (minutes)")
    
    time_values = np.random.uniform(10, 60, size=len(model_names))
    plt.bar(model_names, time_values)
    
    # Compare model complexity (placeholder)
    plt.subplot(2, 2, 4)
    plt.title("Model Complexity Comparison")
    plt.xlabel("Model")
    plt.ylabel("Number of Parameters (millions)")
    
    # Example parameter counts (these should be actual values)
    param_values = {
        "cnn": 0.5,  # CNN model (~500K parameters)
        "transformer_1_head": 1.2,  # Transformer with 1 head (~1.2M parameters)
        "transformer_2_head": 1.5,  # Transformer with 2 heads (~1.5M parameters)
        "transformer_4_head": 2.0,  # Transformer with 4 heads (~2M parameters)
    }
    
    params = [param_values.get(model, 1.0) for model in model_names]
    plt.bar(model_names, params)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "model_comparison.png"))
    
    # Create a table of results
    results_table = pd.DataFrame({
        "Model": model_names,
        "Accuracy (%)": accuracy_values,
        "F1 Score": f1_values,
        "Training Time (minutes)": time_values,
        "Parameters (millions)": params
    })
    
    # Save the table to CSV
    results_table.to_csv(os.path.join(output_dir, "model_comparison.csv"), index=False)
    
    print(f"Comparison visualizations saved to {output_dir}")
    
    return results_table


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize and compare model results")
    
    parser.add_argument("--results_dir", type=str, required=True,
                        help="Directory containing result files")
    parser.add_argument("--output_dir", type=str, default="./comparisons",
                        help="Directory to save comparison visualizations")
    
    args = parser.parse_args()
    
    # Find result files
    models = find_result_files(args.results_dir)
    
    if not models:
        print(f"No result files found in {args.results_dir}")
    else:
        print(f"Found results for the following models: {', '.join(models.keys())}")
        
        # Compare results
        results_table = compare_results(models, args.output_dir)
        
        # Print the results table
        print("\nResults summary:")
        print(results_table.to_string(index=False))
