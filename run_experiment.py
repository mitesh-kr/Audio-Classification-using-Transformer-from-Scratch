"""
Script to run experiments with different models.
"""

import subprocess
import os
import argparse
from datetime import datetime


def run_experiment(args):
    """
    Run experiments with different models.
    
    Args:
        args: Command line arguments
    """
    # Create output directory for results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_path, f"experiment_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Common arguments
    common_args = [
        "--data_path", args.data_path,
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--save_model"
    ]
    
    if args.use_wandb:
        common_args.extend([
            "--use_wandb",
            "--wandb_key", args.wandb_key,
            "--project_name", args.project_name
        ])
    
    # Models to run
    models = []
    if args.run_cnn:
        models.append(("cnn", 0.0001))
    
    if args.run_transformer:
        models.extend([
            ("transformer_1_head", 0.0001),
            ("transformer_2_head", 0.001),
            ("transformer_4_head", 0.001)
        ])
    
    # Run each model
    results = {}
    for model_type, lr in models:
        print(f"\n{'=' * 50}")
        print(f"Running experiment with {model_type}")
        print(f"{'=' * 50}\n")
        
        # Build command
        cmd = ["python", "main.py", 
               "--model_type", model_type,
               "--learning_rate", str(lr)]
        
        if "transformer" in model_type:
            cmd.extend(["--dropout", str(args.dropout)])
            
        cmd.extend(common_args)
        
        # Run the command
        try:
            subprocess.run(cmd, check=True)
            results[model_type] = "Success"
        except subprocess.CalledProcessError:
            results[model_type] = "Failed"
    
    # Print summary
    print("\n\nExperiment Summary:")
    print("=" * 50)
    for model_type, status in results.items():
        print(f"{model_type}: {status}")
    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run experiments with different models")
    
    # Dataset arguments
    parser.add_argument("--data_path", type=str, required=True, 
                        help="Path to the dataset directory")
    parser.add_argument("--output_path", type=str, default="./results", 
                        help="Path to save the results")
    
    # Experiment configuration
    parser.add_argument("--run_cnn", action="store_true", 
                        help="Run CNN model experiment")
    parser.add_argument("--run_transformer", action="store_true", 
                        help="Run Transformer model experiments")
    parser.add_argument("--epochs", type=int, default=100, 
                        help="Number of epochs to train for")
    parser.add_argument("--batch_size", type=int, default=40, 
                        help="Batch size for training")
    parser.add_argument("--dropout", type=float, default=0.2, 
                        help="Dropout rate for transformer models")
    
    # WandB arguments
    parser.add_argument("--use_wandb", action="store_true", 
                        help="Whether to use Weights & Biases for logging")
    parser.add_argument("--wandb_key", type=str, default="", 
                        help="WandB API key")
    parser.add_argument("--project_name", type=str, default="Audio_Classification", 
                        help="WandB project name")
    
    args = parser.parse_args()
    
    # Ensure at least one model type is selected
    if not (args.run_cnn or args.run_transformer):
        parser.error("At least one of --run_cnn or --run_transformer must be specified")
    
    run_experiment(args)
