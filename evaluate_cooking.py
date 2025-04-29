#!/usr/bin/env python
"""
Standalone evaluation script for the cooking environment.
This script evaluates a trained model in the cooking environment.
"""

import os
import sys
import torch
import argparse
import numpy as np
from pathlib import Path

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import required modules
from src.configs.args import Args
from src.utils.evaluate import CookingLSTMEvaluator
from src.utils.util import make_env

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate a trained model in the cooking environment")
    parser.add_argument("--model_path", type=str, required=False, help="Path to the trained model")
    parser.add_argument("--model_run_id", type=str, required=False, default="u2wo0jdp", help="Wandb run id of the trained model")
    parser.add_argument("--run_name", type=str, default="evaluation", help="Name for this evaluation run")
    parser.add_argument("--layout_name", type=str, default="small", help="Layout name for the cooking environment")
    parser.add_argument("--max_episode_steps", type=int, default=1000, help="Maximum number of steps per episode")
    parser.add_argument("--eval_episodes", type=int, default=1, help="Number of episodes to evaluate")
    parser.add_argument("--capture_video", action="store_true", help="Capture video of the evaluation")
    parser.add_argument("--visualize", action="store_true", help="Visualize the trajectory")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cuda or cpu)")
    parser.add_argument("--cuda", action="store_true", help="Use CUDA if available")
    parser.add_argument("--EXPLORE", action="store_true", help="Enable exploration in the LangStayAgent")
    parser.add_argument("--VISION_LIMIT", action="store_true", help="Enable vision limit in the LangStayAgent")
    parser.add_argument("--discretization", type=str, default="simple", help="Discretization method for the environment")
    return parser.parse_args()

def main():
    """Main function to run the evaluation."""
    # Parse command line arguments
    cli_args = parse_args()
    if cli_args.model_path is None:
        import wandb
        api = wandb.Api()
        run = api.run(f"tri/timing/{cli_args.model_run_id}")
        cli_args.model_path = run.config['filepath'] + "/agent.pt"
        if not os.path.exists(cli_args.model_path):
            cli_args.model_path = None
            print(f"Model file not found: {cli_args.model_path}")
    
    # Create Args object with default values
    args = Args()
    
    # Update Args with command line arguments
    args.model_path = cli_args.model_path
    # args.layout_name = cli_args.layout_name
    # args.max_episode_steps = cli_args.max_episode_steps
    # args.device = cli_args.device
    # args.cuda = cli_args.cuda
    # args.EXPLORE = cli_args.EXPLORE
    # args.VISION_LIMIT = cli_args.VISION_LIMIT
    # args.discretization = cli_args.discretization
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    if args.device is None:
        args.device = device
    
    # Create evaluator
    evaluator = CookingLSTMEvaluator(args, cli_args.run_name)
    
    # Run evaluation
    episodic_returns, type2_counts, overwritten_counts, action_length_varieties = evaluator.evaluate(
        model_path=args.model_path,
        make_env=make_env,
        eval_episodes=cli_args.eval_episodes,
        model=None,  # This will be loaded inside the evaluator
        device=device,
        capture_video=True,
        visualize=cli_args.visualize,
        use_random_start_state=False,
        fixed_objects_start_state_mode=2,
    )
    
    # Print results
    print(f"Evaluation results for {cli_args.run_name}:")
    print(f"Average episodic return: {np.mean(episodic_returns):.2f} ± {np.std(episodic_returns):.2f}")
    print(f"Average type2 counts: {np.mean(type2_counts):.2f} ± {np.std(type2_counts):.2f}")
    print(f"Average overwritten counts: {np.mean(overwritten_counts):.2f} ± {np.std(overwritten_counts):.2f}")
    
    # Print action length varieties
    print("Action length varieties:")
    for length, counts in action_length_varieties.items():
        print(f"  Length {length}: {np.mean(counts):.2f} ± {np.std(counts):.2f}")
    
    # Save results to file
    results_dir = Path(f"results/{cli_args.run_name}")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    with open(results_dir / "evaluation_results.txt", "w") as f:
        f.write(f"Evaluation results for {cli_args.run_name}:\n")
        f.write(f"Average episodic return: {np.mean(episodic_returns):.2f} ± {np.std(episodic_returns):.2f}\n")
        f.write(f"Average type2 counts: {np.mean(type2_counts):.2f} ± {np.std(type2_counts):.2f}\n")
        f.write(f"Average overwritten counts: {np.mean(overwritten_counts):.2f} ± {np.std(overwritten_counts):.2f}\n")
        f.write("Action length varieties:\n")
        for length, counts in action_length_varieties.items():
            f.write(f"  Length {length}: {np.mean(counts):.2f} ± {np.std(counts):.2f}\n")
    
    print(f"Results saved to {results_dir / 'evaluation_results.txt'}")

if __name__ == "__main__":
    main() 