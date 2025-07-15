import os
import subprocess
import time
import re
import argparse
from pathlib import Path

def run_collect_data(env_id, model_run_id, human_agent_run_id, human_reaction_delay, human_comprehend_bool=True):
    """Run data collection for a specific human reaction delay"""
    cmd = [
        "python", "collect_data.py",
        "--env_id", env_id,
        "--model_run_id", model_run_id,
        "--agent_type", "mlp",
        "--trainer_type", "base",
        "--agent_obs_mode", "history",
        "--feature_extractor", "none",
        "--human_agent_type", "mlp",
        "--human_agent_run_id", human_agent_run_id,
        "--human_reaction_delay", str(human_reaction_delay),
        "--num_envs", "16",
        "--num_steps", "600"
    ]
    
    if human_comprehend_bool:
        cmd.append("--human_comprehend_bool")
    
    print(f"Running data collection for delay {human_reaction_delay}...")
    # Capture the output
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Print the output for debugging
    print(result.stdout)
    if result.stderr:
        print("Errors:", result.stderr)
    
    # Extract the data directory from the output
    match = re.search(r"Data saved to (data/.*?)$", result.stdout, re.MULTILINE)
    if not match:
        raise RuntimeError(f"Could not find data directory in output for delay {human_reaction_delay}")
    
    data_dir = match.group(1)
    print(f"Found data directory: {data_dir}")
    return data_dir

def run_analysis(data_dirs, policy_names, output_dir, save_figure=True):
    """Run trajectory analysis on collected data"""
    cmd = [
        "python", "analyze_trajectories.py",
        "--data_dirs"
    ] + data_dirs + [
        "--policy_names"
    ] + policy_names + [
        "--output_dir", output_dir
    ]
    
    if save_figure:
        cmd.append("--save_figure")
    
    print(f"Running analysis...")
    subprocess.run(cmd)

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run delay analysis for SimpleNotiDangerZoneLunarLander')
    parser.add_argument('--model_run_id', type=str, required=True,
                      help='Model run ID to use for data collection')
    args = parser.parse_args()

    # Configuration for SimpleNotiDangerZoneLunarLander
    env_id = "SimpleNotiDangerZoneLunarLander"
    model_run_id = args.model_run_id
    human_agent_run_id = "xlq34dpt"
    base_output_dir = f"analysis_results/ll_robustness_analysis_{model_run_id}"
    
    # Create output directory
    os.makedirs(base_output_dir, exist_ok=True)
    
    # Collect data for different delays
    data_dirs = []
    policy_names = []
    
    for delay in range(5):  # 0 to 4
        try:
            data_dir = run_collect_data(
                env_id=env_id,
                model_run_id=model_run_id,
                human_agent_run_id=human_agent_run_id,
                human_reaction_delay=delay
            )
            data_dirs.append(data_dir)
            policy_names.append(f"delay_{delay}")
        except Exception as e:
            print(f"Error collecting data for delay {delay}: {e}")
            continue
    
    if not data_dirs:
        print("No data was collected successfully")
        return
    
    # Run analysis
    try:
        run_analysis(
            data_dirs=data_dirs,
            policy_names=policy_names,
            output_dir=base_output_dir,
            save_figure=True
        )
        print(f"Analysis complete. Results saved to {base_output_dir}")
    except Exception as e:
        print(f"Error running analysis: {e}")

if __name__ == "__main__":
    main() 