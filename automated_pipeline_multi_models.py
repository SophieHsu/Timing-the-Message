#!/usr/bin/env python3
"""
Automated Pipeline for Multi-Model Agent Trajectory Analysis (with Checkpoint Selection)

This script automates the complete pipeline for comparing multiple models:
1. For each model_run_id, finds all checkpoints and selects the best performing one
2. Collects trajectory data using the best checkpoint
3. Runs analysis on the collected data
4. Extracts metrics and generates a comparative table including checkpoint information
5. Saves results in multiple formats (CSV, markdown table, console output)

Usage:
    python automated_pipeline_multi_models.py --model_run_ids model1,model2,model3 --delay 2

Required parameters:
    --model_run_ids: Comma-separated list of model run IDs to compare

Optional parameters:
    --delay: Human reaction delay to use for all models (default: 2)
    --env_id: Environment ID (default: SimpleNotiDangerZoneLunarLander)
    --human_agent_run_id: Human agent run ID (default: xlq34dpt) 
    --output_base_dir: Base output directory (default: analysis_results)
    --human_comprehend_bool: Whether human comprehends notifications (default: True)
    --save_figures: Whether to save analysis figures (default: True)
"""

import os
import subprocess
import time
import re
import argparse
import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple


def run_collect_data_with_checkpoint_selection(env_id: str, model_run_id: str, human_agent_run_id: str, 
                                              human_reaction_delay: str, human_comprehend_bool: bool = True, num_envs: int = 16, num_steps: int = 600) -> Tuple[str, Dict[str, Any]]:
    """Run data collection with checkpoint selection for a specific model"""
    cmd = [
        "python", "collect_data_checkpoint.py",
        "--env_id", env_id,
        "--model_run_id", model_run_id,
        "--agent_type", "mlp",
        "--trainer_type", "base",
        "--agent_obs_mode", "history",
        "--feature_extractor", "none",
        "--human_agent_type", "mlp",
        "--human_agent_run_id", human_agent_run_id,
        "--human_reaction_delay", human_reaction_delay,
        "--num_envs", str(num_envs),
        "--num_steps", str(num_steps),
        "--norm_obs"
    ]
    
    if human_comprehend_bool:
        cmd.append("--human_comprehend_bool")
    
    print(f"🔄 Running checkpoint-based data collection for model {model_run_id} with delay {human_reaction_delay}...")
    
    # Capture the output
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Print the output for debugging
    print(result.stdout)
    if result.stderr:
        print("❌ Errors:", result.stderr)
        
    if result.returncode != 0:
        raise RuntimeError(f"Data collection failed for model {model_run_id}")
    
    # Extract the data directory from the output
    data_dir_match = re.search(r"Data saved to (data/.*?)$", result.stdout, re.MULTILINE)
    if not data_dir_match:
        raise RuntimeError(f"Could not find data directory in output for model {model_run_id}")
    
    data_dir = data_dir_match.group(1)
    
    # Extract checkpoint information from the output
    checkpoint_info = {}
    checkpoint_match = re.search(r"Best checkpoint: (\d+) \(avg reward: ([\d.-]+)\)", result.stdout)
    if checkpoint_match:
        checkpoint_info = {
            'checkpoint_num': int(checkpoint_match.group(1)),
            'avg_reward': float(checkpoint_match.group(2))
        }
    
    # Try to load more detailed checkpoint info from the saved file
    try:
        checkpoint_file = Path(data_dir) / "checkpoint_selection.json"
        if checkpoint_file.exists():
            with open(checkpoint_file, 'r') as f:
                detailed_info = json.load(f)
                checkpoint_info.update(detailed_info.get('best_checkpoint', {}))
    except Exception as e:
        print(f"⚠️  Could not load detailed checkpoint info: {e}")
    
    print(f"✅ Data saved to: {data_dir}")
    print(f"🏆 Best checkpoint: {checkpoint_info.get('checkpoint_num', 'unknown')} (reward: {checkpoint_info.get('avg_reward', 'unknown')})")
    
    return data_dir, checkpoint_info


def run_analysis_and_capture_metrics(data_dirs: List[str], policy_names: List[str], 
                                   output_dir: str, save_figure: bool = True) -> str:
    """Run trajectory analysis and capture the output for metric extraction"""
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
    
    print(f"🔄 Running trajectory analysis...")
    
    # Capture both stdout and stderr
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    print("📊 Analysis output:")
    print(result.stdout)
    
    if result.stderr:
        print("❌ Analysis errors:")
        print(result.stderr)
        
    if result.returncode != 0:
        raise RuntimeError("Analysis failed")
        
    print(f"✅ Analysis complete. Results saved to {output_dir}")
    return result.stdout


def parse_metrics_from_output(output: str, policy_names: List[str], checkpoint_infos: List[Dict[str, Any]], 
                            env_type: str = "DangerZoneLunarLander") -> List[Dict[str, Any]]:
    """Parse metrics from the analysis output and include checkpoint information"""
    lines = output.split('\n')
    metrics_data = []
    
    # Find lines that contain comma-separated metrics
    # Format: success_rate, avg_steps_to_success, avg_overwrite_rate, avg_notification_rate, length_selection_rates[5], [entry_rate]
    metric_lines = []
    for line in lines:
        if ',' in line and any(char.isdigit() for char in line):
            # Check if this looks like a metrics line (contains numbers and commas)
            parts = line.strip().split(',')
            if len(parts) >= 5:  # At least 5 metrics expected
                try:
                    # Try to parse the first few values as floats to confirm this is a metrics line
                    float(parts[0].strip())
                    float(parts[1].strip())
                    metric_lines.append(line.strip())
                except ValueError:
                    continue
    
    if len(metric_lines) != len(policy_names):
        print(f"⚠️  Warning: Found {len(metric_lines)} metric lines but expected {len(policy_names)}")
        print("Metric lines found:")
        for i, line in enumerate(metric_lines):
            print(f"  {i}: {line}")
    
    for i, (policy_name, line) in enumerate(zip(policy_names, metric_lines)):
        parts = [p.strip() for p in line.split(',')]
        
        # Get checkpoint info for this model
        checkpoint_info = checkpoint_infos[i] if i < len(checkpoint_infos) else {}
        
        try:
            metrics = {
                'model_run_id': policy_name,
                'checkpoint': checkpoint_info.get('checkpoint_num', 'unknown'),
                'checkpoint_reward': checkpoint_info.get('avg_reward', 'unknown'),
                'success_rate': float(parts[0]),
                'steps_to_success_overwrite': float(parts[1]),
                'noti_rate': float(parts[2]),
                'long_rate': float(parts[3]),
                'domain': float(parts[4])
            }
            
            # Add environment-specific metrics
            if env_type == "DangerZoneLunarLander" and len(parts) > 5:
                metrics['entry_rate'] = float(parts[5])
            elif env_type == "multi-merge-v0" and len(parts) > 5:
                metrics['avg_vx'] = float(parts[5])
                
            metrics_data.append(metrics)
            
        except (ValueError, IndexError) as e:
            print(f"❌ Error parsing metrics for {policy_name}: {e}")
            print(f"   Line: {line}")
            continue
    
    return metrics_data


def format_table(metrics_data: List[Dict[str, Any]], delay: int) -> str:
    """Format metrics data into a nice table"""
    if not metrics_data:
        return "❌ No metrics data to display"
    
    # Create DataFrame
    df = pd.DataFrame(metrics_data)
    
    # Reorder columns for better presentation - checkpoint info comes early
    column_order = ['model_run_id', 'checkpoint', 'checkpoint_reward', 'success_rate', 'steps_to_success_overwrite', 'noti_rate', 'long_rate', 'domain']
    if 'entry_rate' in df.columns:
        column_order.append('entry_rate')
    if 'avg_vx' in df.columns:
        column_order.append('avg_vx')
    
    df = df[column_order]
    
    # Round numeric columns to appropriate precision
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df[numeric_columns] = df[numeric_columns].round(3)
    
    # Create table header
    table_str = f"\n{'='*100}\n"
    table_str += f"📊 MULTI-MODEL TRAJECTORY ANALYSIS RESULTS (with Best Checkpoints) - Delay: {delay}\n"
    table_str += f"{'='*100}\n\n"
    
    # Add the DataFrame as a string
    table_str += df.to_string(index=False, float_format='%.3f')
    table_str += f"\n\n{'='*100}\n"
    
    # Add summary statistics (excluding non-numeric columns)
    if len(df) > 1:
        table_str += "\n📈 SUMMARY STATISTICS:\n"
        table_str += f"{'='*50}\n"
        
        exclude_cols = ['model_run_id', 'checkpoint']
        for col in numeric_columns:
            if col not in exclude_cols:
                mean_val = df[col].mean()
                std_val = df[col].std()
                table_str += f"{col:<30}: {mean_val:.3f} ± {std_val:.3f}\n"
        table_str += f"{'='*50}\n"
    
    return table_str


def save_results(metrics_data: List[Dict[str, Any]], table_str: str, output_dir: str, delay: int):
    """Save results in multiple formats"""
    # Create results directory
    results_dir = Path(output_dir) / "results"
    results_dir.mkdir(exist_ok=True)
    
    # Save as CSV
    if metrics_data:
        df = pd.DataFrame(metrics_data)
        csv_path = results_dir / f"multi_model_metrics_with_checkpoints_delay{delay}.csv"
        df.to_csv(csv_path, index=False)
        print(f"💾 CSV saved to: {csv_path}")
    
    # Save as JSON
    json_path = results_dir / f"multi_model_metrics_with_checkpoints_delay{delay}.json"
    with open(json_path, 'w') as f:
        json.dump(metrics_data, f, indent=2)
    print(f"💾 JSON saved to: {json_path}")
    
    # Save formatted table
    table_path = results_dir / f"multi_model_table_with_checkpoints_delay{delay}.txt"
    with open(table_path, 'w') as f:
        f.write(table_str)
    print(f"💾 Table saved to: {table_path}")
    
    # Save markdown table
    if metrics_data:
        df = pd.DataFrame(metrics_data)
        markdown_path = results_dir / f"multi_model_table_with_checkpoints_delay{delay}.md"
        with open(markdown_path, 'w') as f:
            f.write(f"# Multi-Model Trajectory Analysis Results (with Checkpoints) - Delay: {delay}\n\n")
            f.write(df.to_markdown(index=False, floatfmt=".3f"))
            f.write(f"\n\n*Generated on {time.strftime('%Y-%m-%d %H:%M:%S')}*\n")
        print(f"💾 Markdown saved to: {markdown_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Automated pipeline for multi-model agent trajectory analysis with checkpoint selection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument('--model_run_ids', type=str, required=True,
                      help='Comma-separated list of model run IDs to compare')
    parser.add_argument('--delay', type=int, default=2,
                      help='Human reaction delay to use for all models (default: 2)')
    parser.add_argument('--env_id', type=str, default="SimpleNotiDangerZoneLunarLander",
                      help='Environment ID (default: SimpleNotiDangerZoneLunarLander)')
    parser.add_argument('--human_agent_run_id', type=str, default="xlq34dpt",
                      help='Human agent run ID (default: xlq34dpt)')
    parser.add_argument('--output_base_dir', type=str, default="analysis_results",
                      help='Base output directory (default: analysis_results)')
    parser.add_argument('--human_comprehend_bool', action='store_true', default=True,
                      help='Whether human comprehends notifications (default: True)')
    parser.add_argument('--save_figures', action='store_true', default=False,
                      help='Whether to save analysis figures (default: False)')
    parser.add_argument('--continue_on_error', action='store_true',
                      help='Continue pipeline even if some steps fail')
    parser.add_argument('--num_envs', type=int, default=16,
                      help='Number of environments to use for data collection (default: 16)')
    parser.add_argument('--num_steps', type=int, default=600,
                      help='Number of steps to run for data collection (default: 600)')
    args = parser.parse_args()
    
    # Parse model run IDs
    try:
        model_run_ids = [m.strip() for m in args.model_run_ids.split(',')]
        if not model_run_ids or any(not m for m in model_run_ids):
            raise ValueError("Empty model run ID found")
    except ValueError as e:
        print(f"❌ Invalid model_run_ids format: {e}")
        print("Use comma-separated model IDs (e.g., 'br2gpxv3,zpaw3tkx,w1m1nikn')")
        return
    
    # Configuration
    delay = args.delay
    env_id = args.env_id
    human_agent_run_id = args.human_agent_run_id
    output_dir = f"{args.output_base_dir}/multi_model_checkpoint_analysis_delay{delay}_{int(time.time())}"
    
    print(f"🚀 Starting multi-model analysis pipeline with checkpoint selection")
    print(f"📋 Configuration:")
    print(f"   Model Run IDs: {model_run_ids}")
    print(f"   Environment: {env_id}")
    print(f"   Human Agent: {human_agent_run_id}")
    print(f"   Delay: {delay}")
    print(f"   Output: {output_dir}")
    print(f"{'='*100}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Collect data for different models with checkpoint selection
    print(f"\n📊 STEP 1: DATA COLLECTION WITH CHECKPOINT SELECTION")
    print(f"{'='*60}")
    
    data_dirs = []
    policy_names = []
    checkpoint_infos = []
    successful_models = []
    
    for model_run_id in model_run_ids:
        try:
            data_dir, checkpoint_info = run_collect_data_with_checkpoint_selection(
                env_id=env_id,
                model_run_id=model_run_id,
                human_agent_run_id=human_agent_run_id,
                human_reaction_delay=str(delay),
                human_comprehend_bool=args.human_comprehend_bool,
                num_envs=args.num_envs,
                num_steps=args.num_steps
            )
            data_dirs.append(data_dir)
            policy_names.append(model_run_id)  # Use model ID as policy name
            checkpoint_infos.append(checkpoint_info)
            successful_models.append(model_run_id)
            
        except Exception as e:
            print(f"❌ Error collecting data for model {model_run_id}: {e}")
            if not args.continue_on_error:
                print("💀 Stopping pipeline due to error. Use --continue_on_error to skip failed steps.")
                return
            continue
    
    if not data_dirs:
        print("❌ No data was collected successfully")
        return
    
    print(f"✅ Data collection complete. Collected {len(data_dirs)} datasets.")
    
    # Print checkpoint summary
    print(f"\n🏆 CHECKPOINT SELECTION SUMMARY:")
    print(f"{'='*50}")
    for model_id, checkpoint_info in zip(successful_models, checkpoint_infos):
        print(f"   {model_id}: checkpoint {checkpoint_info.get('checkpoint_num', 'unknown')} "
              f"(reward: {checkpoint_info.get('avg_reward', 'unknown')})")
    
    # Step 2: Run analysis
    print(f"\n📈 STEP 2: TRAJECTORY ANALYSIS")
    print(f"{'='*50}")
    
    try:
        analysis_output = run_analysis_and_capture_metrics(
            data_dirs=data_dirs,
            policy_names=policy_names,
            output_dir=output_dir,
            save_figure=args.save_figures
        )
    except Exception as e:
        print(f"❌ Error running analysis: {e}")
        if not args.continue_on_error:
            return
        analysis_output = ""
    
    # Step 3: Parse metrics and generate table
    print(f"\n📋 STEP 3: METRICS EXTRACTION AND TABLE GENERATION")
    print(f"{'='*60}")
    
    env_type = "DangerZoneLunarLander" if "DangerZoneLunarLander" in env_id else "multi-merge-v0"
    metrics_data = parse_metrics_from_output(analysis_output, policy_names, checkpoint_infos, env_type)
    
    if not metrics_data:
        print("❌ No metrics could be extracted from analysis output")
        print("🔍 Raw analysis output:")
        print(analysis_output[-1000:])  # Show last 1000 chars
        return
    
    # Generate formatted table
    table_str = format_table(metrics_data, delay)
    print(table_str)
    
    # Step 4: Save results
    print(f"\n💾 STEP 4: SAVING RESULTS")
    print(f"{'='*40}")
    
    save_results(metrics_data, table_str, output_dir, delay)
    
    print(f"\n🎉 PIPELINE COMPLETE!")
    print(f"📁 All results saved to: {output_dir}")
    print(f"📊 Compared {len(successful_models)} models with delay {delay}")
    print(f"🏆 Best checkpoints automatically selected for each model")
    print(f"⏱️  Total time: {time.strftime('%H:%M:%S', time.gmtime(time.time()))}")


if __name__ == "__main__":
    main() 