#!/usr/bin/env python3
"""
ISRTF Metrics Analyzer

Analyzes ISRTF scheduling metrics and generates visualizations.
[NOTE, hyunnnchoi, 2025.12.07] Created for ISRTF evaluation
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import kendalltau

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def load_metrics(log_dir: str) -> Dict[str, pd.DataFrame]:
    """Load all metric CSV files."""
    metrics = {}
    
    # Load request metrics
    request_csv = os.path.join(log_dir, "request_metrics.csv")
    if os.path.exists(request_csv):
        metrics['requests'] = pd.read_csv(request_csv)
        print(f"✓ Loaded {len(metrics['requests'])} requests")
    
    # Load Kendall's tau
    kendall_csv = os.path.join(log_dir, "kendall_tau.csv")
    if os.path.exists(kendall_csv):
        metrics['kendall'] = pd.read_csv(kendall_csv)
        print(f"✓ Loaded {len(metrics['kendall'])} Kendall's tau measurements")
    
    # Load prediction history
    prediction_csv = os.path.join(log_dir, "prediction_history.csv")
    if os.path.exists(prediction_csv):
        metrics['predictions'] = pd.read_csv(prediction_csv)
        print(f"✓ Loaded {len(metrics['predictions'])} prediction updates")
    
    # Load summary
    summary_json = os.path.join(log_dir, "summary.json")
    if os.path.exists(summary_json):
        with open(summary_json, 'r') as f:
            metrics['summary'] = json.load(f)
        print(f"✓ Loaded summary statistics")
    
    return metrics


def analyze_prediction_accuracy(metrics: Dict[str, pd.DataFrame], output_dir: str):
    """Analyze and visualize prediction accuracy."""
    print("\n📊 Analyzing prediction accuracy...")
    
    df = metrics['requests']
    
    # Filter out inf values
    df_clean = df[df['prediction_error'] != float('inf')].copy()
    
    if len(df_clean) == 0:
        print("⚠️  No valid prediction data")
        return
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Predicted vs Actual scatter plot
    ax = axes[0, 0]
    ax.scatter(df_clean['actual_output_tokens'], df_clean['final_prediction'], 
               alpha=0.5, s=50)
    max_val = max(df_clean['actual_output_tokens'].max(), 
                   df_clean['final_prediction'].max())
    ax.plot([0, max_val], [0, max_val], 'r--', label='Perfect prediction')
    ax.set_xlabel('Actual Output Tokens', fontsize=12)
    ax.set_ylabel('Predicted Remaining Tokens', fontsize=12)
    ax.set_title('Predicted vs Actual Output Length', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add correlation coefficient
    if len(df_clean) >= 2:
        tau, p_value = kendalltau(df_clean['final_prediction'], 
                                   df_clean['actual_output_tokens'])
        ax.text(0.05, 0.95, f"Kendall's τ = {tau:.3f}\np-value = {p_value:.4f}",
                transform=ax.transAxes, fontsize=11,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 2. Error distribution
    ax = axes[0, 1]
    ax.hist(df_clean['prediction_error'], bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(df_clean['prediction_error'].mean(), color='r', 
               linestyle='--', linewidth=2, label=f'Mean: {df_clean["prediction_error"].mean():.1f}')
    ax.axvline(df_clean['prediction_error'].median(), color='g', 
               linestyle='--', linewidth=2, label=f'Median: {df_clean["prediction_error"].median():.1f}')
    ax.set_xlabel('Prediction Error (tokens)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Prediction Error Distribution', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Error rate distribution (percentage)
    ax = axes[1, 0]
    error_rate_pct = df_clean['prediction_error_rate'] * 100
    error_rate_pct = error_rate_pct[error_rate_pct < 200]  # Cap at 200% for visualization
    ax.hist(error_rate_pct, bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(error_rate_pct.mean(), color='r', 
               linestyle='--', linewidth=2, label=f'Mean: {error_rate_pct.mean():.1f}%')
    ax.axvline(error_rate_pct.median(), color='g', 
               linestyle='--', linewidth=2, label=f'Median: {error_rate_pct.median():.1f}%')
    ax.set_xlabel('Prediction Error Rate (%)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Prediction Error Rate Distribution', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Error vs actual length
    ax = axes[1, 1]
    ax.scatter(df_clean['actual_output_tokens'], df_clean['prediction_error'], 
               alpha=0.5, s=50)
    ax.set_xlabel('Actual Output Tokens', fontsize=12)
    ax.set_ylabel('Prediction Error (tokens)', fontsize=12)
    ax.set_title('Prediction Error vs Output Length', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, "prediction_accuracy.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def analyze_kendall_tau(metrics: Dict[str, pd.DataFrame], output_dir: str):
    """Analyze and visualize Kendall's tau over time."""
    print("\n📊 Analyzing Kendall's tau...")
    
    if 'kendall' not in metrics or len(metrics['kendall']) == 0:
        print("⚠️  No Kendall's tau data")
        return
    
    df = metrics['kendall']
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # 1. Kendall's tau over time
    ax = axes[0]
    ax.plot(range(len(df)), df['kendall_tau'], 'b-', linewidth=2, marker='o', markersize=6)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.fill_between(range(len(df)), df['kendall_tau'], 0, alpha=0.3)
    ax.set_xlabel('Measurement Window', fontsize=12)
    ax.set_ylabel("Kendall's τ", fontsize=12)
    ax.set_title("Kendall's Tau Correlation Over Time", fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add mean line
    mean_tau = df['kendall_tau'].mean()
    ax.axhline(y=mean_tau, color='r', linestyle='--', linewidth=2, 
               label=f'Mean τ = {mean_tau:.3f}')
    ax.legend()
    
    # Add interpretation text
    if mean_tau > 0.7:
        interpretation = "Strong positive correlation ✓"
        color = 'green'
    elif mean_tau > 0.4:
        interpretation = "Moderate positive correlation"
        color = 'orange'
    else:
        interpretation = "Weak correlation"
        color = 'red'
    
    ax.text(0.02, 0.98, interpretation,
            transform=ax.transAxes, fontsize=12, fontweight='bold',
            verticalalignment='top', color=color,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 2. Prediction error over time
    ax = axes[1]
    ax.plot(range(len(df)), df['mean_prediction_error'], 'r-', 
            linewidth=2, marker='s', markersize=6, label='Mean Error')
    ax.plot(range(len(df)), df['median_prediction_error'], 'g-', 
            linewidth=2, marker='^', markersize=6, label='Median Error')
    ax.set_xlabel('Measurement Window', fontsize=12)
    ax.set_ylabel('Prediction Error (tokens)', fontsize=12)
    ax.set_title('Prediction Error Over Time', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, "kendall_tau_analysis.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def analyze_latency(metrics: Dict[str, pd.DataFrame], output_dir: str):
    """Analyze and visualize latency metrics."""
    print("\n📊 Analyzing latency...")
    
    df = metrics['requests']
    df_clean = df[df['total_latency'].notna()].copy()
    
    if len(df_clean) == 0:
        print("⚠️  No latency data")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Total latency distribution
    ax = axes[0, 0]
    ax.hist(df_clean['total_latency'], bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(df_clean['total_latency'].mean(), color='r', 
               linestyle='--', linewidth=2, label=f'Mean: {df_clean["total_latency"].mean():.2f}s')
    ax.axvline(df_clean['total_latency'].median(), color='g', 
               linestyle='--', linewidth=2, label=f'Median: {df_clean["total_latency"].median():.2f}s')
    ax.set_xlabel('Total Latency (seconds)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Total Latency Distribution', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Queue wait time distribution
    ax = axes[0, 1]
    ax.hist(df_clean['queue_wait_time'], bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(df_clean['queue_wait_time'].mean(), color='r', 
               linestyle='--', linewidth=2, label=f'Mean: {df_clean["queue_wait_time"].mean():.2f}s')
    ax.set_xlabel('Queue Wait Time (seconds)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Queue Wait Time Distribution', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. TTFT distribution
    df_ttft = df_clean[df_clean['time_to_first_token'].notna()]
    if len(df_ttft) > 0:
        ax = axes[1, 0]
        ax.hist(df_ttft['time_to_first_token'], bins=50, edgecolor='black', alpha=0.7)
        ax.axvline(df_ttft['time_to_first_token'].mean(), color='r', 
                   linestyle='--', linewidth=2, label=f'Mean: {df_ttft["time_to_first_token"].mean():.2f}s')
        ax.set_xlabel('Time to First Token (seconds)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Time to First Token (TTFT) Distribution', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 4. Latency vs output length
    ax = axes[1, 1]
    ax.scatter(df_clean['actual_output_tokens'], df_clean['total_latency'], 
               alpha=0.5, s=50)
    ax.set_xlabel('Actual Output Tokens', fontsize=12)
    ax.set_ylabel('Total Latency (seconds)', fontsize=12)
    ax.set_title('Latency vs Output Length', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, "latency_analysis.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def print_summary(metrics: Dict[str, pd.DataFrame]):
    """Print summary statistics."""
    print("\n" + "="*60)
    print("  ISRTF Metrics Summary")
    print("="*60)
    
    if 'summary' in metrics:
        summary = metrics['summary']
        print(f"\n📊 Total Requests: {summary.get('total_requests', 0)}")
        
        if 'prediction_accuracy' in summary:
            acc = summary['prediction_accuracy']
            print(f"\n🎯 Prediction Accuracy:")
            print(f"  Mean Error: {acc.get('mean_error', 0):.2f} tokens")
            print(f"  Median Error: {acc.get('median_error', 0):.2f} tokens")
            print(f"  Std Error: {acc.get('std_error', 0):.2f} tokens")
            if acc.get('mean_error_rate') is not None:
                print(f"  Mean Error Rate: {acc['mean_error_rate']*100:.1f}%")
                print(f"  Median Error Rate: {acc['median_error_rate']*100:.1f}%")
        
        if 'kendall_tau' in summary:
            kt = summary['kendall_tau']
            print(f"\n📈 Kendall's Tau Correlation:")
            print(f"  τ = {kt.get('tau', 0):.4f}")
            print(f"  p-value = {kt.get('p_value', 0):.4f}")
            print(f"  Sample Size = {kt.get('sample_size', 0)}")
            
            # Interpretation
            tau = kt.get('tau', 0)
            if tau > 0.7:
                print(f"  → Strong positive correlation ✓")
            elif tau > 0.4:
                print(f"  → Moderate positive correlation")
            elif tau > 0.2:
                print(f"  → Weak positive correlation")
            else:
                print(f"  → Very weak or no correlation")
        
        if 'latency' in summary:
            lat = summary['latency']
            print(f"\n⏱️  Latency Metrics:")
            if lat.get('mean') is not None:
                print(f"  Mean: {lat['mean']:.2f}s")
                print(f"  Median: {lat['median']:.2f}s")
                print(f"  P95: {lat['p95']:.2f}s")
                print(f"  P99: {lat['p99']:.2f}s")
    
    print("\n" + "="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Analyze ISRTF metrics")
    parser.add_argument(
        "--log-dir",
        type=str,
        default="/tmp/isrtf_metrics",
        help="Directory containing ISRTF metric logs"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save analysis plots (default: same as log-dir)"
    )
    
    args = parser.parse_args()
    
    # Set output directory
    output_dir = args.output_dir if args.output_dir else args.log_dir
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n📂 Loading metrics from: {args.log_dir}")
    print(f"💾 Saving analysis to: {output_dir}\n")
    
    # Load metrics
    metrics = load_metrics(args.log_dir)
    
    if not metrics:
        print("❌ No metrics found!")
        return
    
    # Generate analyses
    if 'requests' in metrics:
        analyze_prediction_accuracy(metrics, output_dir)
        analyze_latency(metrics, output_dir)
    
    if 'kendall' in metrics:
        analyze_kendall_tau(metrics, output_dir)
    
    # Print summary
    print_summary(metrics)
    
    print("✅ Analysis complete!")
    print(f"\n📁 Check output directory: {output_dir}")


if __name__ == "__main__":
    main()

