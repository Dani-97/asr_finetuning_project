#!/usr/bin/env python3
"""
Plot training and validation loss from Whisper training history.
Generates individual plots for each fold and a mean plot with standard deviation.
Plots are based on EPOCHS.
"""

import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import glob
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="Plot training loss from CSV history")
    parser.add_argument("--folds_dir", default="./whisper-finetuned", help="Directory containing fold_X subdirectories")
    parser.add_argument("--log_scale", action="store_true", default=False, help="Use logarithmic scale for y-axis")
    return parser.parse_args()

def plot_fold(df, fold_name, output_dir, log_scale=False):
    """Plot training and validation loss for a single fold."""
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if 'train_loss' in df.columns:
        train_data = df[df['train_loss'].notna()]
        ax.plot(train_data['epoch'], train_data['train_loss'], label='Training Loss', marker='o', linewidth=2)
    
    if 'val_loss' in df.columns:
        val_data = df[df['val_loss'].notna()]
        ax.plot(val_data['epoch'], val_data['val_loss'], label='Validation Loss', marker='o', linewidth=2)
    
    if log_scale:
        ax.set_yscale('log')
        
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.set_title(f'Loss Evolution - {fold_name}')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    
    output_path = os.path.join(output_dir, f"{fold_name.replace('/', '_')}_loss.pdf")
    plt.savefig(output_path)
    plt.close(fig)
    
    return output_path, df

def main():
    args = parse_args()
    folds_dir = args.folds_dir
    
    print(f"Looking for training history in {folds_dir}...")
    
    fold_dirs = sorted(glob.glob(os.path.join(folds_dir, "**/fold_*"), recursive=True))
    fold_dirs = [d for d in fold_dirs if os.path.isdir(d) and os.path.exists(os.path.join(d, 'training_history.csv'))]
    
    if not fold_dirs:
        if os.path.exists(os.path.join(folds_dir, "training_history.csv")):
            fold_dirs = [folds_dir]
            print("Found single training run in root directory.")
        else:
            print("No fold directories or training_history.csv found.")
            return
    
    all_train_losses = []
    all_val_losses = []
    
    for fold_path in fold_dirs:
        fold_name = os.path.basename(fold_path)
        if fold_name == ".": 
            fold_name = "single_run"
        
        parent_name = os.path.basename(os.path.dirname(fold_path))
        if parent_name.startswith('holdout'):
            fold_name = f"{parent_name}_{fold_name}"
            
        history_path = os.path.join(fold_path, "training_history.csv")
        
        if not os.path.exists(history_path):
            print(f"Skipping {fold_name}, no history found at {history_path}")
            continue
            
        try:
            df = pd.read_csv(history_path)
            
            out_file, merged_df = plot_fold(df, fold_name, fold_path, args.log_scale)
            print(f"Plotted {fold_name} -> {out_file}")
            
            if 'train_loss' in merged_df.columns:
                train_data = merged_df[['epoch', 'train_loss']].set_index('epoch')
                train_data.columns = [fold_name]
                all_train_losses.append(train_data)
                
            if 'val_loss' in merged_df.columns:
                val_data = merged_df[['epoch', 'val_loss']].set_index('epoch')
                val_data.columns = [fold_name]
                all_val_losses.append(val_data)
                
        except Exception as e:
            print(f"Error processing {fold_name}: {e}")
    
    if all_train_losses or all_val_losses:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if all_train_losses:
            combined_train = pd.concat(all_train_losses, axis=1)
            mean_train = combined_train.mean(axis=1)
            std_train = combined_train.std(axis=1).fillna(0)
            
            ax.plot(mean_train.index, mean_train.values, label='Mean Training Loss', color='blue', linewidth=2, marker='o')
            if len(all_train_losses) > 1:
                ax.fill_between(mean_train.index, 
                                mean_train.values - std_train.values, 
                                mean_train.values + std_train.values, 
                                alpha=0.2, color='blue', label='Train Std Dev')

        if all_val_losses:
            combined_val = pd.concat(all_val_losses, axis=1)
            mean_val = combined_val.mean(axis=1)
            std_val = combined_val.std(axis=1).fillna(0)
            
            ax.plot(mean_val.index, mean_val.values, label='Mean Validation Loss', color='orange', linewidth=2, marker='o')
            if len(all_val_losses) > 1:
                ax.fill_between(mean_val.index, 
                                mean_val.values - std_val.values, 
                                mean_val.values + std_val.values, 
                                alpha=0.2, color='orange', label='Val Std Dev')
        
        if args.log_scale:
            ax.set_yscale('log')
            
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss')
        ax.set_title('Mean Loss Evolution (All Folds)')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        
        output_path = os.path.join(folds_dir, "mean_loss_plot.pdf")
        plt.savefig(output_path)
        plt.close(fig)
        print(f"\nSaved mean plot to {output_path}")
    else:
        print("No data available to plot mean.")

if __name__ == "__main__":
    main()
