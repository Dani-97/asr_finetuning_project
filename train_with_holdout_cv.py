#!/usr/bin/env python3
"""
Train Whisper model with Holdout + Cross-Validation scheme.

Pipeline:
- Multiple holdout repetitions (each with its own seed)
- For each holdout: train/test split with configurable test size
- Training split → N-fold cross-validation
- Evaluate on test set for each repetition

Output: Metrics per holdout repetition (no separate files)
"""

import os
import sys
import argparse
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils_model import Whisper_Manager
from utils_dataset import Unified_Dataset, create_audio_augmentation
from utils_train import CrossValidationTrainer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Whisper with Holdout + CV scheme"
    )

    parser.add_argument(
        "--language",
        choices=["spanish", "galician", "english"],
        default="spanish",
        help="Language to train on"
    )

    parser.add_argument(
        "--model_name",
        default="openai/whisper-tiny",
        help="Base Whisper model"
    )

    parser.add_argument(
        "--seed_list",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4],
        help="List of seeds for holdout repetitions"
    )
    parser.add_argument(
        "--n_holdouts",
        type=int,
        default=5,
        help="Number of holdout repetitions (should match seed_list length)"
    )
    parser.add_argument(
        "--cv_folds",
        type=int,
        default=5,
        help="Number of cross-validation folds on training split"
    )

    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Size of test set (percentage or absolute number)"
    )
    parser.add_argument(
        "--test_size_is_absolute",
        action="store_true",
        help="If set, test_size is treated as absolute number of samples"
    )

    parser.add_argument(
        "--audio_dir",
        default="./common_voice_dataset",
        help="Directory containing audio files"
    )
    parser.add_argument(
        "--metadata_file_path",
        default=None,
        help="Path to metadata file (TSV for Common Voice, CSV for TORGO). If None, derived from --language"
    )
    parser.add_argument(
        "--output_dir",
        default="./training_results",
        help="Directory to save results"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Max epochs (None for patience-based early stopping)"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Early stopping patience"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Training batch size"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Learning rate"
    )
    parser.add_argument(
        "--max_audio_length",
        type=float,
        default=30.0,
        help="Maximum audio length in seconds"
    )

    parser.add_argument(
        "--augmentation_seed",
        type=int,
        default=42,
        help="Base seed for data augmentation reproducibility"
    )
    parser.add_argument(
        "--use_augmentation",
        action="store_true",
        help="Enable data augmentation during training"
    )
    parser.add_argument(
        "--speed_prob",
        type=float,
        default=0.5,
        help="Probability of applying speed perturbation (0.0-1.0)"
    )
    parser.add_argument(
        "--volume_prob",
        type=float,
        default=0.5,
        help="Probability of applying volume perturbation (0.0-1.0)"
    )
    parser.add_argument(
        "--noise_prob",
        type=float,
        default=0.3,
        help="Probability of applying noise injection (0.0-1.0)"
    )
    parser.add_argument(
        "--time_shift_prob",
        type=float,
        default=0.5,
        help="Probability of applying time shift (0.0-1.0)"
    )
    parser.add_argument(
        "--train_mode",
        choices=["full", "frozen_encoder"],
        default="full",
        help="Training mode: full model or frozen encoder (decoder-only training)"
    )

    return parser.parse_args()


def _print_training_header(args):
    """Print training configuration header."""
    print("=" * 80)
    print("WHISPER TRAINING: HOLDOUT + CROSS-VALIDATION")
    print("=" * 80)
    print(f"Language: {args.language}")
    print(f"Model: {args.model_name}")
    print(f"Train mode: {args.train_mode}")
    print(f"Seed list: {args.seed_list}")
    print(f"Number of holdouts: {args.n_holdouts}")
    print(f"CV folds: {args.cv_folds}")
    print(f"Test size: {args.test_size} ({'absolute' if args.test_size_is_absolute else 'percentage'})")
    print(f"Audio dir: {args.audio_dir}")
    print(f"Output dir: {args.output_dir}")

    if args.use_augmentation:
        print(f"Augmentation: enabled (seed={args.augmentation_seed})")
        print(f"  - Speed prob: {args.speed_prob}")
        print(f"  - Volume prob: {args.volume_prob}")
        print(f"  - Noise prob: {args.noise_prob}")
        print(f"  - Time shift prob: {args.time_shift_prob}")
    else:
        print("Augmentation: disabled")

    print("=" * 80)


def _print_summary(args, all_holdout_results):
    """Print final training summary."""
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)

    if all_holdout_results:
        print(f"Number of holdouts completed: {len(all_holdout_results)}")

        summary = {
            "language": args.language,
            "model": args.model_name,
            "train_mode": args.train_mode,
            "seed_list": args.seed_list[:args.n_holdouts],
            "cv_folds": args.cv_folds,
            "test_size": args.test_size,
            "test_size_is_absolute": args.test_size_is_absolute,
            "num_holdouts_completed": len(all_holdout_results),
            "all_holdout_results": all_holdout_results,
        }

        summary_path = os.path.join(args.output_dir, "training_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSaved summary to: {summary_path}")
    else:
        print("No holdouts completed.")


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    _print_training_header(args)

    temp_manager = Whisper_Manager(
        model_name=args.model_name,
        language=args.language,
    )

    full_dataset = Unified_Dataset(
        processor=temp_manager.processor,
        audio_dir=args.audio_dir,
        language=args.language,
        sample_size=None,
        random_state=args.seed_list[0],
        metadata_file_path=args.metadata_file_path,
    )

    full_df = full_dataset.df[["path", "sentence"]].copy()
    print(f"Loaded dataset with {len(full_df)} samples.")

    del temp_manager

    augmentor = create_audio_augmentation(
        use_augmentation=args.use_augmentation,
        use_speed_perturbation=True,
        use_volume_perturbation=True,
        use_noise_injection=True,
        use_time_shift=True,
        speed_apply_prob=args.speed_prob,
        volume_apply_prob=args.volume_prob,
        noise_apply_prob=args.noise_prob,
        time_shift_apply_prob=args.time_shift_prob,
        seed=args.augmentation_seed,
    )

    trainer = CrossValidationTrainer(
        model_name=args.model_name,
        audio_dir=args.audio_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_audio_length=args.max_audio_length,
        num_epochs=args.epochs,
        patience=args.patience,
        train_mode=args.train_mode,
    )

    all_holdout_results = []

    for holdout_idx in range(min(args.n_holdouts, len(args.seed_list))):
        base_seed = args.seed_list[holdout_idx]

        holdout_result = trainer.train(
            full_df=full_df,
            base_seed=base_seed,
            holdout_idx=holdout_idx,
            n_holdouts=args.n_holdouts,
            cv_folds=args.cv_folds,
            test_size=args.test_size,
            test_size_is_absolute=args.test_size_is_absolute,
            augmentor=augmentor,
            language=args.language,
        )

        all_holdout_results.append(holdout_result)

    _print_summary(args, all_holdout_results)


if __name__ == "__main__":
    main()
