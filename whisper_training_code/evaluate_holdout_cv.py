#!/usr/bin/env python3
"""
Evaluate Whisper models on test sets.

Supports both HuggingFace models and locally trained models.
Supports Common Voice and TORGO datasets.
"""

import os
import sys
import argparse
import json
import pandas as pd
import numpy as np
import torch
import glob

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils_dataset import Unified_Dataset
from utils_model import Local_Whisper_Manager, Whisper_Manager


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate Whisper models on test sets"
    )
    parser.add_argument(
        "--language",
        choices=["spanish", "galician", "english"],
        default="english",
        help="Language of the models to evaluate"
    )
    parser.add_argument(
        "--audio_dir",
        default="./torgo_dataset/wav_files",
        help="Directory containing audio files"
    )
    parser.add_argument(
        "--metadata_file_path",
        default="./torgo_dataset/metadata.csv",
        help="Path to metadata file (TSV for Common Voice, CSV for TORGO)"
    )
    parser.add_argument(
        "--dataset_type",
        choices=["common_voice", "torgo", "amicos"],
        default="torgo",
        help="Type of dataset: 'common_voice' for CV TSV files, 'torgo' for TORGO CSV files, 'amicos' for AMICOS dataset"
    )
    parser.add_argument(
        "--input_dir",
        default=None,
        help="Input directory for AMICOS dataset (contains audio/ and transcriptions.csv)"
    )
    parser.add_argument(
        "--model_source",
        choices=["local", "huggingface"],
        default="huggingface",
        help="Source of the model: 'local' for trained models, 'huggingface' for HF models"
    )
    parser.add_argument(
        "--model_name_or_path",
        required=True,
        help="Path to local model, HuggingFace model name, or training results directory (when using --trained_models_dir)"
    )
    parser.add_argument(
        "--trained_models_dir",
        default=None,
        help="Directory containing trained models from training_with_holdout_cv.py. When set, evaluates all holdout/fold models"
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Test set size (fraction or absolute number if --test_size_is_absolute)"
    )
    parser.add_argument(
        "--test_size_is_absolute",
        action="store_true",
        help="If set, test_size is treated as absolute number of samples"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for GPU evaluation (higher = faster but uses more GPU memory)"
    )
    parser.add_argument(
        "--output_dir",
        default="./evaluation_results",
        help="Directory to save evaluation results"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/test split"
    )
    parser.add_argument(
        "--seed_list",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4, 5],
        help="List of seeds for multiple holdout evaluations"
    )
    parser.add_argument(
        "--n_holdouts",
        type=int,
        default=5,
        help="Number of holdout repetitions to evaluate"
    )
    return parser.parse_args()


def load_test_dataset(metadata_file_path, audio_dir, language, random_state=42, dataset_type="common_voice"):
    """Load test dataset using Unified_Dataset."""
    temp_manager = Whisper_Manager(
        model_name="openai/whisper-tiny",
        language=language,
    )
    
    full_dataset = Unified_Dataset(
        processor=temp_manager.processor,
        audio_dir=audio_dir,
        language=language,
        sample_size=None,
        random_state=random_state,
        metadata_file_path=metadata_file_path,
    )
    
    del temp_manager
    
    if dataset_type == "torgo":
        return full_dataset.df
    else:
        return full_dataset.df[["path", "sentence"]]


def create_whisper_manager(model_source, model_name_or_path, language):
    """Factory function to create the appropriate Whisper manager."""
    if model_source == "huggingface":
        print(f"  Creating HuggingFace manager for: {model_name_or_path}")
        return Whisper_Manager(
            model_name=model_name_or_path,
            language=language,
        )
    else:
        print(f"  Creating Local manager for: {model_name_or_path}")
        return Local_Whisper_Manager(
            local_model_path=model_name_or_path,
            language=language,
        )


def find_trained_models(trained_models_dir):
    """Find all trained model checkpoints in a training results directory.
    
    Returns a list of tuples: (holdout_idx, fold_idx, model_path, seed)
    Sorted by holdout_idx then fold_idx.
    """
    import re
    
    models = []
    if not os.path.isdir(trained_models_dir):
        return models
    
    for entry in os.listdir(trained_models_dir):
        holdout_match = re.match(r"holdout_(\d+)_seed_(\d+)", entry)
        if holdout_match:
            holdout_idx = int(holdout_match.group(1))
            seed = int(holdout_match.group(2))
            holdout_dir = os.path.join(trained_models_dir, entry)
            
            if os.path.isdir(holdout_dir):
                for fold_entry in os.listdir(holdout_dir):
                    fold_match = re.match(r"fold_(\d+)", fold_entry)
                    if fold_match:
                        fold_idx = int(fold_match.group(1))
                        fold_dir = os.path.join(holdout_dir, fold_entry)
                        
                        safetensors_path = os.path.join(fold_dir, "model.safetensors")
                        if os.path.exists(safetensors_path):
                            models.append({
                                "holdout_idx": holdout_idx,
                                "fold_idx": fold_idx,
                                "seed": seed,
                                "model_path": safetensors_path,
                                "holdout_dir": holdout_dir,
                                "fold_dir": fold_dir,
                            })
    
    models.sort(key=lambda x: (x["holdout_idx"], x["fold_idx"]))
    return models


class EvaluationStrategy:
    """Base class for evaluation strategies."""
    
    def __init__(self, args, full_df):
        self.args = args
        self.full_df = full_df
        self.results = []
    
    def get_test_df(self):
        """Split dataset into train/test and return test DataFrame."""
        from sklearn.model_selection import train_test_split
        
        test_size = self.args.test_size
        if self.args.test_size_is_absolute:
            test_size = self.args.test_size / len(self.full_df)
        
        _, test_indices = train_test_split(
            range(len(self.full_df)),
            test_size=test_size,
            random_state=self.args.seed,
        )
        
        return self.full_df.iloc[test_indices].reset_index(drop=True)
    
    def evaluate(self):
        """Run evaluation on a single test set."""
        raise NotImplementedError("evaluate() must be implemented by subclass")
    
    def save_results(self, results_df):
        """Save evaluation results."""
        raise NotImplementedError("save_results() must be implemented by subclass")
    
    def evaluate_baseline_model(self):
        """Evaluate baseline/pretrained model (supports multiple holdouts)."""
        raise NotImplementedError("evaluate_baseline_model() is not implemented for this dataset type")
    
    def evaluate_finetuned_models(self, models):
        """Evaluate fine-tuned models."""
        raise NotImplementedError("evaluate_finetuned_models() is not implemented for this dataset type")


class CommonVoiceEvaluationStrategy(EvaluationStrategy):
    """Evaluation strategy for Common Voice datasets."""

    def __init__(self, args, full_df, trained_model_path=None):
        super().__init__(args, full_df)
        self.trained_model_path = trained_model_path

    def evaluate(self):
        """Evaluate on Common Voice dataset."""
        print(f"\n{'='*60}")
        print(f"Evaluating Common Voice dataset")
        print(f"{'='*60}")

        test_df = self.get_test_df()
        print(f"  Test samples: {len(test_df)}")

        if self.trained_model_path:
            print(f"  Loading trained model: {self.trained_model_path}")
            manager = Local_Whisper_Manager(
                local_model_path=self.trained_model_path,
                base_model_name=self.args.model_name_or_path,
                language=self.args.language,
            )
        else:
            manager = create_whisper_manager(
                self.args.model_source,
                self.args.model_name_or_path,
                self.args.language,
            )

        results = manager.evaluate_model(
            test_df,
            audio_dir=self.args.audio_dir,
            verbose=True,
            batch_size=getattr(self.args, 'batch_size', 8),
        )

        predictions_df = results["results_df"].copy()
        try:
            wer_mean = predictions_df["wer"].mean() if "wer" in predictions_df.columns and len(predictions_df) > 0 else 0
        except (KeyError, TypeError, AttributeError):
            wer_mean = results.get('wer_mean', 0)
        try:
            wer_ci_mean = predictions_df["wer_ci"].mean() if "wer_ci" in predictions_df.columns and len(predictions_df) > 0 else 0
        except (KeyError, TypeError, AttributeError):
            wer_ci_mean = results.get('wer_ci_mean', 0)

        print(f"  WER (mean): {wer_mean:.4f}")
        print(f"  WER (case-insensitive, mean): {wer_ci_mean:.4f}")

        del manager
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.results = [{
            "model_source": "local" if self.trained_model_path else self.args.model_source,
            "model_name_or_path": self.trained_model_path or self.args.model_name_or_path,
            "wer_mean": wer_mean,
            "wer_ci_mean": wer_ci_mean,
            "num_samples": results['num_samples'],
        }]

        return predictions_df

    def save_results(self, results_df):
        """Save Common Voice evaluation results with detailed predictions."""
        os.makedirs(self.args.output_dir, exist_ok=True)

        summary_path = os.path.join(self.args.output_dir, "evaluation_summary.json")
        summary = {
            "model_source": "local" if self.trained_model_path else self.args.model_source,
            "model_name_or_path": self.trained_model_path or self.args.model_name_or_path,
            "language": self.args.language,
            "num_samples": len(results_df),
            "wer_mean": float(results_df["wer"].mean()) if "wer" in results_df.columns else None,
            "wer_ci_mean": float(results_df["wer_ci"].mean()) if "wer_ci" in results_df.columns else None,
        }

        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        predictions_path = os.path.join(self.args.output_dir, "detailed_predictions.csv")
        results_df.to_csv(predictions_path, index=False)

        print(f"  Saved summary to: {summary_path}")
        print(f"  Saved detailed predictions to: {predictions_path}")
    
    def evaluate_finetuned_models(self, models):
        """Evaluate fine-tuned models (Common Voice: multiple holdouts)."""
        models_by_seed = {}
        for model in models:
            seed = model["seed"]
            if seed not in models_by_seed:
                models_by_seed[seed] = []
            models_by_seed[seed].append(model)
        
        seeds = sorted(models_by_seed.keys())
        base_output_dir = self.args.output_dir
        
        all_predictions = []
        all_results = []
        
        for seed_idx, seed in enumerate(seeds):
            seed_models = models_by_seed[seed]
            holdout_dir = seed_models[0]["holdout_dir"]
            holdout_output_dir = os.path.join(
                base_output_dir,
                f"holdout_{seed_idx + 1}_seed_{seed}"
            )
            
            print(f"\n{'='*80}")
            print(f"HOLDOUT {seed_idx + 1}/{len(seeds)} (seed={seed})")
            print(f"{'='*80}")
            
            self.args.output_dir = holdout_output_dir
            self.args.seed = seed
            
            best_cv_fold = None
            holdout_results_path = os.path.join(holdout_dir, "holdout_results.json")
            if os.path.exists(holdout_results_path):
                with open(holdout_results_path, "r") as f:
                    holdout_results = json.load(f)
                    best_cv_fold = holdout_results.get("best_cv_fold")
                    print(f"Best CV fold from training: {best_cv_fold}")
            
            if best_cv_fold is None:
                print("Warning: Could not find best CV fold, using fold 1")
                best_cv_fold = 1
            
            best_model_info = None
            for model_info in seed_models:
                if model_info["fold_idx"] == best_cv_fold:
                    best_model_info = model_info
                    break
            
            if best_model_info is None:
                best_model_info = seed_models[0]
            
            fold_idx = best_model_info["fold_idx"]
            model_path = best_model_info["model_path"]
            
            print(f"\n--- Evaluating best model from fold {fold_idx} ---")
            print(f"Model: {model_path}")
            
            self.trained_model_path = model_path
            results_df = self.evaluate()
            
            if not results_df.empty:
                results_df["holdout_idx"] = seed_idx + 1
                results_df["seed"] = seed
                results_df["fold_idx"] = fold_idx
                all_predictions.append(results_df)
                
                try:
                    wer_mean = results_df["wer"].mean()
                except (KeyError, TypeError):
                    wer_mean = 0
                try:
                    wer_ci_mean = results_df["wer_ci"].mean() if "wer_ci" in results_df.columns and len(results_df) > 0 else 0
                except (KeyError, TypeError):
                    wer_ci_mean = 0
                print(f"Average WER: {wer_mean:.4f}")
                print(f"Average WER (case-insensitive): {wer_ci_mean:.4f}")
                
                self.save_results(results_df)
            
            all_results.append({
                "holdout_idx": seed_idx + 1,
                "seed": seed,
                "fold_idx": fold_idx,
                "results": self.results,
            })
        
        if all_predictions:
            all_predictions_df = pd.concat(all_predictions, ignore_index=True)
            final_output_dir = base_output_dir
            
            os.makedirs(final_output_dir, exist_ok=True)
            all_predictions_path = os.path.join(final_output_dir, "all_predictions.csv")
            all_predictions_df.to_csv(all_predictions_path, index=False)
            print(f"\nSaved all predictions to: {all_predictions_path}")
            
            test_wers = [r["results"][0]["wer_mean"] for r in all_results if r["results"]]
            test_wers_ci = [r["results"][0].get("wer_ci_mean", 0) for r in all_results if r["results"]]
            print(f"\n{'='*80}")
            print("FINAL SUMMARY ACROSS ALL HOLDOUTS")
            print(f"{'='*80}")
            print(f"Number of holdouts: {len(test_wers)}")
            print(f"WER: mean={np.mean(test_wers):.4f}, std={np.std(test_wers):.4f}, min={np.min(test_wers):.4f}, max={np.max(test_wers):.4f}")
            print(f"WER (case-insensitive): mean={np.mean(test_wers_ci):.4f}, std={np.std(test_wers_ci):.4f}, min={np.min(test_wers_ci):.4f}, max={np.max(test_wers_ci):.4f}")
            
            summary = {
                "language": self.args.language,
                "model_source": "local",
                "model_name_or_path": self.args.trained_models_dir if hasattr(self.args, 'trained_models_dir') else self.args.model_name_or_path,
                "seed_list": seeds,
                "num_holdouts_completed": len(all_results),
                "wer_mean": float(np.mean(test_wers)),
                "wer_std": float(np.std(test_wers)),
                "wer_min": float(np.min(test_wers)),
                "wer_max": float(np.max(test_wers)),
                "wer_ci_mean": float(np.mean(test_wers_ci)),
                "wer_ci_std": float(np.std(test_wers_ci)),
                "wer_ci_min": float(np.min(test_wers_ci)),
                "wer_ci_max": float(np.max(test_wers_ci)),
                "holdout_results": all_results,
            }
            
            summary_path = os.path.join(final_output_dir, "evaluation_summary.json")
            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2)
            print(f"Saved summary to: {summary_path}")
        
        print("\n" + "=" * 80)
        print("EVALUATION COMPLETE")
        print("=" * 80)
    
    def evaluate_baseline_model(self):
        """Evaluate pretrained/baseline model (Common Voice: multiple holdouts)."""
        base_output_dir = self.args.output_dir
        
        all_holdout_results = []
        all_predictions = []
        
        for holdout_idx in range(min(self.args.n_holdouts, len(self.args.seed_list))):
            base_seed = self.args.seed_list[holdout_idx]
            
            holdout_output_dir = os.path.join(
                base_output_dir,
                f"holdout_{holdout_idx + 1}_seed_{base_seed}"
            )
            self.args.output_dir = holdout_output_dir
            self.args.seed = base_seed
            
            print(f"\n{'='*80}")
            print(f"HOLDOUT {holdout_idx + 1}/{self.args.n_holdouts} (seed={base_seed})")
            print(f"{'='*80}")
            
            results_df = self.evaluate()
            
            if not results_df.empty:
                results_df["holdout_idx"] = holdout_idx + 1
                results_df["seed"] = base_seed
                all_predictions.append(results_df)
                
                print("\n" + "=" * 80)
                print("SUMMARY")
                print("=" * 80)
                
                wer_mean = results_df["wer"].mean() if "wer" in results_df.columns and len(results_df) > 0 else 0
                wer_ci_mean = results_df["wer_ci"].mean() if "wer_ci" in results_df.columns and len(results_df) > 0 else 0
                print(f"Average WER: {wer_mean:.4f}")
                print(f"Average WER (case-insensitive): {wer_ci_mean:.4f}")
                wer_min = results_df["wer"].min() if "wer" in results_df.columns else 0
                wer_max = results_df["wer"].max() if "wer" in results_df.columns else 0
                print(f"Min WER: {wer_min:.4f}")
                print(f"Max WER: {wer_max:.4f}")
                
                self.save_results(results_df)
            
            all_holdout_results.append({
                "holdout_idx": holdout_idx + 1,
                "seed": base_seed,
                "results": self.results,
            })
        
        if all_predictions:
            all_predictions_df = pd.concat(all_predictions, ignore_index=True)
            final_output_dir = base_output_dir
            
            os.makedirs(final_output_dir, exist_ok=True)
            all_predictions_path = os.path.join(final_output_dir, "all_predictions.csv")
            all_predictions_df.to_csv(all_predictions_path, index=False)
            print(f"\nSaved all predictions to: {all_predictions_path}")
            
            test_wers = [r["results"][0]["wer_mean"] for r in all_holdout_results if r["results"]]
            test_wers_ci = [r["results"][0].get("wer_ci_mean", 0) for r in all_holdout_results if r["results"]]
            if test_wers:
                print(f"\n{'='*80}")
                print("FINAL SUMMARY ACROSS ALL HOLDOUTS")
                print(f"{'='*80}")
                print(f"Number of holdouts: {len(test_wers)}")
                print(f"WER: mean={np.mean(test_wers):.4f}, std={np.std(test_wers):.4f}, min={np.min(test_wers):.4f}, max={np.max(test_wers):.4f}")
                print(f"WER (case-insensitive): mean={np.mean(test_wers_ci):.4f}, std={np.std(test_wers_ci):.4f}, min={np.min(test_wers_ci):.4f}, max={np.max(test_wers_ci):.4f}")
                
                summary = {
                    "language": self.args.language,
                    "model_source": self.args.model_source,
                    "model_name_or_path": self.args.model_name_or_path,
                    "seed_list": self.args.seed_list[:self.args.n_holdouts],
                    "num_holdouts_completed": len(all_holdout_results),
                    "wer_mean": float(np.mean(test_wers)),
                    "wer_std": float(np.std(test_wers)),
                    "wer_min": float(np.min(test_wers)),
                    "wer_max": float(np.max(test_wers)),
                    "wer_ci_mean": float(np.mean(test_wers_ci)),
                    "wer_ci_std": float(np.std(test_wers_ci)),
                    "wer_ci_min": float(np.min(test_wers_ci)),
                    "wer_ci_max": float(np.max(test_wers_ci)),
                    "holdout_results": all_holdout_results,
                }
                
                summary_path = os.path.join(final_output_dir, "evaluation_summary.json")
                with open(summary_path, "w") as f:
                    json.dump(summary, f, indent=2)
                print(f"Saved summary to: {summary_path}")
        
        print("\n" + "=" * 80)
        print("EVALUATION COMPLETE")
        print("=" * 80)


class TorgoEvaluationStrategy(EvaluationStrategy):
    """Evaluation strategy for TORGO dataset with per-session evaluation."""
    
    def __init__(self, args, full_df, trained_model_path=None):
        super().__init__(args, full_df)
        self.trained_model_path = trained_model_path
        self.results_dir = getattr(args, 'results_dir', './training_results')
    
    def evaluate(self):
        """Evaluate TORGO dataset per (person_id, session) group."""
        print(f"\n{'='*60}")
        print(f"Evaluating TORGO dataset (per-session)")
        print(f"{'='*60}")
        
        groups = self.full_df.groupby(["person_id", "session"])
        print(f"  Found {len(groups)} (person, session) groups")
        
        if self.trained_model_path:
            print(f"  Loading trained model: {self.trained_model_path}")
            manager = Local_Whisper_Manager(
                local_model_path=self.trained_model_path,
                base_model_name=self.args.model_name_or_path,
                language=self.args.language,
            )
        else:
            manager = create_whisper_manager(
                self.args.model_source,
                self.args.model_name_or_path,
                self.args.language,
            )
        
        all_predictions = []
        session_results = []
        
        for (person_id, session), group_df in groups:
            print(f"\n  Evaluating {person_id} - {session} ({len(group_df)} samples)")
            
            test_df = group_df.reset_index(drop=True)
            
            results = manager.evaluate_model(
                test_df,
                audio_dir=self.args.audio_dir,
                verbose=True,
                batch_size=getattr(self.args, 'batch_size', 8),
            )
            
            gender = test_df["gender"].iloc[0] if "gender" in test_df.columns else ""
            condition = test_df["condition"].iloc[0] if "condition" in test_df.columns else ""
            
            predictions_df = results["results_df"].copy()
            wer_mean = predictions_df["wer"].mean() if "wer" in predictions_df.columns and len(predictions_df) > 0 else results['wer_mean']
            wer_ci_mean = predictions_df["wer_ci"].mean() if "wer_ci" in predictions_df.columns and len(predictions_df) > 0 else results.get('wer_ci_mean', 0)
            
            session_results.append({
                "person_id": person_id,
                "gender": gender,
                "session": session,
                "condition": condition,
                "wer_mean": wer_mean,
                "wer_ci_mean": wer_ci_mean,
                "num_samples": results['num_samples'],
            })
            
            predictions_df["person_id"] = person_id
            predictions_df["session"] = session
            predictions_df["gender"] = gender
            predictions_df["condition"] = condition
            all_predictions.append(predictions_df)
            
            print(f"    WER (mean): {wer_mean:.4f}")
            print(f"    WER (case-insensitive, mean): {wer_ci_mean:.4f}")
        
        del manager
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.results = session_results
        
        if all_predictions:
            return pd.concat(all_predictions, ignore_index=True)
        return pd.DataFrame()
    
    def save_results(self, results_df):
        """Save TORGO evaluation results with per-session breakdown."""
        os.makedirs(self.args.output_dir, exist_ok=True)
        
        session_results_df = pd.DataFrame(self.results)
        
        summary_path = os.path.join(self.args.output_dir, "evaluation_summary.json")
        wer_mean = float(session_results_df["wer_mean"].mean()) if not session_results_df.empty else None
        wer_ci_mean = float(session_results_df["wer_ci_mean"].mean()) if "wer_ci_mean" in session_results_df.columns and not session_results_df.empty else None
        summary = {
            "model_source": "local" if self.trained_model_path else self.args.model_source,
            "model_name_or_path": self.trained_model_path or self.args.model_name_or_path,
            "language": self.args.language,
            "num_sessions": len(session_results_df),
            "wer_mean": wer_mean,
            "wer_ci_mean": wer_ci_mean,
            "by_person": session_results_df.to_dict(orient="records"),
        }
        
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        
        session_results_path = os.path.join(self.args.output_dir, "session_results.csv")
        session_results_df.to_csv(session_results_path, index=False)
        
        if not results_df.empty:
            predictions_path = os.path.join(self.args.output_dir, "detailed_predictions.csv")
            results_df.to_csv(predictions_path, index=False)
            print(f"  Saved predictions to: {predictions_path}")
        
        print(f"  Saved session results to: {session_results_path}")
        print(f"  Saved summary to: {summary_path}")
    
    def evaluate_finetuned_models(self, models):
        """Evaluate fine-tuned models (TORGO: single evaluation on all sessions, pick best model)."""
        os.makedirs(self.args.output_dir, exist_ok=True)
        
        models_by_seed = {}
        for model in models:
            seed = model["seed"]
            if seed not in models_by_seed:
                models_by_seed[seed] = []
            models_by_seed[seed].append(model)
        
        seeds = sorted(models_by_seed.keys())
        
        best_overall_wer = float("inf")
        best_seed = None
        best_fold = None
        best_predictions_df = None
        best_results = None
        
        print(f"\n{'='*80}")
        print("TORGO EVALUATION (fine-tuned models)")
        print(f"{'='*80}")
        
        for seed in seeds:
            seed_models = models_by_seed[seed]
            holdout_dir = seed_models[0]["holdout_dir"]
            
            best_cv_fold = None
            holdout_results_path = os.path.join(holdout_dir, "holdout_results.json")
            if os.path.exists(holdout_results_path):
                with open(holdout_results_path, "r") as f:
                    holdout_results = json.load(f)
                    best_cv_fold = holdout_results.get("best_cv_fold")
            
            if best_cv_fold is None:
                best_cv_fold = 1
            
            best_model_info = None
            for model_info in seed_models:
                if model_info["fold_idx"] == best_cv_fold:
                    best_model_info = model_info
                    break
            
            if best_model_info is None:
                best_model_info = seed_models[0]
            
            fold_idx = best_model_info["fold_idx"]
            model_path = best_model_info["model_path"]
            
            print(f"\n--- Evaluating seed={seed}, fold={fold_idx} ---")
            
            self.trained_model_path = model_path
            results_df = self.evaluate()
            
            if not results_df.empty:
                overall_wer = results_df["wer"].mean() if "wer" in results_df.columns and len(results_df) > 0 else 0.0
                print(f"\n  Seed {seed} - Overall WER: {overall_wer:.4f}")
                
                if overall_wer < best_overall_wer:
                    best_overall_wer = overall_wer
                    best_seed = seed
                    best_fold = fold_idx
                    best_predictions_df = results_df
                    best_results = self.results
        
        print(f"\n{'='*80}")
        print("FINAL SUMMARY")
        print(f"{'='*80}")
        print(f"Best model: seed={best_seed}, fold={best_fold}")
        print(f"WER: {best_overall_wer:.4f}")
        
        if best_predictions_df is not None:
            best_predictions_df["seed"] = best_seed
            best_predictions_df["fold_idx"] = best_fold
            
            predictions_path = os.path.join(self.args.output_dir, "detailed_predictions.csv")
            best_predictions_df.to_csv(predictions_path, index=False)
            print(f"Saved predictions to: {predictions_path}")
            
            if best_results:
                session_results_df = pd.DataFrame(best_results)
                session_results_path = os.path.join(self.args.output_dir, "session_results.csv")
                session_results_df.to_csv(session_results_path, index=False)
                print(f"Saved session results to: {session_results_path}")
        
        summary = {
            "language": self.args.language,
            "model_source": "local",
            "model_name_or_path": self.args.trained_models_dir if hasattr(self.args, 'trained_models_dir') else self.args.model_name_or_path,
            "seed_list": seeds,
            "best_seed": best_seed,
            "best_fold": best_fold,
            "wer": best_overall_wer,
            "session_results": best_results,
        }
        
        summary_path = os.path.join(self.args.output_dir, "evaluation_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Saved summary to: {summary_path}")
        
        print("\n" + "=" * 80)
        print("EVALUATION COMPLETE")
        print("=" * 80)
    
    def evaluate_baseline_model(self):
        """Evaluate baseline/pretrained model (TORGO: evaluate all sessions once)."""
        os.makedirs(self.args.output_dir, exist_ok=True)
        
        print(f"\n{'='*80}")
        print("TORGO EVALUATION (pretrained/original model)")
        print(f"{'='*80}")
        
        results_df = self.evaluate()

        if not results_df.empty:
            session_results_df = pd.DataFrame(self.results)
            if not session_results_df.empty:
                print(f"\nPer-session results:")
                print(session_results_df.to_string(index=False))
                wer_mean = session_results_df['wer_mean'].mean()
                wer_ci_mean = session_results_df['wer_ci_mean'].mean() if 'wer_ci_mean' in session_results_df.columns else 0
                print(f"\nOverall WER: {wer_mean:.4f}")
                print(f"Overall WER (case-insensitive): {wer_ci_mean:.4f}")

            self.save_results(results_df)
        
        print("\n" + "=" * 80)
        print("EVALUATION COMPLETE")
        print("=" * 80)


class AmicosEvaluationStrategy(EvaluationStrategy):
    """Evaluation strategy for AMICOS dataset with per-file evaluation."""
    
    def __init__(self, args, trained_model_path=None):
        self.args = args
        self.trained_model_path = trained_model_path
        self.results = []
        self.full_df = None
        
    def load_transcriptions(self):
        """Load transcriptions.csv from input_dir."""
        if self.args.dataset_type != "amicos":
            return None
            
        if self.args.input_dir:
            transcription_file = os.path.join(self.args.input_dir, "transcriptions.csv")
        else:
            transcription_file = self.args.metadata_file_path
            
        if not os.path.exists(transcription_file):
            raise FileNotFoundError(f"Transcription file not found: {transcription_file}")
            
        df = pd.read_csv(transcription_file)
        return df
    
    def get_audio_path(self, audio_filename):
        """Get full path to audio file."""
        if self.args.audio_dir:
            return os.path.join(self.args.audio_dir, audio_filename)
        elif self.args.input_dir:
            return os.path.join(self.args.input_dir, "audio", audio_filename)
        return audio_filename
    
    def get_test_df(self):
        """Return full DataFrame for AMICOS (evaluate all files)."""
        return self.full_df
    
    def evaluate(self):
        """Evaluate AMICOS dataset file by file."""
        print(f"\n{'='*60}")
        print(f"Evaluating AMICOS dataset (per-file)")
        print(f"{'='*60}")
        
        self.full_df = self.load_transcriptions()
        if self.full_df is None:
            raise ValueError("Not an AMICOS dataset")
        
        print(f"  Found {len(self.full_df)} audio files to evaluate")
        
        if self.trained_model_path:
            print(f"  Loading trained model: {self.trained_model_path}")
            manager = Local_Whisper_Manager(
                local_model_path=self.trained_model_path,
                base_model_name=self.args.model_name_or_path,
                language=self.args.language,
            )
        else:
            manager = create_whisper_manager(
                self.args.model_source,
                self.args.model_name_or_path,
                self.args.language,
            )
        
        all_predictions = []
        person_results = {}
        
        for idx, row in self.full_df.iterrows():
            audio_filename = row['audio_filename']
            reference = row['transcription']
            person_id = row['person_id']
            
            audio_path = self.get_audio_path(audio_filename)
            
            print(f"\n  [{idx + 1}/{len(self.full_df)}] {audio_filename}")
            print(f"    Reference: {reference}")
            
            try:
                prediction = manager.predict_transcription(audio_path)
                print(f"    Prediction: {prediction}")
                
                sample_wer = manager.wer_metric.compute(
                    predictions=[prediction],
                    references=[reference]
                )
                sample_wer_ci = manager.wer_metric.compute(
                    predictions=[prediction.lower()],
                    references=[reference.lower()]
                )
                print(f"    WER: {sample_wer:.4f} (case-insensitive: {sample_wer_ci:.4f})")
                
            except Exception as e:
                print(f"    Error transcribing: {e}")
                prediction = ""
                sample_wer = 1.0
                sample_wer_ci = 1.0
            
            all_predictions.append({
                "person_id": person_id,
                "audio_filename": audio_filename,
                "reference": reference,
                "prediction": prediction,
                "wer": sample_wer,
                "wer_ci": sample_wer_ci,
            })
            
            if person_id not in person_results:
                person_results[person_id] = {
                    "person_id": person_id,
                    "wers": [],
                    "wers_ci": [],
                    "num_samples": 0,
                }
            person_results[person_id]["wers"].append(sample_wer)
            person_results[person_id]["wers_ci"].append(sample_wer_ci)
            person_results[person_id]["num_samples"] += 1
        
        del manager
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        predictions_df = pd.DataFrame(all_predictions)
        
        person_summary = []
        for person_id, data in person_results.items():
            person_summary.append({
                "person_id": person_id,
                "num_samples": data["num_samples"],
                "wer_mean": sum(data["wers"]) / len(data["wers"]) if data["wers"] else 0,
                "wer_ci_mean": sum(data["wers_ci"]) / len(data["wers_ci"]) if data["wers_ci"] else 0,
            })
        
        person_summary_df = pd.DataFrame(person_summary)
        
        overall_wer = predictions_df["wer"].mean() if len(predictions_df) > 0 else 0
        overall_wer_ci = predictions_df["wer_ci"].mean() if len(predictions_df) > 0 else 0
        
        self.results = {
            "predictions_df": predictions_df,
            "person_summary_df": person_summary_df,
            "overall_wer": overall_wer,
            "overall_wer_ci": overall_wer_ci,
        }
        
        print(f"\n{'='*60}")
        print(f"Overall WER: {overall_wer:.4f}")
        print(f"Overall WER (case-insensitive): {overall_wer_ci:.4f}")
        print(f"{'='*60}")
        
        return predictions_df
    
    def save_results(self, results_df):
        """Save AMICOS evaluation results."""
        os.makedirs(self.args.output_dir, exist_ok=True)
        
        if self.results.get("predictions_df") is not None:
            predictions_path = os.path.join(self.args.output_dir, "detailed_predictions.csv")
            self.results["predictions_df"].to_csv(predictions_path, index=False)
            print(f"  Saved detailed predictions to: {predictions_path}")
        
        if self.results.get("person_summary_df") is not None:
            person_summary_path = os.path.join(self.args.output_dir, "person_summary.csv")
            self.results["person_summary_df"].to_csv(person_summary_path, index=False)
            print(f"  Saved person summary to: {person_summary_path}")
        
        summary = {
            "model_source": "local" if self.trained_model_path else self.args.model_source,
            "model_name_or_path": self.trained_model_path or self.args.model_name_or_path,
            "language": self.args.language,
            "num_samples": len(results_df),
            "wer_mean": self.results.get("overall_wer"),
            "wer_ci_mean": self.results.get("overall_wer_ci"),
            "num_persons": len(self.results.get("person_summary_df", pd.DataFrame())),
        }
        
        summary_path = os.path.join(self.args.output_dir, "evaluation_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"  Saved summary to: {summary_path}")
    
    def evaluate_baseline_model(self):
        """Evaluate pretrained/baseline model on AMICOS dataset."""
        os.makedirs(self.args.output_dir, exist_ok=True)
        
        print(f"\n{'='*80}")
        print("AMICOS EVALUATION (pretrained/original model)")
        print(f"{'='*80}")
        
        results_df = self.evaluate()
        
        if not results_df.empty:
            if self.results.get("person_summary_df") is not None:
                print(f"\nPer-person results:")
                print(self.results["person_summary_df"].to_string(index=False))
            
            self.save_results(results_df)
        
        print("\n" + "=" * 80)
        print("EVALUATION COMPLETE")
        print("=" * 80)
    
    def evaluate_finetuned_models(self, models):
        """Evaluate fine-tuned models (AMICOS)."""
        raise NotImplementedError("evaluate_finetuned_models() is not implemented for AMICOS dataset")


def create_evaluation_strategy(args, full_df=None, trained_model_path=None):
    """Factory function to create the appropriate evaluation strategy."""
    if args.dataset_type == "torgo":
        return TorgoEvaluationStrategy(args, full_df, trained_model_path)
    elif args.dataset_type == "amicos":
        return AmicosEvaluationStrategy(args, trained_model_path)
    else:
        return CommonVoiceEvaluationStrategy(args, full_df, trained_model_path)


def main():
    args = parse_args()
    
    print("=" * 80)
    print("WHISPER MODEL EVALUATION")
    print("=" * 80)
    print(f"Dataset type: {args.dataset_type}")
    print(f"Language: {args.language}")
    print(f"Audio dir: {args.audio_dir}")
    print(f"Metadata file: {args.metadata_file_path}")
    print(f"Output dir: {args.output_dir}")
    
    if args.dataset_type == "amicos" and args.input_dir:
        print(f"Input dir: {args.input_dir}")
    
    if args.trained_models_dir:
        print(f"Trained models dir: {args.trained_models_dir}")
        models_to_evaluate = find_trained_models(args.trained_models_dir)
        if not models_to_evaluate:
            print(f"ERROR: No trained models found in {args.trained_models_dir}")
            print("Expected structure: holdout_X_seed_Y/fold_Z/model.safetensors")
            sys.exit(1)
        print(f"Found {len(models_to_evaluate)} models to evaluate")
        print(f"Seeds found: {sorted(set(m['seed'] for m in models_to_evaluate))}")
    else:
        print(f"Model name/path: {args.model_name_or_path}")
    
    print("=" * 80)
    
    if args.dataset_type == "amicos":
        strategy = create_evaluation_strategy(args)
    else:
        full_df = load_test_dataset(
            metadata_file_path=args.metadata_file_path,
            audio_dir=args.audio_dir,
            language=args.language,
            dataset_type=args.dataset_type,
        )
        print(f"Total dataset size: {len(full_df)}")
        strategy = create_evaluation_strategy(args, full_df)
    
    models_to_evaluate = None
    
    if args.trained_models_dir:
        print(f"Finding trained models in {args.trained_models_dir}...")
        models_to_evaluate = find_trained_models(args.trained_models_dir)
        if not models_to_evaluate:
            print(f"ERROR: No trained models found in {args.trained_models_dir}")
            print("Expected structure: holdout_X_seed_Y/fold_Z/model.safetensors")
            sys.exit(1)
        print(f"Found {len(models_to_evaluate)} models to evaluate")
        print(f"Seeds found: {sorted(set(m['seed'] for m in models_to_evaluate))}")
    
    if models_to_evaluate:
        if args.dataset_type in ["torgo", "amicos"]:
            print(f"ERROR: Fine-tuned model evaluation is not supported for {args.dataset_type} dataset.")
            sys.exit(1)
        strategy.evaluate_finetuned_models(models_to_evaluate)
    else:
        strategy.evaluate_baseline_model()


if __name__ == "__main__":
    main()
