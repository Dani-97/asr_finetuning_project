from datasets import Dataset as HFDataset, Audio, Features, Sequence, Array2D, Value, concatenate_datasets
import pandas as pd
import numpy as np
import os
import sys
import pandas as pd
import numpy as np
import torch
import torchvision
import evaluate
from transformers import (
    pipeline, 
    AutoModelForSpeechSeq2Seq, 
    AutoProcessor,
    EarlyStoppingCallback
)
import shutil
import warnings
from abc import ABC, abstractmethod
from tqdm import tqdm
from dataclasses import dataclass
from typing import Iterator


class WhisperTrainingStrategy(ABC):
    """Abstract base class for Whisper training strategies."""

    def __init__(self, model, device, processor):
        self.model = model
        self.device = device
        self.processor = processor

    @abstractmethod
    def setup_model(self):
        """Configure model for training (e.g., freeze layers)."""
        pass

    @abstractmethod
    def get_trainable_parameters(self) -> list:
        """Return iterator of trainable parameter groups for optimizer."""
        pass

    def print_trainable_params_info(self):
        """Print information about trainable/frozen parameters."""
        total_params = 0
        trainable_params = 0
        frozen_params = 0

        for name, param in self.model.named_parameters():
            total_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
            else:
                frozen_params += param.numel()

        print(f"\n{'='*60}")
        print(f"Parameter Summary:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
        print(f"  Frozen: {frozen_params:,} ({100*frozen_params/total_params:.2f}%)")
        print(f"{'='*60}\n")


class FullModelTrainingStrategy(WhisperTrainingStrategy):
    """Training strategy that trains all model parameters."""

    def setup_model(self):
        for param in self.model.parameters():
            param.requires_grad = True
        self.print_trainable_params_info()

    def get_trainable_parameters(self):
        return self.model.parameters()


class FrozenEncoderTrainingStrategy(WhisperTrainingStrategy):
    """Training strategy that freezes encoder and trains only decoder."""

    def setup_model(self):
        for param in self.model.model.encoder.parameters():
            param.requires_grad = False

        for param in self.model.model.decoder.parameters():
            param.requires_grad = True

        if hasattr(self.model, 'base_model_embed_tokens'):
            for param in self.model.base_model_embed_tokens.parameters():
                param.requires_grad = True

        self.print_trainable_params_info()

    def get_trainable_parameters(self):
        trainable_params = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                trainable_params.append(param)
        return trainable_params


def create_training_strategy(strategy_name, model, device, processor):
    """Factory function to create training strategy.

    Args:
        strategy_name: Name of the strategy ("full" or "frozen_encoder")
        model: The Whisper model
        device: Device to run on
        processor: Whisper processor

    Returns:
        WhisperTrainingStrategy instance
    """
    if strategy_name == "full":
        return FullModelTrainingStrategy(model, device, processor)
    elif strategy_name == "frozen_encoder":
        return FrozenEncoderTrainingStrategy(model, device, processor)
    else:
        raise ValueError(f"Unknown training strategy: {strategy_name}")


class DataCollatorForWhisper:
    def __init__(self, processor):
        self.processor = processor
    
    def __call__(self, features):
        input_features = [{"input_features": f["input_features"].squeeze(0) if f["input_features"].dim() > 2 else f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        
        labels = [{"input_ids": f["labels"]} for f in features]
        batch["labels"] = self.processor.tokenizer.pad(labels, return_tensors="pt")["input_ids"]
        
        return batch


class Whisper_Manager:
    
    def __init__(self, model_name="openai/whisper-tiny", device=None, language="spanish", training_strategy=None):
        """
        Initialize Whisper model manager.

        Args:
            model_name: Name of the Whisper model (e.g., "openai/whisper-tiny", "openai/whisper-base")
            device: Device to run on (None for auto-detection)
            language: Target language for transcription
            training_strategy: WhisperTrainingStrategy instance or None for full model training
        """
        self.model_name = model_name
        self.language = language

        if device is None:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        print(f"Loading {model_name} on {self.device}...")

        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            use_safetensors=True
        )
        self.model.to(self.device)

        self.processor = AutoProcessor.from_pretrained(model_name)

        self.wer_metric = evaluate.load("wer")

        self.language_codes = {
            "spanish": "es",
            "galician": "gl",
            "english": "en"
        }

        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            device=0 if self.device == "cuda:0" else -1,
            torch_dtype=self.torch_dtype
        )

        self.training_strategy = None
        if training_strategy is not None:
            self.training_strategy = training_strategy(self.model, self.device, self.processor)
            self.training_strategy.setup_model()
        
    def predict_transcription(self, audio_path):
        """
        Predict transcription for a single audio file.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Transcribed text
        """
        try:
            lang_code = self.language_codes.get(self.language.lower(), self.language)
            
            generate_kwargs = {}
            if not self.model.config.forced_decoder_ids:
                generate_kwargs["language"] = lang_code
            
            result = self.pipe(
                audio_path,
                generate_kwargs=generate_kwargs,
                chunk_length_s=30,
                batch_size=8
            )
            
            return result["text"].strip()
            
        except Exception as e:
            print(f"Error transcribing {audio_path}: {e}")
            return ""
    
    def transcribe_batch(self, audio_paths, audio_dir=""):
        """
        Transcribe a batch of audio files using true GPU batching.
        
        Args:
            audio_paths: List of audio file paths
            audio_dir: Base directory for audio paths
            
        Returns:
            List of transcriptions
        """
        lang_code = self.language_codes.get(self.language.lower(), self.language)
        
        full_paths = [os.path.join(audio_dir, p) if audio_dir else p for p in audio_paths]
        
        audio_arrays = []
        for audio_path in full_paths:
            try:
                audio, sr = librosa.load(audio_path, sr=16000)
                audio_arrays.append(audio)
            except Exception as e:
                print(f"Error loading {audio_path}: {e}")
                audio_arrays.append(np.zeros(16000))
        
        processor_output = self.processor(
            audio_arrays,
            sampling_rate=16000,
            return_tensors="pt"
        )
        
        input_features = processor_output.input_features.to(self.device)
        attention_mask = processor_output.attention_mask.to(self.device) if hasattr(processor_output, 'attention_mask') else None
        
        with torch.no_grad():
            generate_kwargs = {
                "input_features": input_features,
                "attention_mask": attention_mask,
            }
            if lang_code and not self.model.config.forced_decoder_ids:
                generate_kwargs["language"] = lang_code
            
            predicted_ids = self.model.generate(**generate_kwargs)
        
        transcriptions = self.processor.batch_decode(
            predicted_ids,
            skip_special_tokens=True
        )
        
        return [t.strip() for t in transcriptions]
    
    def train(
        self,
        train_loader,
        val_loader,
        optimizer,
        lr_scheduler,
        epochs_manager,
        output_dir,
        trainer=None,
        train_samples_count=0,
        val_samples_count=0,
    ):
        """
        Train the Whisper model using pre-configured data loaders and optimizer.

        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            optimizer: PyTorch optimizer
            lr_scheduler: PyTorch learning rate scheduler
            epochs_manager: TrainingEpochsManager instance
            output_dir: Directory to save models and logs
            trainer: CrossValidationTrainer instance (for managing best_loss and current_epoch)
            train_samples_count: Number of training samples (for reporting)
            val_samples_count: Number of validation samples (for reporting)

        Returns:
            dict: Training results including best model path and metrics
        """
        os.makedirs(output_dir, exist_ok=True)

        history = []
        history_path = os.path.join(output_dir, 'training_history.csv')

        self.model.train()

        total_epochs_display = epochs_manager.get_total_epochs_display()

        print("\nStarting training loop...")
        if train_samples_count > 0 and val_samples_count > 0:
            print(f"Train samples: {train_samples_count}, Val samples: {val_samples_count}")

        best_loss = float('inf')
        epoch = 0

        while True:
            current_epoch = epoch + 1

            self._print_epoch_header(current_epoch, total_epochs_display)

            train_result = self._train_epoch(
                train_loader, optimizer, lr_scheduler, progress_bar_desc="Training"
            )

            val_result = self._validate_epoch(val_loader)

            metrics = self._compute_epoch_metrics(train_result, val_result, optimizer)

            self._print_training_metrics(metrics['avg_train_loss'], metrics['current_lr'])
            self._print_validation_metrics(metrics['avg_val_loss'])

            self._save_epoch_history(
                history, history_path, current_epoch,
                metrics['avg_train_loss'], metrics['avg_val_loss'], metrics['current_lr']
            )

            current_loss = metrics['avg_val_loss']
            previous_best_loss = best_loss
            is_best = current_loss < best_loss
            if is_best:
                best_loss = metrics['avg_val_loss']
            self._save_epoch_models(output_dir, is_best, current_epoch == 1)

            self._print_epoch_summary(
                current_epoch, metrics['avg_train_loss'], metrics['avg_val_loss'],
                metrics['current_lr']
            )

            if not epochs_manager.continue_training(epoch, current_loss, previous_best_loss):
                break

            epoch += 1

        if trainer is not None:
            trainer.best_loss = best_loss
            trainer.current_epoch = epoch

        print(f"\n✓ Training history saved to: {history_path}")

        return {
            "best_model_path": os.path.join(output_dir, "best_model.safetensors"),
            "last_model_path": os.path.join(output_dir, "last_model.safetensors"),
            "train_samples": train_samples_count,
            "val_samples": val_samples_count,
            "best_loss": best_loss,
        }
    
    def _save_model_safetensors_manual(self, output_dir, name):
        """Save model weights as safetensors."""
        safetensors_path = os.path.join(output_dir, f"{name}.safetensors")
        from safetensors import torch as safe_torch
        
        state_dict = {k: v for k, v in self.model.state_dict().items() 
                     if not k.startswith('model.decoder.embed_tokens')}
        safe_torch.save_file(state_dict, safetensors_path)
    
    def _save_necessary_files_manual(self, output_dir):
        """Save necessary tokenizer/processor files once."""
        necessary_files = [
            'tokenizer.json',
            'vocab.json', 
            'merges.txt',
            'normalizer.json',
            'tokenizer_config.json',
            'special_tokens_map.json'
        ]
        
        for filename in necessary_files:
            src_path = None
            if hasattr(self.processor, 'tokenizer') and hasattr(self.processor.tokenizer, 'name_or_path'):
                tokenizer_dir = self.processor.tokenizer.name_or_path
                src_path = os.path.join(tokenizer_dir, filename)
                if not os.path.exists(src_path):
                    src_path = None
            
            if src_path and os.path.exists(src_path):
                dst_path = os.path.join(output_dir, filename)
                shutil.copy2(src_path, dst_path)
        
        # Save config.json
        config_path = os.path.join(output_dir, 'config.json')
        import json
        config = {
            'model_type': 'whisper',
            'vocab_size': self.processor.tokenizer.vocab_size,
            'max_target_positions': 448,
            'd_model': 384,
            'encoder_attention_heads': 6,
            'decoder_attention_heads': 6,
            'encoder_layers': 4,
            'decoder_layers': 4,
            'd_ff': 1536,
        }
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

    def _train_epoch(self, train_loader, optimizer, lr_scheduler, progress_bar_desc="Training"):
        """
        Execute one training epoch.

        Returns:
            dict: Contains 'losses', 'predictions', 'references', 'avg_loss'
        """
        self.model.train()
        train_losses = []
        train_predictions = []
        train_references = []

        progress_bar = tqdm(train_loader, desc=progress_bar_desc)
        for batch_idx, batch in enumerate(progress_bar):
            input_features = batch["input_features"].to(self.device)
            labels = batch["labels"].to(self.device)

            outputs = self.model(input_features=input_features, labels=labels)
            loss = outputs.loss

            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            train_losses.append(loss.item())
            progress_bar.set_postfix({"loss": loss.item()})

            if batch_idx == 0:
                with torch.no_grad():
                    predicted_ids = self.model.generate(input_features)
                    decoded_preds = self.processor.batch_decode(predicted_ids.cpu(), skip_special_tokens=True)

                    labels_mask = labels != -100
                    labels_for_decode = torch.where(labels_mask, labels, self.processor.tokenizer.pad_token_id)
                    decoded_labels = self.processor.batch_decode(labels_for_decode.cpu(), skip_special_tokens=True)

                    train_predictions.extend(decoded_preds)
                    train_references.extend(decoded_labels)

        avg_loss = np.mean(train_losses)

        return {
            "losses": train_losses,
            "predictions": train_predictions,
            "references": train_references,
            "avg_loss": avg_loss,
        }

    def _validate_epoch(self, val_loader):
        """
        Execute one validation epoch.

        Returns:
            dict: Contains 'losses', 'predictions', 'references', 'avg_loss'
        """
        self.model.eval()
        val_losses = []
        val_predictions = []
        val_references = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                input_features = batch["input_features"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(input_features=input_features, labels=labels)
                val_losses.append(outputs.loss.item())

                predicted_ids = self.model.generate(input_features)
                decoded_preds = self.processor.batch_decode(predicted_ids.cpu(), skip_special_tokens=True)

                labels_mask = labels != -100
                labels_for_decode = torch.where(labels_mask, labels, self.processor.tokenizer.pad_token_id)
                decoded_labels = self.processor.batch_decode(labels_for_decode.cpu(), skip_special_tokens=True)

                val_predictions.extend(decoded_preds)
                val_references.extend(decoded_labels)

        avg_loss = np.mean(val_losses)

        return {
            "losses": val_losses,
            "predictions": val_predictions,
            "references": val_references,
            "avg_loss": avg_loss,
        }

    def _save_epoch_history(self, history, history_path, epoch, avg_train_loss, avg_val_loss, current_lr):
        """Save epoch metrics to training history."""
        history_entry = {
            'epoch': epoch,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'learning_rate': current_lr
        }
        history.append(history_entry)

        history_df = pd.DataFrame(history)
        history_df.to_csv(history_path, index=False)

    def _save_epoch_models(self, output_dir, is_best, is_first_epoch):
        """Save model checkpoints after each epoch."""
        self._save_model_safetensors_manual(output_dir, 'last_model')

        if is_best:
            self._save_model_safetensors_manual(output_dir, 'best_model')
            print(f"\n✓ New best model saved")

        if is_first_epoch:
            self._save_necessary_files_manual(output_dir)

    def _print_epoch_summary(self, epoch, avg_train_loss, avg_val_loss, current_lr):
        """Print epoch summary."""
        print(f"\n{'='*60}")
        print(f"Epoch {epoch} Summary:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss:   {avg_val_loss:.4f}")
        print(f"  LR:         {current_lr:.2e}")
        print(f"{'='*60}\n")

    def _compute_epoch_metrics(self, train_result, val_result, optimizer):
        """Extract and compute metrics from train and validation results.

        Returns:
            dict: Contains all metrics for the epoch
        """
        return {
            "avg_train_loss": train_result["avg_loss"],
            "current_lr": optimizer.param_groups[0]["lr"],
            "avg_val_loss": val_result["avg_loss"],
        }

    def _print_epoch_header(self, current_epoch, total_epochs_display):
        """Print epoch header."""
        print(f"\n{'='*60}")
        print(f"Epoch {current_epoch}/{total_epochs_display}")
        print(f"{'='*60}")

    def _print_training_metrics(self, avg_train_loss, current_lr):
        """Print training metrics."""
        print(f"\nTraining Loss: {avg_train_loss:.4f}")
        print(f"Learning Rate: {current_lr:.2e}")

    def _print_validation_metrics(self, avg_val_loss):
        """Print validation metrics."""
        print(f"\nValidation Loss: {avg_val_loss:.4f}")

    def evaluate_model(self, df, audio_dir="", max_samples=None, verbose=True, batch_size=8):
        """
        Compute Word Error Rate (WER) for the model on a dataset.
        
        Args:
            df: DataFrame containing 'path' and 'sentence' columns
            audio_dir: Directory containing audio files (should include language subdir like es_all/)
            max_samples: Maximum number of samples to evaluate (None for all)
            verbose: Whether to print progress
            batch_size: Number of samples to process in parallel
            
        Returns:
            Dictionary with WER metrics and sample predictions
        """
        import time
        
        if max_samples:
            df = df.head(max_samples)
        
        batch_size = 8
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Evaluating model: {self.model_name}")
            print(f"Dataset size: {len(df)} samples")
            print(f"Batch size: {batch_size}")
            print(f"{'='*60}")
        
        predictions = []
        references = []
        wer_scores = []
        all_paths = []
        
        n_samples = len(df)
        n_batches = (n_samples + batch_size - 1) // batch_size
        
        total_start_time = time.time()
        
        for batch_idx in range(n_batches):
            batch_start_time = time.time()
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, n_samples)
            batch_df = df.iloc[start_idx:end_idx].copy()
            
            batch_audio_paths = batch_df['path'].tolist()
            batch_references = [str(row['sentence']).strip() for _, row in batch_df.iterrows()]
            
            try:
                batch_transcriptions = self.transcribe_batch(batch_audio_paths, audio_dir)
                
                for i, (prediction, reference) in enumerate(zip(batch_transcriptions, batch_references)):
                    predictions.append(prediction)
                    references.append(reference)
                    all_paths.append(batch_df.iloc[i]['path'])
                    
                    sample_wer = self.wer_metric.compute(predictions=[prediction], references=[reference])
                    wer_scores.append(sample_wer)
                    
                    if verbose and batch_idx == 0 and i < 3:
                        print(f"\nSample {start_idx + i + 1}:")
                        print(f"Reference: {reference}")
                        print(f"Prediction: {prediction}")
                        print(f"Sample WER: {sample_wer:.4f}")
                        print("-" * 50)
                        
            except Exception as e:
                print(f"Error processing batch {batch_idx + 1}: {e}")
                import traceback
                traceback.print_exc()
                for j in range(len(batch_audio_paths)):
                    predictions.append("")
                    references.append(batch_references[j])
                    all_paths.append(batch_df.iloc[j]['path'])
                    wer_scores.append(1.0)
            
            batch_time = time.time() - batch_start_time
            eta = batch_time * (n_batches - batch_idx - 1)
            if verbose:
                print(f"Batch {batch_idx + 1}/{n_batches} ({start_idx + 1}-{end_idx}/{n_samples}) - {batch_time:.1f}s - ETA: {eta:.0f}s")
        
        total_time = time.time() - total_start_time
        if verbose:
            print(f"\nProcessed {n_samples} samples in {total_time:.1f}s ({n_samples/total_time:.1f} samples/s)")
        
        wer_score = self.wer_metric.compute(predictions=predictions, references=references)
        wer_score_ci = self.wer_metric.compute(
            predictions=[p.lower() for p in predictions],
            references=[r.lower() for r in references]
        )

        results_df = pd.DataFrame({
            'path': all_paths[:len(predictions)],
            'reference': references,
            'prediction': predictions,
            'wer': wer_scores,
            'wer_ci': [self.wer_metric.compute(predictions=[p.lower()], references=[r.lower()]) 
                       for p, r in zip(predictions, references)]
        })

        results = {
            'model': self.model_name,
            'wer': wer_score,
            'wer_ci': wer_score_ci,
            'num_samples': len(predictions),
            'results_df': results_df
        }
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Model: {self.model_name}")
            print(f"WER: {wer_score:.4f}")
            print(f"WER (case-insensitive): {wer_score_ci:.4f}")
            print(f"Number of samples evaluated: {len(predictions)}")
            print(f"{'='*60}")
        
        return results
    
    def evaluate_multiple_models(self, df, model_names, audio_dir="", max_samples=None):
        """
        Evaluate multiple Whisper models on the same dataset.
        
        Args:
            df: DataFrame with 'path' and 'sentence' columns
            model_names: List of model names to evaluate
            audio_dir: Optional directory prefix for audio paths
            max_samples: Maximum number of samples to evaluate
            
        Returns:
            DataFrame with WER scores for each model
        """
        results = []
        
        for model_name in model_names:
            print(f"\n{'='*60}")
            print(f"Evaluating {model_name}...")
            print(f"{'='*60}")
            
            manager = Whisper_Manager(model_name=model_name, language=self.language)
            
            model_results = manager.evaluate_model(df, audio_dir=audio_dir, max_samples=max_samples)
            
            results.append({
                "model": model_name,
                "wer": model_results["wer"],
                "num_samples": model_results["num_samples"]
            })
            
            del manager
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return pd.DataFrame(results)


class Local_Whisper_Manager(Whisper_Manager):
    """
    Whisper manager for local models stored on disk.
    Can load either a HF-style folder or a single .safetensors checkpoint.
    """
    def __init__(self, local_model_path, base_model_name="openai/whisper-tiny", device=None, language="spanish"):
        super().__init__(device=device, language=language)
        self.local_model_path = local_model_path

        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        if os.path.isfile(local_model_path) and local_model_path.endswith(".safetensors"):
            from transformers import WhisperForConditionalGeneration
            from safetensors.torch import load_file

            print(f"Loading checkpoint {local_model_path} into base model {base_model_name} on {self.device}...")
            self.model = WhisperForConditionalGeneration.from_pretrained(
                base_model_name, torch_dtype=torch.float32
            )
            state_dict = load_file(local_model_path)
            self.model.load_state_dict(state_dict, strict=False)
        elif os.path.isdir(local_model_path) or os.path.exists(local_model_path):
            print(f"Loading model from folder {local_model_path} on {self.device}...")
            from transformers import AutoModelForSpeechSeq2Seq
            self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                local_model_path,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
                use_safetensors=True
            )
        else:
            raise ValueError(f"Invalid path: {local_model_path}. Must be a folder or a .safetensors file.")

        self.model.to(self.device)

        from transformers import AutoProcessor
        try:
            self.processor = AutoProcessor.from_pretrained(
                local_model_path if os.path.isdir(local_model_path) else base_model_name
            )
        except Exception as e:
            print(f"Warning: Could not load processor from {local_model_path}. Falling back to {base_model_name}. Error: {e}")
            self.processor = AutoProcessor.from_pretrained(base_model_name)

        from transformers import pipeline
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            device=0 if self.device == "cuda:0" else -1,
            torch_dtype=self.torch_dtype
        )

    def predict_transcription(self, audio_path):
        lang_code = self.language_codes.get(self.language.lower(), self.language)
        
        generate_kwargs = {}
        if not self.model.config.forced_decoder_ids:
            generate_kwargs["language"] = lang_code
        
        result = self.pipe(
            audio_path,
            generate_kwargs=generate_kwargs,
            chunk_length_s=30,
            batch_size=8
        )
        return result["text"].strip()
