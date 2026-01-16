import os
import json
import numpy as np
from sklearn.model_selection import train_test_split, KFold

class TrainingEpochsManager:
    """Base class for training epochs management."""

    def __init__(self, patience=3):
        self.patience = patience

    def continue_training(self, epoch, current_value, best_value) -> bool:
        raise NotImplementedError("Subclasses must implement continue_training()")

    def get_current_epoch(self, epoch: int) -> int:
        return epoch + 1

    def get_total_epochs_display(self) -> str:
        raise NotImplementedError("Subclasses must implement get_total_epochs_display()")


class FixedEpochsManager(TrainingEpochsManager):
    """Manager for training with a fixed number of epochs."""

    def __init__(self, num_epochs: int, patience: int = 3):
        super().__init__(patience)
        self.num_epochs = num_epochs

    def continue_training(self, epoch: int, current_value: float, best_value: float) -> bool:
        return epoch < self.num_epochs

    def get_current_epoch(self, epoch: int) -> int:
        return epoch + 1

    def get_total_epochs_display(self) -> str:
        return str(self.num_epochs)


class EarlyStoppingManager(TrainingEpochsManager):
    """Manager for training with early stopping based on validation loss."""

    def __init__(self, max_epochs: int, patience: int):
        super().__init__(patience)
        self.max_epochs = max_epochs
        self.epochs_without_improvement = 0

    def continue_training(self, epoch: int, current_value: float, best_value: float) -> bool:
        if epoch >= self.max_epochs:
            return False

        if current_value < best_value:
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1

        return self.epochs_without_improvement < self.patience

    def get_current_epoch(self, epoch: int) -> int:
        return epoch + 1

    def get_total_epochs_display(self) -> str:
        return f"unlimited (early stopping, max={self.max_epochs})"


def create_epochs_manager(num_epochs: int, patience: int) -> TrainingEpochsManager:
    """
    Factory method to create the appropriate epochs manager.

    Args:
        num_epochs: Number of epochs (None for infinite early stopping)
        patience: Patience for early stopping

    Returns:
        TrainingEpochsManager instance (FixedEpochsManager or EarlyStoppingManager)
    """
    if num_epochs is None:
        return EarlyStoppingManager(max_epochs=1000, patience=patience)
    else:
        return FixedEpochsManager(num_epochs=num_epochs, patience=patience)


class CrossValidationTrainer:
    """
    Base class for training pipelines.

    Encapsulates the holdout + cross-validation training logic.
    Subclasses can override specific methods to customize behavior.
    """

    def __init__(
        self,
        model_name="openai/whisper-tiny",
        audio_dir="./common_voice_dataset",
        output_dir="./training_results",
        batch_size=8,
        learning_rate=5e-5,
        max_audio_length=30.0,
        num_epochs=None,
        patience=5,
        max_samples_eval=100,
        train_mode="full",
    ):
        self.model_name = model_name
        self.audio_dir = audio_dir
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.max_audio_length = max_audio_length
        self.num_epochs = num_epochs
        self.patience = patience
        self.max_samples_eval = max_samples_eval
        self.train_mode = train_mode

    def prepare_dataset_samples(self, train_df, val_df, processor, augmentor=None):
        """Prepare training and validation samples from dataframes."""
        import librosa
        from tqdm import tqdm
        import numpy as np

        def prepare_sample(row, idx):
            audio_path = self._get_audio_path(row)
            try:
                audio, sr = librosa.load(audio_path, sr=16000)
                audio = audio[:int(self.max_audio_length * 16000)]
                if len(audio) < int(0.1 * 16000):
                    audio = np.pad(audio, (0, int(0.1 * 16000) - len(audio)))

                if augmentor is not None:
                    audio = augmentor.apply(audio, sr, idx)

                input_features = processor(audio, sampling_rate=16000, return_tensors="pt").input_features
                labels = processor(text=str(row['sentence'])).input_ids
                return {
                    "input_features": input_features,
                    "labels": labels
                }
            except Exception as e:
                print(f"Error processing {audio_path}: {e}")
                return None

        print("Preparing training data...")
        train_samples = []
        for idx, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Train"):
            sample = prepare_sample(row, idx)
            if sample is not None:
                train_samples.append(sample)

        print("Preparing validation data...")
        val_samples = []
        for idx, row in tqdm(val_df.iterrows(), total=len(val_df), desc="Val"):
            sample = prepare_sample(row, idx)
            if sample is not None:
                val_samples.append(sample)

        return train_samples, val_samples

    def create_data_loaders(self, train_samples, val_samples, processor):
        """Create training and validation data loaders."""
        from torch.utils.data import DataLoader
        from utils_model import DataCollatorForWhisper

        data_collator = DataCollatorForWhisper(processor)

        train_loader = DataLoader(
            train_samples,
            batch_size=self.batch_size,
            collate_fn=data_collator,
            shuffle=True
        )
        val_loader = DataLoader(
            val_samples,
            batch_size=self.batch_size,
            collate_fn=data_collator
        )

        return train_loader, val_loader

    def create_optimizer_and_scheduler(self, model, train_loader, whisper_manager=None):
        """Create optimizer and learning rate scheduler."""
        import torch

        if whisper_manager is not None and whisper_manager.training_strategy is not None:
            trainable_params = whisper_manager.training_strategy.get_trainable_parameters()
            optimizer = torch.optim.AdamW(trainable_params, lr=self.learning_rate)
        else:
            optimizer = torch.optim.AdamW(model.parameters(), lr=self.learning_rate)

        num_training_steps = len(train_loader) * self.num_epochs if self.num_epochs else len(train_loader) * 100
        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1.0, end_factor=0.0, total_iters=num_training_steps
        )

        return optimizer, lr_scheduler

    def _get_audio_path(self, row):
        """Get full audio path from row."""
        path = row['path']
        if self.audio_dir:
            return self._get_audio_path_with_subdir(path)
        return path

    def _get_audio_path_with_subdir(self, path):
        """Get audio path with subdirectory handling."""
        if '/' in path:
            return os.path.join(self.audio_dir, path)
        return os.path.join(self.audio_dir, path)

    def _create_whisper_manager(self, language=None):
        """Create a Whisper manager instance."""
        from utils_model import Whisper_Manager, FrozenEncoderTrainingStrategy

        strategy_class = None
        if self.train_mode == "frozen_encoder":
            strategy_class = FrozenEncoderTrainingStrategy

        return Whisper_Manager(
            model_name=self.model_name,
            language=language,
            training_strategy=strategy_class,
        )

    def run_fold(self, fold_train_df, fold_val_df, fold_dir, augmentor=None, language=None):
        """Run training for a single fold."""
        import os
        import numpy as np

        os.makedirs(fold_dir, exist_ok=True)

        whisper_manager = self._create_whisper_manager(language=language)

        train_samples, val_samples = self.prepare_dataset_samples(
            fold_train_df, fold_val_df,
            processor=whisper_manager.processor,
            augmentor=augmentor
        )

        if len(train_samples) == 0 or len(val_samples) == 0:
            raise ValueError("No samples available for training or validation")

        train_loader, val_loader = self.create_data_loaders(
            train_samples, val_samples,
            processor=whisper_manager.processor
        )

        optimizer, lr_scheduler = self.create_optimizer_and_scheduler(
            whisper_manager.model, train_loader, whisper_manager
        )

        from utils_train import create_epochs_manager
        epochs_manager = create_epochs_manager(self.num_epochs, self.patience)

        whisper_manager.train(
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            epochs_manager=epochs_manager,
            output_dir=fold_dir,
            trainer=self,
            train_samples_count=len(train_samples),
            val_samples_count=len(val_samples),
        )

        del whisper_manager

        return {
            "fold_dir": fold_dir,
            "train_size": len(fold_train_df),
            "val_size": len(fold_val_df),
        }

    def train(
        self,
        full_df,
        base_seed,
        holdout_idx,
        n_holdouts,
        cv_folds,
        test_size,
        test_size_is_absolute,
        augmentor=None,
        language=None,
    ):
        """Run a single holdout repetition with CV on training split."""
        import os
        import json
        import numpy as np
        from sklearn.model_selection import train_test_split, KFold

        holdout_output_dir = os.path.join(
            self.output_dir,
            f"holdout_{holdout_idx + 1}_seed_{base_seed}"
        )
        os.makedirs(holdout_output_dir, exist_ok=True)

        if augmentor is not None:
            print(f"Using data augmentation with seed {augmentor.seed}")

        print(f"\n{'='*80}")
        print(f"HOLDOUT {holdout_idx + 1}/{n_holdouts} (seed={base_seed})")
        print(f"{'='*80}")

        if test_size_is_absolute:
            test_n = int(test_size)
            train_n = len(full_df) - test_n
            test_size_calc = test_n / len(full_df)
            print(f"Test samples: {test_n} ({test_size_calc:.2%})")
            print(f"Train samples: {train_n} ({1-test_size_calc:.2%})")
        else:
            print(f"Test percentage: {test_size:.2%}")

        train_df, test_df = train_test_split(
            full_df,
            test_size=test_size if not test_size_is_absolute else test_size / len(full_df),
            random_state=base_seed,
        )

        train_df = train_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)

        print(f"Train size: {len(train_df)}")
        print(f"Test size: {len(test_df)}")

        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=base_seed)

        cv_results = []

        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(train_df)):
            fold_train_df = train_df.iloc[train_idx].reset_index(drop=True)
            fold_val_df = train_df.iloc[val_idx].reset_index(drop=True)

            fold_dir = os.path.join(holdout_output_dir, f"fold_{fold_idx + 1}")
            os.makedirs(fold_dir, exist_ok=True)

            print(f"\n--- Fold {fold_idx + 1}/{cv_folds} ---")
            print(f"Train: {len(fold_train_df)}, Val: {len(fold_val_df)}")

            fold_result = self.run_fold(
                fold_train_df=fold_train_df,
                fold_val_df=fold_val_df,
                fold_dir=fold_dir,
                augmentor=augmentor,
                language=language,
            )

            cv_results.append({
                "fold": fold_idx + 1,
                "train_size": fold_result["train_size"],
                "val_size": fold_result["val_size"],
            })

        holdout_summary = {
            "holdout_idx": holdout_idx + 1,
            "seed": base_seed,
            "total_samples": len(full_df),
            "train_samples": len(train_df),
            "test_samples": len(test_df),
            "cv_folds": cv_folds,
            "cv_results": cv_results,
        }

        holdout_results_path = os.path.join(holdout_output_dir, "holdout_results.json")
        with open(holdout_results_path, "w") as f:
            json.dump(holdout_summary, f, indent=2)
        print(f"Saved holdout results to: {holdout_results_path}")

        return holdout_summary
