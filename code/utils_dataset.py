import os
import warnings
import numpy as np
import pandas as pd
import librosa
import torch
from torch.utils.data import IterableDataset
from abc import ABC, abstractmethod

warnings.filterwarnings("ignore")


# =========================
# Base Dataset Classes (for metadata loading)
# =========================

class BaseDataset(ABC):
    """Abstract base class for dataset metadata loaders."""

    def __init__(self, metadata_file_path, sample_size=None, random_state=42):
        self.metadata_file_path = metadata_file_path
        self.sample_size = sample_size
        self.random_state = random_state
        self.df = self._load_and_process()

    @abstractmethod
    def _load_and_process(self) -> pd.DataFrame:
        """Load metadata and return DataFrame with 'path' and 'sentence' columns."""
        pass

    def __len__(self):
        if self.df is None:
            return 0
        return len(self.df)

    def __repr__(self):
        return f"{self.__class__.__name__}({len(self.df)} samples)"


class CommonVoiceDataset(BaseDataset):
    """Dataset loader for Common Voice TSV files."""

    def _load_and_process(self) -> pd.DataFrame:
        df = pd.read_csv(self.metadata_file_path, sep="\t")

        if self.sample_size is not None and len(df) > self.sample_size:
            df = df.sample(n=self.sample_size, random_state=self.random_state)

        return df[["path", "sentence"]].reset_index(drop=True)


class TorgoDataset(BaseDataset):
    """Dataset loader for TORGO CSV files."""

    def _load_and_process(self) -> pd.DataFrame:
        df = pd.read_csv(self.metadata_file_path)

        df = df.rename(columns={
            "wav_path": "path",
            "transcription": "sentence"
        })

        if self.sample_size is not None and len(df) > self.sample_size:
            df = df.sample(n=self.sample_size, random_state=self.random_state)

        return df.reset_index(drop=True)


def create_dataset(metadata_file_path, sample_size=None, random_state=42):
    """
    Factory function to create the appropriate dataset based on file extension.

    Args:
        metadata_file_path: Path to metadata file (TSV for Common Voice, CSV for TORGO)
        sample_size: Optional limit on number of samples
        random_state: Random seed for sampling

    Returns:
        BaseDataset instance (CommonVoiceDataset or TorgoDataset)
    """
    ext = os.path.splitext(metadata_file_path)[1].lower()

    if ext == ".tsv":
        return CommonVoiceDataset(
            metadata_file_path=metadata_file_path,
            sample_size=sample_size,
            random_state=random_state,
        )
    elif ext == ".csv":
        return TorgoDataset(
            metadata_file_path=metadata_file_path,
            sample_size=sample_size,
            random_state=random_state,
        )
    else:
        raise ValueError(f"Unsupported metadata file format: {ext}. Use .tsv or .csv")


# =========================
# Base iterable dataset
# =========================

class BaseAudioIterableDataset(IterableDataset):
    """Memory-safe iterable dataset for Whisper training."""

    def __init__(
        self,
        processor,
        tsv_path,
        audio_dir,
        audio_subdir=None,
        sample_size=None,
        random_state=42,
        filter_fn=None,
    ):
        self.processor = processor
        self.audio_dir = audio_dir
        self.audio_subdir = audio_subdir
        self.sample_size = sample_size
        self.random_state = random_state

        # Load metadata ONCE
        df = pd.read_csv(tsv_path, sep="\t")

        if filter_fn is not None:
            df = filter_fn(df)

        if sample_size is not None and len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=random_state)

        self.df = df.reset_index(drop=True)

    def __len__(self):
        if self.df is None:
            return 0
        return len(self.df)

    def _get_indices_for_worker(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            return range(len(self.df))

        per_worker = int(np.ceil(len(self.df) / worker_info.num_workers))
        start = worker_info.id * per_worker
        end = min(start + per_worker, len(self.df))
        return range(start, end)

    def _resolve_audio_path(self, relative_path):
        """
        Metadata path + optional audio subdirectory
        """
        if self.audio_subdir is None:
            return os.path.join(self.audio_dir, relative_path)
        return os.path.join(self.audio_dir, self.audio_subdir, relative_path)

    def __iter__(self):
        indices = self._get_indices_for_worker()

        for idx in indices:
            row = self.df.iloc[idx]

            # Resolve audio path
            audio_path = self._resolve_audio_path(row["path"])

            try:
                # Load audio file directly
                audio_array, sampling_rate = librosa.load(
                    audio_path,
                    sr=16000,
                    duration=30.0,
                    dtype=np.float32,
                )

                with torch.no_grad():
                    input_features = self.processor.feature_extractor(
                        audio_array,
                        sampling_rate=sampling_rate,
                        return_tensors="pt",
                    ).input_features.squeeze(0)

                # Tokenize and truncate labels
                labels = self.processor.tokenizer(
                    str(row["sentence"]),
                    return_tensors="pt",
                    truncation=True,
                    max_length=448
                ).input_ids.squeeze(0)

                yield {
                    "input_features": input_features,
                    "labels": labels,
                    "audio_path": audio_path,
                }

            except Exception as e:
                # print(f"Skipping {audio_path}: {e}")
                continue

# =========================
# Unified Dataset (supports Common Voice and TORGO)
# =========================

class Unified_Dataset(BaseAudioIterableDataset):
    def __init__(
        self,
        processor,
        audio_dir="./common_voice_dataset",
        language="spanish",
        sample_size=None,
        random_state=42,
        metadata_file_path=None,
    ):
        if metadata_file_path is not None:
            dataset = create_dataset(
                metadata_file_path=metadata_file_path,
                sample_size=sample_size,
                random_state=random_state,
            )
            audio_subdir = None
            filter_fn = None
        else:
            if language == "spanish":
                metadata_file_path = os.path.join(audio_dir, "es_all.tsv")
                audio_subdir = "es_all"
                filter_fn = self._spain_filter
            elif language == "galician":
                metadata_file_path = os.path.join(audio_dir, "gl_all.tsv")
                audio_subdir = "gl_all"
                filter_fn = None
            else:
                raise ValueError(f"Unsupported language: {language}")

            df = pd.read_csv(metadata_file_path, sep="\t")

            if filter_fn is not None:
                df = filter_fn(df)

            if sample_size is not None and len(df) > sample_size:
                df = df.sample(n=sample_size, random_state=random_state)

            dataset = df

        super().__init__(
            processor=processor,
            tsv_path=metadata_file_path,
            audio_dir=audio_dir,
            audio_subdir=audio_subdir,
            sample_size=sample_size,
            random_state=random_state,
            filter_fn=filter_fn,
        )

        if hasattr(dataset, 'df'):
            self.df = dataset.df.reset_index(drop=True)
        else:
            self.df = dataset.reset_index(drop=True)

    def _spain_filter(self, df):
        accents = df["accents"].astype(str)
        return df[
            accents.str.contains("España: Norte")
            | accents.str.contains("España: Galicia")
        ]


# =========================
# Data Augmentation Framework (Strategy Pattern)
# =========================

class AudioAugmentation:
    """Abstract base class for audio augmentation."""

    def __init__(self, seed=42):
        self.seed = seed

    def apply(self, audio, sr, sample_idx=0):
        raise NotImplementedError("Subclasses must implement apply()")


class NoAudioAugmentation(AudioAugmentation):
    """No augmentation - returns audio unchanged."""

    def __init__(self, seed=42):
        super().__init__(seed=seed)

    def apply(self, audio, sr, sample_idx=0):
        return audio


class CompositeAudioAugmentation(AudioAugmentation):
    """Combine multiple augmentation strategies."""

    def __init__(self, strategies=None, seed=42):
        super().__init__(seed=seed)
        self.strategies = strategies or []

    def apply(self, audio, sr, sample_idx=0):
        unique_seed = self.seed + sample_idx if self.seed else None

        for i, strategy in enumerate(self.strategies):
            strategy_seed = unique_seed + i if unique_seed else None
            audio = strategy.apply(audio, sr, strategy_seed)

        return audio


class AudioAugmentationStrategy:
    """Abstract base class for audio augmentation strategies."""

    def __init__(self, apply_prob=1.0):
        self.apply_prob = apply_prob

    def apply(self, audio, sr, seed):
        raise NotImplementedError("Subclasses must implement apply()")

    def _should_apply(self, seed):
        rng = np.random.RandomState(seed)
        return rng.random() < self.apply_prob


class SpeedPerturbationStrategy(AudioAugmentationStrategy):
    """Apply random speed perturbation to audio."""

    def __init__(self, factors=[0.9, 1.0, 1.1], apply_prob=0.5):
        super().__init__(apply_prob=apply_prob)
        self.factors = factors

    def apply(self, audio, sr, seed):
        if not self._should_apply(seed):
            return audio

        rng = np.random.RandomState(seed)
        factor = rng.choice(self.factors)

        audio = librosa.effects.time_stretch(audio, rate=factor)
        return audio


class VolumePerturbationStrategy(AudioAugmentationStrategy):
    """Apply random volume scaling to audio."""

    def __init__(self, min_scale=0.8, max_scale=1.2, apply_prob=0.5):
        super().__init__(apply_prob=apply_prob)
        self.min_scale = min_scale
        self.max_scale = max_scale

    def apply(self, audio, sr, seed):
        if not self._should_apply(seed):
            return audio

        rng = np.random.RandomState(seed)
        scale = rng.uniform(self.min_scale, self.max_scale)
        audio = audio * scale
        return audio


class NoiseInjectionStrategy(AudioAugmentationStrategy):
    """Inject random noise into audio."""

    def __init__(self, noise_levels=[0.0, 0.01, 0.03, 0.05], apply_prob=0.5):
        super().__init__(apply_prob=apply_prob)
        self.noise_levels = noise_levels

    def apply(self, audio, sr, seed):
        if not self._should_apply(seed):
            return audio

        rng = np.random.RandomState(seed)
        noise_level = rng.choice(self.noise_levels)

        if noise_level == 0.0:
            return audio

        noise = rng.randn(len(audio)).astype(np.float32) * noise_level
        audio = audio + noise
        return audio


class TimeShiftStrategy(AudioAugmentationStrategy):
    """Apply random time shift to audio."""

    def __init__(self, max_shift_sec=0.1, apply_prob=0.5):
        super().__init__(apply_prob=apply_prob)
        self.max_shift_sec = max_shift_sec

    def apply(self, audio, sr, seed):
        if not self._should_apply(seed):
            return audio

        rng = np.random.RandomState(seed)
        max_shift = int(self.max_shift_sec * sr)
        shift = rng.randint(-max_shift, max_shift)

        if shift > 0:
            audio = np.pad(audio, (shift, 0), mode='constant')[:len(audio)]
        elif shift < 0:
            audio = np.pad(audio, (0, -shift), mode='constant')[-shift:]
        return audio


def create_audio_augmentation(
    use_augmentation=True,
    use_speed_perturbation=True,
    use_volume_perturbation=True,
    use_noise_injection=True,
    use_time_shift=False,
    speed_apply_prob=0.5,
    volume_apply_prob=0.5,
    noise_apply_prob=0.3,
    time_shift_apply_prob=0.5,
    seed=42,
):
    """
    Factory function to create the appropriate audio augmentation.

    Args:
        use_augmentation: If False, returns NoAudioAugmentation
        use_speed_perturbation: Enable speed perturbation
        use_volume_perturbation: Enable volume perturbation
        use_noise_injection: Enable noise injection
        use_time_shift: Enable time shift
        speed_apply_prob: Probability for speed perturbation
        volume_apply_prob: Probability for volume perturbation
        noise_apply_prob: Probability for noise injection
        time_shift_apply_prob: Probability for time shift
        seed: Random seed for reproducibility

    Returns:
        AudioAugmentation instance (NoAudioAugmentation or CompositeAudioAugmentation)
    """
    if not use_augmentation:
        return NoAudioAugmentation(seed=seed)

    strategies = []

    if use_speed_perturbation:
        strategies.append(SpeedPerturbationStrategy(
            factors=[0.9, 1.0, 1.1],
            apply_prob=speed_apply_prob
        ))

    if use_volume_perturbation:
        strategies.append(VolumePerturbationStrategy(
            min_scale=0.8,
            max_scale=1.2,
            apply_prob=volume_apply_prob
        ))

    if use_noise_injection:
        strategies.append(NoiseInjectionStrategy(
            noise_levels=[0.0, 0.005, 0.01, 0.02],
            apply_prob=noise_apply_prob
        ))

    if use_time_shift:
        strategies.append(TimeShiftStrategy(
            max_shift_sec=0.1,
            apply_prob=time_shift_apply_prob
        ))

    return CompositeAudioAugmentation(strategies=strategies, seed=seed)


