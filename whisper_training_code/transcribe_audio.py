import argparse
import torch
from utils_model import Local_Whisper_Manager, OpenAI_Whisper_Manager
import os
import librosa
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Chunk-wise transcription with Whisper")
    parser.add_argument("--audio_path", type=str, required=True, help="Path to the audio file")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to local fine-tuned model folder or .safetensors file")
    parser.add_argument("--base_model", type=str, default="openai/whisper-tiny",
                        help="Base model architecture for loading a .safetensors checkpoint")
    parser.add_argument("--language", type=str, default="spanish", help="Language for transcription")
    parser.add_argument("--device", type=str, default=None, help="Device to use, e.g. 'cuda:0' or 'cpu'")
    parser.add_argument("--chunk_sec", type=float, default=3.0, help="Chunk size in seconds for transcription")
    args = parser.parse_args()

    if not os.path.isfile(args.audio_path):
        raise FileNotFoundError(f"Audio file not found: {args.audio_path}")
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model path not found: {args.model_path}")

    # Initialize local Whisper manager
    whisper_manager = Local_Whisper_Manager(
        local_model_path=args.model_path,
        base_model_name=args.base_model,
        device=args.device,
        language=args.language
    )

    # Load and preprocess audio
    audio_array, sr = librosa.load(args.audio_path, sr=16000, mono=True)
    audio_array = audio_array.astype(np.float32)

    chunk_samples = int(args.chunk_sec * sr)
    total_samples = len(audio_array)
    transcription = ""

    print(f"\nStarting chunk-wise transcription ({args.chunk_sec}s chunks)...\n")
    
    for start_idx in range(0, total_samples, chunk_samples):
        end_idx = min(start_idx + chunk_samples, total_samples)
        current_chunk = audio_array[start_idx:end_idx]

        # Transcribe current chunk
        result = whisper_manager.pipe(
            current_chunk,
            generate_kwargs={
                "language": whisper_manager.language_codes.get(args.language.lower(), args.language),
                "task": "transcribe"
            },
            chunk_length_s=30,
            batch_size=8
        )
        chunk_text = result["text"].strip()
        transcription += " " + chunk_text  # concatenate sequentially
        print(f"[{start_idx/sr:.1f}-{end_idx/sr:.1f}s] Chunk transcription:\n{chunk_text}\n{'-'*50}")

    transcription = transcription.strip()
    print(f"\nFinal concatenated transcription:\n{transcription}\n{'='*60}\n")

if __name__ == "__main__":
    main()
