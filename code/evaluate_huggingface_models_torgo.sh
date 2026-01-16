# Whisper Tiny
python evaluate_holdout_cv.py \
    --language english \
    --model_source huggingface \
    --model_name_or_path openai/whisper-tiny \
    --audio_dir torgo_dataset/wav_files/ \
    --metadata_file_path torgo_dataset/metadata.csv \
    --dataset_type torgo \
    --output_dir whisper-tiny-baseline-torgo

# Whisper Base
python evaluate_holdout_cv.py \
    --language english \
    --model_source huggingface \
    --model_name_or_path openai/whisper-base \
    --audio_dir torgo_dataset/wav_files/ \
    --metadata_file_path torgo_dataset/metadata.csv \
    --dataset_type torgo \
    --output_dir whisper-base-baseline-torgo

# Whisper Small
python evaluate_holdout_cv.py \
    --language english \
    --model_source huggingface \
    --model_name_or_path openai/whisper-small \
    --audio_dir torgo_dataset/wav_files/ \
    --metadata_file_path torgo_dataset/metadata.csv \
    --dataset_type torgo \
    --output_dir whisper-small-baseline-torgo

# Whisper Large (commented out - requires more memory)
# python evaluate_holdout_cv.py \
#     --language english \
#     --model_source huggingface \
#     --model_name_or_path openai/whisper-large \
#     --audio_dir torgo_dataset/wav_files/ \
#     --metadata_file_path torgo_dataset/metadata.csv \
#     --dataset_type torgo \
#     --output_dir whisper-large-baseline-torgo
