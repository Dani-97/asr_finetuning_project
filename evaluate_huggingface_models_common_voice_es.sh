#!/bin/bash
# Evaluate all Whisper models on Common Voice Spanish dataset

echo "========================================"
echo "Evaluating Whisper models on Common Voice Spanish"
echo "========================================"

# Whisper Tiny
echo "Evaluating whisper-tiny..."
python evaluate_holdout_cv.py \
    --language spanish \
    --model_source huggingface \
    --model_name_or_path openai/whisper-tiny \
    --audio_dir common_voice_dataset/es_all/ \
    --metadata_file_path common_voice_dataset/es_all.tsv \
    --dataset_type common_voice \
    --test_size 3960 \
    --test_size_is_absolute \
    --output_dir whisper-tiny-baseline-spanish
echo "Done: whisper-tiny-baseline-spanish"
echo ""

# Whisper Base
echo "Evaluating whisper-base..."
python evaluate_holdout_cv.py \
    --language spanish \
    --model_source huggingface \
    --model_name_or_path openai/whisper-base \
    --audio_dir common_voice_dataset/es_all/ \
    --metadata_file_path common_voice_dataset/es_all.tsv \
    --dataset_type common_voice \
    --test_size 3960 \
    --test_size_is_absolute \
    --output_dir whisper-base-baseline-spanish
echo "Done: whisper-base-baseline-spanish"
echo ""

# Whisper Small
echo "Evaluating whisper-small..."
python evaluate_holdout_cv.py \
    --language spanish \
    --model_source huggingface \
    --model_name_or_path openai/whisper-small \
    --audio_dir common_voice_dataset/es_all/ \
    --metadata_file_path common_voice_dataset/es_all.tsv \
    --dataset_type common_voice \
    --test_size 3960 \
    --test_size_is_absolute \
    --output_dir whisper-small-baseline-spanish
echo "Done: whisper-small-baseline-spanish"
echo ""

# Whisper Large
# echo "Evaluating whisper-large..."
# python evaluate_holdout_cv.py \
#     --language spanish \
#     --model_source huggingface \
#     --model_name_or_path openai/whisper-large \
#     --audio_dir common_voice_dataset/es_all/ \
#     --metadata_file_path common_voice_dataset/es_all.tsv \
#     --dataset_type common_voice \
#     --test_size 3960 \
#     --test_size_is_absolute \
#     --output_dir whisper-large-baseline-spanish
# echo "Done: whisper-large-baseline-spanish"
# echo ""

echo "========================================"
echo "All evaluations complete!"
echo "========================================"
