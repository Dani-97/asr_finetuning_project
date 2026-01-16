#!/bin/bash
# Run Whisper Transcription Docker container

docker run -it --rm --gpus all --mount type=bind,src=.,dst=/app asr_finetuning_project bash
