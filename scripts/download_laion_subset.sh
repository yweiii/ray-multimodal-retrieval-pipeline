#!/usr/bin/env bash
set -euo pipefail

mkdir -p data/raw_images

img2dataset \
  --url_list inputs/laion_sample.tsv \
  --input_format "tsv" \
  --url_col "url" \
  --caption_col "caption" \
  --output_format "files" \
  --output_folder data/raw_images \
  --processes_count 8 \
  --thread_count 64 \
  --image_size 512 \
  --retries 2 \
  --timeout 10
