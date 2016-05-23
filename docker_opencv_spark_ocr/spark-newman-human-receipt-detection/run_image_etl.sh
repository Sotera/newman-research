#!/usr/bin/env bash

set +x
set -e

echo "===========================================$0"

OUTPUT_DIR='etl_output'
if [[ -d "/$OUTPUT_DIR" ]]; then
    rm -rf "/$OUTPUT_DIR"
fi

spark-submit --master local[*] --driver-memory 8g --conf spark.storage.memoryFraction=.8 hog_gist_feature_extraction.py
