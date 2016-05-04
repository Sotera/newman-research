#!/usr/bin/env bash

set +x
set -e

echo "===========================================$0"

OUTPUT_DIR='ocr_output'
if [[ -d "pst-extract/$OUTPUT_DIR" ]]; then
    rm -rf "pst-extract/$OUTPUT_DIR"
fi

spark-submit --master local[*] --driver-memory 8g --conf spark.storage.memoryFraction=.8 spark/run_ocr.py pst-extract/post-spam-filter pst-extract/$OUTPUT_DIR

