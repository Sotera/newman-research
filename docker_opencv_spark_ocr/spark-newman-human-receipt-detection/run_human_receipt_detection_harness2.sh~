#!/usr/bin/env bash

set +x
set -e

echo "===========================================$0"

OUTPUT_DIR='human_receipt_detection_output'
if [[ -d "pst-extract/$OUTPUT_DIR" ]]; then
    rm -rf "pst-extract/$OUTPUT_DIR"
fi

spark-submit --master local[*] --driver-memory 8g --files hog_gist_feature_extraction.py,default_param.py,gabor_features.py,human_classifier.pkl,receipt_classifier.pkl,SLIP.py,resize_image.py,LogGabor.py --conf spark.storage.memoryFraction=.8 run_human_receipt_detection.py pst-extract/pst-json pst-extract/$OUTPUT_DIR

