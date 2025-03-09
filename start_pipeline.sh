#!/bin/bash

# Check for correct number of arguments.
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <scan_path> <results_folder>"
    exit 1
fi

SCAN_PATH="$1"
RESULTS_FOLDER="$2"

echo "Starting new pipeline..."
python final_run_program.py --scan_path "$SCAN_PATH" --results_folder "$RESULTS_FOLDER"