#!/bin/bash


executable="python range_test.py"

# Check if arguments are passed
if [ $# -eq 0 ]; then
    echo "No arguments provided"
    exit 1
fi

# Iterate over each argument
for arg in "$@"; do
    # Call the executable with the current argument
    $executable "$arg"
done
