#!/bin/bash

BASEDIR=$(dirname "$0")

# Generate the compilation database file.
cmake -S $BASEDIR/../ -B $BASEDIR/../build -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DPYBIND=1

# Move to home directory
mv "$BASEDIR/../build/compile_commands.json" "$BASEDIR/../compile_commands.json"
