#!/bin/bash

# Loop through each directory in the test directory
for dir in ./test/*/; do
  # Check if it is a directory
  if [ -d "$dir" ]; then
    echo "Cleaning directory: $dir"
    # Add your cleaning commands here, for example:
    rm -rf "$dir"/.Xil
    rm -rf "$dir"/build
    rm -rf "$dir"/vitis-comp.json
  fi
done