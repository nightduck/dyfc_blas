#!/bin/bash

# Usage: ./test.sh [test1] [test2] ...

# Directory containing the tests
TEST_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/test" && pwd)"

# Get the list of command line arguments
if [ "$#" -eq 0 ]; then
  # No arguments passed, use all subdirectories in TEST_DIR
  TEST_LIST=$(find "$TEST_DIR" -mindepth 1 -maxdepth 1 -type d)
else
  # Arguments passed, verify they match subdirectories in TEST_DIR
  TEST_LIST=""
  for ARG in "$@"; do
    if [ -d "$TEST_DIR/$ARG" ]; then
      TEST_LIST="$TEST_LIST $TEST_DIR/$ARG"
    else
      echo "!! Directory $ARG does not exist in $TEST_DIR."
    fi
  done
fi

# Iterate over all the tests in the test directory
for TEST in $TEST_LIST; do
  if [ -d "$TEST" ]; then
    TEST_NAME=$(basename "$TEST")
    CONFIG_FILE="$TEST/hls_config.cfg"
    
    # Run the csim command
    (cd "$TEST" && vitis-run --mode hls --csim --config "$CONFIG_FILE" --work_dir $TEST/build > /dev/null 2>&1)
    
    # Check if the command was successful
    if [ $? -ne 0 ]; then
      echo "!! Test $TEST_NAME failed."
    else
      echo "Test $TEST_NAME passed."
    fi
  fi
done