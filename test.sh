#!/bin/bash

# Directory containing the tests
TEST_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/test" && pwd)"

# Iterate over all the tests in the test directory
for TEST in "$TEST_DIR"/*; do
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