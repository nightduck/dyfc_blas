#!/bin/bash

# Directory containing the tests
TEST_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/test" && pwd)"

# Iterate over all the tests in the test directory
for TEST in "$TEST_DIR"/*; do
  if [ -d "$TEST" ]; then
    TEST_NAME=$(basename "$TEST")
    CONFIG_FILE="$TEST/hls_config.cfg"
    
    # TODO: Open some benchmark configuration file for the test that indicates which range of m,n,k to test with
    # TODO: 3 nested for loops to iterate over m,n,k

    # TODO: Edit hls_config.cfg with the current m,n,k values in the format
      # tb.cflags=-D dimK=64 -D dimM=64 -D dimN=64
      # syn.cflags=-D dimK=64 -D dimM=64 -D dimN=64

    # Run the v++ command
    (cd "$TEST" && v++ -c --mode hls --config "$CONFIG_FILE" --work_dir $TEST/build > /dev/null 2>&1)
    
    # Check if the command was successful
    if [ $? -ne 0 ]; then
      echo "!! Test $TEST_NAME failed."
    else
      echo "Test $TEST_NAME passed."
    fi

    # TODO: Check the csynth.rpt file for the latency and throughput.

    # TODO: Indicate if there's any II errors in the synthesis.

    # TODO: Write the n,m,k values next to the latency into a csv file.
  fi
done