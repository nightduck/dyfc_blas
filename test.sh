#!/bin/bash

# Usage: ./test.sh [test1] [test2] ...

# Parse command line flags
PARALLEL=false
PARALLEL_JOBS=$(nproc)
while getopts "p:" opt; do
  case $opt in
  p) PARALLEL=true
     if [[ "$OPTARG" =~ ^[0-9]+$ ]]; then
       PARALLEL_JOBS=$OPTARG
     else
       echo "Invalid number: $OPTARG" >&2
       echo "-p must be followed by a positive integer" >&2
       exit 1
     fi ;;
  \?)
    echo "Invalid option -$OPTARG" >&2
    exit 1
    ;;
  esac
done
shift $((OPTIND - 1))

# Directory containing the tests
TEST_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/test" && pwd)"

# Get the list of command line arguments
if [ "$#" -eq 0 ]; then
  # No arguments passed, use all subdirectories in TEST_DIR (except logs and .theia)
  TEST_LIST=$(find "$TEST_DIR" -mindepth 1 -maxdepth 1 -type d)
  TEST_LIST=$(echo "$TEST_LIST" | grep -v "logs" | grep -v ".theia")
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

run_test() {
  # Run the csim command
  (cd "$TEST" && vitis-run --mode hls --csim --config "$CONFIG_FILE" --work_dir $TEST/build > /dev/null 2>&1)
  
  # Check if the command was successful
  if [ $? -ne 0 ]; then
    echo "!! Test $TEST_NAME failed."
  else
    echo "Test $TEST_NAME passed."
  fi
}

# Iterate over all the tests in the test directory
for TEST in $TEST_LIST; do
  if [ -d "$TEST" ]; then
    TEST_NAME=$(basename "$TEST")
    CONFIG_FILE="$TEST/hls_config.cfg"
    
    if [ "$PARALLEL" = true ]; then
      run_test "$TEST_NAME" "$M:$N:$K" &
      if (( $(jobs -r | wc -l) >= PARALLEL_JOBS )); then
        wait -n
      fi
    else
      run_test "$TEST_NAME" "$M:$N:$K"
    fi
  fi
done

wait