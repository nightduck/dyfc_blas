#!/bin/bash

# Usage: ./benchmark.sh [-p] [test1][:[N][:[M][:[K]]]] [test2] ...
# -p: Run tests in parallel

# To test the dgemm_rc test with matrices of 128x256 and 256x192, run
# ./benchmark.sh dgemm_rc:192:256:128

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
# TEST_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/test" && pwd)"
WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Clean up temp directory, in case a prior run of this script was interrupted
rm -rf "$WORKSPACE_DIR/temp"

# Check if Vitis is sourced
if ! command -v v++ &>/dev/null; then
  echo "Vitis is not sourced. Please source Vitis before running the script."
  exit 1
fi

# Get the list of command line arguments
if [ "$#" -eq 0 ]; then
  # No arguments passed, use the default test list
  TEST_LIST="daxpy ddot dgemv_rm dgemv_cm prefixsum dgemm_rcr dgemm_rcc"
  TEST_LIST=$(echo "$TEST_LIST" | sed "s~[^ ]*~$WORKSPACE_DIR/test/&~g")
else
  # Arguments passed, verify they match subdirectories in test/
  TEST_LIST=""
  SIZE_LIST=""
  for ARG in "$@"; do
    IFS=':' read -r DIR N M K <<<"$ARG"
    if [ -d "$WORKSPACE_DIR/test/$DIR" ]; then
      TEST_LIST="$TEST_LIST $WORKSPACE_DIR/test/$DIR"
      SIZE_LIST="$SIZE_LIST ${N}:${M}:${K}"
    else
      echo "!! Directory $DIR does not exist in $WORKSPACE_DIR/test."
    fi
  done
fi

# Function to run a single test
run_test() {
  local TEST="$1"
  local SIZE="$2"

  # Make temp directory for the test
  local TEMP_DIR="$WORKSPACE_DIR/temp/$TEST:$SIZE"
  mkdir -p "$TEMP_DIR"

  # Copy contents of test to the temp directory
  cp -r "$WORKSPACE_DIR/test/$TEST/"* "$TEMP_DIR/"

  local CONFIG_FILE="$TEMP_DIR/hls_config.cfg"
  
  # Clean up hls_config.cfg in case a prior run of this script was interrupted
  if tail -n 1 "$CONFIG_FILE" | grep -q "syn.cflags=-D dimK"; then
    sed -i '$d' "$CONFIG_FILE"
  fi
  if tail -n 1 "$CONFIG_FILE" | grep -q "tb.cflags=-D dimK"; then
    sed -i '$d' "$CONFIG_FILE"
  fi

  # Edit hls_config.cfg with the current m,n,k values in the format
  echo "tb.cflags=-D dimK=$K -D dimM=$M -D dimN=$N" >>"$CONFIG_FILE"
  echo "syn.cflags=-D dimK=$K -D dimM=$M -D dimN=$N" >>"$CONFIG_FILE"

  # Run the v++ command
  (cd "$TEMP_DIR" && v++ -c --mode hls --config "$CONFIG_FILE" --work_dir $TEMP_DIR/build >/dev/null 2>&1)

  # Check if the command was successful
  if [ $? -ne 0 ]; then
    echo "  !! $TEST_NAME failed"
  fi

  # Extract the relevant information from csynth.rpt
  CSYNTH_RPT="$TEMP_DIR/build/hls/syn/report/csynth.rpt"
  if [ -f "$CSYNTH_RPT" ]; then
    read latency_cycles latency_ns bram dsp ff lut uram issue_type < <(awk '
    BEGIN {
      hyphen_rows = 0;
      reading_table = 0;
    }
    {
      if ($0 ~ /\+-+\+/) {
        hyphen_rows++;
        if (reading_table == 1) {
          reading_table = 0;
          print latency_cycles, latency_ns, bram[1], dsp[1], ff[1], lut[1], uram[1], "";
          exit;
        }
      }
      if (hyphen_rows == 2 && reading_table == 0) {
        reading_table = 1;
        getline;
        split($0, columns, "|");
        latency_cycles = columns[5];
        latency_ns = columns[6];
        split(columns[11], bram, " ");
        split(columns[12], dsp, " ");
        split(columns[13], ff, " ");
        split(columns[14], lut, " ");
        split(columns[15], uram, " ");
      }
      if (hyphen_rows == 2 && reading_table == 1) {
        split($0, columns, "|");
        issue_type = columns[3];
        if (issue_type ~ /II/) {
          print latency_cycles, latency_ns, bram[1], dsp[1], ff[1], lut[1], uram[1], "II";
          exit;
        }
      }
    }
    ' "$CSYNTH_RPT")

    # Print the extracted values
    if [ ! -z "$issue_type" ]; then
      echo "  Issue Type: $issue_type"
    else
      echo "  Latency (cycles): $latency_cycles"
      echo "  Latency (ns): $latency_ns"
      echo "  BRAM: $bram"
      echo "  DSP: $dsp"
      echo "  FF: $ff"
      echo "  LUT: $lut"
      echo "  URAM: $uram"
    fi

    # Write the extracted values to the benchmark file
    if ([ -z "$issue_type" ]); then
      echo "$M,$N,$K,$latency_cycles,$latency_ns,$bram,$dsp,$ff,$lut,$uram" >> $TEMP_DIR/output.txt
    else
      echo "$M,$N,$K,$issue_type" >> $TEMP_DIR/output.txt
    fi
  else
    echo "!! csynth.rpt not found for $TEST_NAME."
  fi

  # Remove the last two lines from the config file that we added to specify the m,n,k values
  if tail -n 1 "$CONFIG_FILE" | grep -q "syn.cflags=-D dimK=$K -D dimM=$M -D dimN=$N"; then
    sed -i '$d' "$CONFIG_FILE"
  fi
  if tail -n 1 "$CONFIG_FILE" | grep -q "tb.cflags=-D dimK=$K -D dimM=$M -D dimN=$N"; then
    sed -i '$d' "$CONFIG_FILE"
  fi
}

# Create benchmarks directory if it doesn't exist, and clear it if it does
mkdir -p "$WORKSPACE_DIR"/benchmarks

# Iterate over all the tests in the test directory
TEST_ARRAY=($TEST_LIST)
SIZE_ARRAY=($SIZE_LIST)

# Run tests in parallel
for i in "${!TEST_ARRAY[@]}"; do
  TEST="${TEST_ARRAY[$i]}"
  SIZE="${SIZE_ARRAY[$i]}"

  if [ ! -d "$TEST" ]; then
    return
  fi

  TEST_NAME=$(basename "$TEST")

  IFS=':' read -r N_MAX M_MAX K_MAX <<<"$SIZE"

  # If a dimension isn't specified, get the range to text over from the test's header file
  if [ -z "$M_MAX" ]; then
    M_MIN=$(cat $TEST/$TEST_NAME.hpp | grep dimMSweepMin | cut -d' ' -f3)
    M_MAX=$(cat $TEST/$TEST_NAME.hpp | grep dimMSweepMax | cut -d' ' -f3)
    if [ -z "$M_MAX" ]; then
      M_MAX=1
      M_MIN=1
    fi
  else
    M_MIN=$M_MAX
  fi
  if [ -z "$N_MAX" ]; then
    N_MIN=$(cat $TEST/$TEST_NAME.hpp | grep dimNSweepMin | cut -d' ' -f3)
    N_MAX=$(cat $TEST/$TEST_NAME.hpp | grep dimNSweepMax | cut -d' ' -f3)
    if [ -z "$N_MAX" ]; then
      N_MAX=1
      N_MIN=1
    fi
  else
    N_MIN=$N_MAX
  fi
  if [ -z "$K_MAX" ]; then
    K_MIN=$(cat $TEST/$TEST_NAME.hpp | grep dimKSweepMin | cut -d' ' -f3)
    K_MAX=$(cat $TEST/$TEST_NAME.hpp | grep dimKSweepMax | cut -d' ' -f3)
    if [ -z "$K_MAX" ]; then
      K_MAX=1
      K_MIN=1
    fi
  else
    K_MIN=$K_MAX
  fi

  # Create csv file to store the results
  echo "M,N,K,Latency (cycles),Latency (ns),BRAM,DSP,FF,LUT,URAM" >"$WORKSPACE_DIR"/benchmarks/"$TEST_NAME".csv

  test_outputs=()
  i=0
  M=$M_MIN
  while [ "$M" -le "$M_MAX" ]; do
    N=$N_MIN
    while [ "$N" -le "$N_MAX" ]; do
      K=$K_MIN
      while [ "$K" -le "$K_MAX" ]; do
        echo "$TEST_NAME: M=$M, N=$N, K=$K"

        if [ "$PARALLEL" = true ]; then
          run_test "$TEST_NAME" "$M:$N:$K" &
          i=$((i + 1))
          if (( $(jobs -r | wc -l) >= PARALLEL_JOBS )); then
            wait -n
          fi
        else
          run_test "$TEST_NAME" "$M:$N:$K"
        fi

        # Increment K
        K=$((K * 2))
      done
      N=$((N * 2))
    done
    M=$((M * 2))
  done
  echo "Waiting for rest of $TEST_NAME jobs to finish"
  wait

  for d in $(ls -1v "$WORKSPACE_DIR"/temp | grep "$TEST_NAME"); do
    cat "$WORKSPACE_DIR/temp/$d/output.txt" >>"$WORKSPACE_DIR"/benchmarks/"$TEST_NAME".csv
  done
done

# Remove the temp directory
rm -rf "$WORKSPACE_DIR/temp"
