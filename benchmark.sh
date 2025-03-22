#!/bin/bash

# Usage: ./benchmark.sh [-p] [test1][:[N][:[M][:[K]]]] [test2] ...
# -p: Run tests in parallel

# To test the dgemm_rc test with matrices of 128x256 and 256x192, run
# ./benchmark.sh dgemm_rc:192:256:128

# Parse command line flags
PARALLEL=false
while getopts "p" opt; do
  case $opt in
  p) PARALLEL=true ;;
  \?)
    echo "Invalid option -$OPTARG" >&2
    exit 1
    ;;
  esac
done
shift $((OPTIND - 1))

# Directory containing the tests
TEST_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/test" && pwd)"

# Check if Vitis is sourced
if ! command -v v++ &>/dev/null; then
  echo "Vitis is not sourced. Please source Vitis before running the script."
  exit 1
fi

# Get the list of command line arguments
if [ "$#" -eq 0 ]; then
  # No arguments passed, use the default test list
  TEST_LIST="daxpy ddot dgemv_rm dgemv_cm prefixsum dgemm_rc"
  TEST_LIST=$(echo "$TEST_LIST" | sed "s~[^ ]*~$TEST_DIR/&~g")
else
  # Arguments passed, verify they match subdirectories in TEST_DIR
  TEST_LIST=""
  SIZE_LIST=""
  for ARG in "$@"; do
    IFS=':' read -r DIR N M K <<<"$ARG"
    if [ -d "$TEST_DIR/$DIR" ]; then
      TEST_LIST="$TEST_LIST $TEST_DIR/$DIR"
      SIZE_LIST="$SIZE_LIST ${N}:${M}:${K}"
    else
      echo "!! Directory $DIR does not exist in $TEST_DIR."
    fi
  done
fi

# Function to run a single test
run_test() {
  local TEST="$1"
  local SIZE="$2"

  if [ ! -d "$TEST" ]; then
    return
  fi

  local TEST_NAME=$(basename "$TEST")
  local CONFIG_FILE="$TEST/hls_config.cfg"

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
  echo "M,N,K,Latency (cycles),Latency (ns),BRAM,DSP,FF,LUT,URAM" >"$TEST_DIR"/../benchmarks/"$TEST_NAME".csv

  M=$M_MIN
  while [ "$M" -le "$M_MAX" ]; do
    N=$N_MIN
    while [ "$N" -le "$N_MAX" ]; do
      K=$K_MIN
      while [ "$K" -le "$K_MAX" ]; do
        echo "$TEST_NAME: M=$M, N=$N, K=$K"

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

        # TODO: Have a flag that lets each size permutation of each test run in parallel. They would go into
        # different folders, eg build64:64:64, build64:64:128, etc. This line (which is 99% of the compute time)
        # would need to go into a different bash function. You would also need to deal with each size permutation
        # writing to the same csv file. Maybe have the lines below return the line they're contributing to the csv
        # file and this function will aggregate them. It can submit multiple jobs to the background, one for each
        # size permutation. It will wait on the output of each job in order to aggregate the results. Meanwhile,
        # other tests that also called this function would be submitting their own jobs to run. Something in this script
        # has to set a limit on how many jobs can run in parallel and coordinate everything.

        # Run the v++ command
        (cd "$TEST" && v++ -c --mode hls --config "$CONFIG_FILE" --work_dir $TEST/build >/dev/null 2>&1)

        # Check if the command was successful
        if [ $? -ne 0 ]; then
          echo "  !! $TEST_NAME failed"
        fi

        # Extract the relevant information from csynth.rpt
        CSYNTH_RPT="$TEST/build/hls/syn/report/csynth.rpt"
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
            echo "$M,$N,$K,$latency_cycles,$latency_ns,$bram,$dsp,$ff,$lut,$uram" >>"$TEST_DIR"/../benchmarks/"$TEST_NAME".csv
          else
            echo "$M,$N,$K,$issue_type" >>"$TEST_DIR"/../benchmarks/"$TEST_NAME".csv
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

        # Increment K
        K=$((K * 2))
      done
      N=$((N * 2))
    done
    M=$((M * 2))
  done
}

# Create benchmarks directory if it doesn't exist, and clear it if it does
mkdir -p "$TEST_DIR"/../benchmarks

# Iterate over all the tests in the test directory
TEST_ARRAY=($TEST_LIST)
SIZE_ARRAY=($SIZE_LIST)

if [ "$PARALLEL" = true ]; then
  # Run tests in parallel
  for i in "${!TEST_ARRAY[@]}"; do
    TEST="${TEST_ARRAY[$i]}"
    SIZE="${SIZE_ARRAY[$i]}"
    run_test "$TEST" "$SIZE" &
  done
  # Wait for all parallel jobs to complete
  wait
else
  # Run tests sequentially
  for i in "${!TEST_ARRAY[@]}"; do
    TEST="${TEST_ARRAY[$i]}"
    SIZE="${SIZE_ARRAY[$i]}"
    run_test "$TEST" "$SIZE"
  done
fi
