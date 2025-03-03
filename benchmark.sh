#!/bin/bash

# Usage: ./benchmark.sh [test1][:[N][:[M][:[K]]]] [test2] ...

# To test the dgemm_rc test with matrices of 128x256 and 256x192, run
# ./benchmark.sh dgemm_rc:192:256:128

# Directory containing the tests
TEST_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/test" && pwd)"

# Check if Vitis is sourced
if ! command -v v++ &> /dev/null; then
  echo "Vitis is not sourced. Please source Vitis before running the script."
  exit 1
fi

# Get the list of command line arguments
if [ "$#" -eq 0 ]; then
  # No arguments passed, use all subdirectories in TEST_DIR
  TEST_LIST=$(find "$TEST_DIR" -mindepth 1 -maxdepth 1 -type d)
else
  # Arguments passed, verify they match subdirectories in TEST_DIR
  TEST_LIST=""
  SIZE_LIST=""
  for ARG in "$@"; do
    IFS=':' read -r DIR N M K <<< "$ARG"
    if [ -d "$TEST_DIR/$DIR" ]; then
      TEST_LIST="$TEST_LIST $TEST_DIR/$DIR"
      SIZE_LIST="$SIZE_LIST ${N}:${M}:${K}"
    else
      echo "!! Directory $DIR does not exist in $TEST_DIR."
    fi
  done
fi

# Create benchmarks directory if it doesn't exist, and clear it if it does
mkdir -p "$TEST_DIR"/../benchmarks
rm -r "$TEST_DIR"/../benchmarks/*

# Iterate over all the tests in the test directory
TEST_ARRAY=($TEST_LIST)
SIZE_ARRAY=($SIZE_LIST)

for i in "${!TEST_ARRAY[@]}"; do
  TEST="${TEST_ARRAY[$i]}"
  SIZE="${SIZE_ARRAY[$i]}"
  
  if [ -d "$TEST" ]; then
    TEST_NAME=$(basename "$TEST")
    CONFIG_FILE="$TEST/hls_config.cfg"
    
    IFS=':' read -r N_MAX M_MAX K_MAX <<< "$SIZE"

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

    M=$M_MIN
    while [ "$M" -le "$M_MAX" ]; do
      N=$N_MIN
      while [ "$N" -le "$N_MAX" ]; do
      K=$K_MIN
      while [ "$K" -le "$K_MAX" ]; do
        echo "$TEST_NAME: M=$M, N=$N, K=$K"

        # Edit hls_config.cfg with the current m,n,k values in the format
        echo "tb.cflags=-D dimK=$K -D dimM=$M -D dimN=$N" >> "$CONFIG_FILE"
        echo "syn.cflags=-D dimK=$K -D dimM=$M -D dimN=$N" >> "$CONFIG_FILE"

        # Run the v++ command
        (cd "$TEST" && v++ -c --mode hls --config "$CONFIG_FILE" --work_dir $TEST/build > /dev/null 2>&1)
        
        # Check if the command was successful
        if [ $? -ne 0 ]; then
          echo "  !! Test failed"
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
              print latency_cycles, latency_ns, bram[1], dsp[1], ff[1], lut[1], uram[1], "";
            }
            if (hyphen_rows == 2 && reading_table == 1) {
              if ($0 ~ /\+-+\+/) {
                reading_table = 0;
                exit;
              }
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

          if ([ -z "$issue_type" ]); then
            echo "$M,$N,$K,$latency_cycles,$latency_ns,$bram,$dsp,$ff,$lut,$uram" >> "$TEST_DIR"/../benchmarks/"$TEST"_benchmark.csv
          else
            echo "$M,$N,$K,$issue_type" >> "$TEST_DIR"/../benchmarks/"$TEST"_benchmark.csv
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

        sleep 1
        # Increment K
        K=$((K * 2))
      done
      N=$((N * 2))
      done
      M=$((M * 2))
    done
  fi
done