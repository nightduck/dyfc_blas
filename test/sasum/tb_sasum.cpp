// Copyright 2025 Oren Bell
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <stdio.h>

#include <cmath>
#include <cstdlib>

#include "blas.hpp"
#include "sasum.hpp"

#define RANDOM (float)(rand() % 100 - 50) / (float)(rand() % 100 + 1)

bool approximatelyEqual(float a, float b, double epsilon) {
  return (a == b) || std::abs(a / b - 1) <= epsilon;
}

int main(int argc, char** argv) {
  float* x = (float*)malloc(dimN * sizeof(float));
  float r;
  float r_gold;

  // Initialize variables with random floats
  srand(0xDEADBEEF);
  for (int i = 0; i < dimN; i++) {
    x[i] = RANDOM;
  }

  // Compute the correct result to compare against (this fictitious function is an element-wise
  // add along the diagonal)
  r_gold = 0;
  for (int i = 0; i < dimN; i++) {
    r_gold += std::abs(x[i]);
  }

  // Make call to kernel
  sasum(x, r);

  // Verify results. Due to potential floating point error, we need to use an approximate comparison
  int failed_index = -1;
  float epsilon = dimN / (1e-9);  // asum in particular is very bad at accumulating errors over
                                  // large datasets. Different implementations can create very
                                  // different answers
  if (!approximatelyEqual(r, r_gold, epsilon)) {
    failed_index = 0;
  }

  if (failed_index > -1) {
    std::cout << "FAILED TEST" << std::endl;
    std::cout << "r[" << failed_index << "] (" << r << ") != "
              << "r_gold[" << failed_index << "] (" << r_gold << ")" << std::endl;
    return -1;
  } else {
    std::cout << "PASSED TEST" << std::endl;
    return 0;
  }
}
