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
#include "ddot.hpp"

#define RANDOM (double)(rand() % 100) / (double)(rand() % 100 + 1)

bool approximatelyEqual(double a, double b, double epsilon) {
  if (a > b) {
    return (a / b) - 1 <= epsilon;
  } else if (a < b) {
    return (b / a) - 1 <= epsilon;
  } else {
    return true;
  }
}

int main(int argc, char** argv) {
  double x[dimN];
  double y[dimN];
  double r;
  double r_gold;

  // Initialize variables with random floats
  srand(0xDEADBEEF);
  for (int i = 0; i < dimN; i++) {
    x[i] = RANDOM;
  }

  // Compute the correct result to compare against (this fictitious function is an element-wise
  // add along the diagonal)
  r_gold = 0;
  for (int i = 0; i < dimN; i++) {
    r_gold += x[i] * y[i];
  }

  // Make call to kernel
  ddot(x, y, r);

  // Verify results. Due to potential floating point error, we need to use an approximate comparison
  int failed_index = -1;
  double epsilon = dimN / (1e-17);  // asum in particular is very bad at accumulating errors over
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
