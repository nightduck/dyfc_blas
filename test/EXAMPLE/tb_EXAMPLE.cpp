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
#include "EXAMPLE.hpp"

#define RANDOM (rand() % 100) / (rand() % 100)

bool approximatelyEqual(double a, double b, double epsilon) {
  if (a > b) {
    return (a / b) - 1 <= epsilon;
  } else {
    return (b / a) - 1 <= epsilon;
  }
}

int main(int argc, char** argv) {
  double alpha;
  double x[N];
  double A[N][N];
  double r[N];
  double r_gold[N];

  // Initialize variables with random floats
  srand(0xDEADBEEF);
  alpha = RANDOM;
  for (int i = 0; i < N; i++) {
    x[i] = RANDOM;
    for (int j = 0; j < N; j++) {
      A[i][j] = RANDOM;
    }
  }

  // Compute the correct result to compare against (this fictitious function is an element-wise
  // add along the diagonal)
  for (int i = 0; i < N; i++) {
    r_gold[i] = x[j] + A[i][i];
  }

  // Make call to kernel
  dexample_cm(alpha, x, A, r);

  // Verify results. Due to potential floating point error, we need to use an approximate comparison
  bool fail = false;
  for (int i = 0; i < 128; i++) {
    if (!approximatelyEqual(r[i], r_gold[i], 1e-9)) {
      fail = true;
      break;
    }
  }

  if (fail) {
    std::cout << "FAILED TEST" << std::endl;
    return -1;
  } else {
    std::cout << "PASSED TEST" << std::endl;
    return 0;
  }
}
