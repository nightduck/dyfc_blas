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
#include "sgemv_cm.hpp"

#define RANDOM (float)(rand() % 100 - 50) / (float)(rand() % 100 + 1)

bool approximatelyEqual(float a, float b, double epsilon) {
  return (a == b) || std::abs(a/b - 1) <= epsilon;
}

void print_vector(float *v, int n) {
  for (int i = 0; i < n; i++) {
    printf("%f ", v[i]);
  }
  printf("\n");
  printf("\n");
}

void print_matrix(float *A, int m, int n) {
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      printf("%f ", A[i * n + j]);
    }
    printf("\n");
  }
  printf("\n");
}

int main(int argc, char** argv) {
  float alpha;
  float A[dimN][dimM];
  float x[dimN];
  float beta;
  float y[dimN];
  float r[dimN];
  float r_gold[dimN];

  // Initialize variables with random floats
  srand(0xDEADBEEF);
  alpha = RANDOM;
  beta = RANDOM;
  for (int i = 0; i < dimN; i++) {
    x[i] = RANDOM;
    for (int j = 0; j < dimM; j++) {
      A[i][j] = RANDOM;
    }
  }
  for (int i = 0; i < dimM; i++) {
    y[i] = RANDOM;
  }

  // Compute the correct result to compare against (this fictitious function is an element-wise
  // add along the diagonal)
  for (int i = 0; i < dimM; i++) {
    r_gold[i] = beta * y[i];
  }
  for (int i = 0; i < dimN; i++) {
    for (int j = 0; j < dimM; j++) {
      r_gold[j] += alpha * A[i][j] * x[i];
    }
  }

  // Make call to kernel
  sgemv_cm(alpha, A, x, beta, y, r);

  // Verify results. Due to potential floating point error, we need to use an approximate comparison
  int failed_index = -1;
  double epsilon =
      std::max(dimN, dimM) / (1e-9);  // asum in particular is very bad at accumulating errors over
                                      // large datasets. Different implementations can create very
                                      // different answers
  for (int i = 0; i < dimN; i++) {
    if (!approximatelyEqual(r[i], r_gold[i], epsilon)) {
      failed_index = i;
      break;
    }
  }

  if (failed_index > -1) {
    std::cout << "FAILED TEST" << std::endl;
    std::cout << "r[" << failed_index << "] (" << r[failed_index] << ") != "
              << "r_gold[" << failed_index << "] (" << r_gold[failed_index] << ")" << std::endl;
    return -1;
  } else {
    std::cout << "PASSED TEST" << std::endl;
    return 0;
  }
}
