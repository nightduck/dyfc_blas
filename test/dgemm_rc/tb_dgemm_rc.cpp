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
#include "dgemm_rc.hpp"

#define RANDOM (double)(rand() % 100 - 50) / (double)(rand() % 100 + 1)

bool approximatelyEqual(double a, double b, double epsilon) {
  if (a > b) {
    return (a / b) - 1 <= epsilon;
  } else if (a < b) {
    return (b / a) - 1 <= epsilon;
  } else {
    return true;
  }
}

void print_vector(double *v, int n) {
  for (int i = 0; i < n; i++) {
    printf("%f ", v[i]);
  }
  printf("\n");
  printf("\n");
}

void print_matrix(double *A, int m, int n) {
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      printf("%f ", A[i * n + j]);
    }
    printf("\n");
  }
  printf("\n");
}

int main(int argc, char** argv) {
  double alpha;
  double A[dimM][dimK];
  double B[dimN][dimK];   // B is stored in col-major order
  double beta;
  double C[dimM][dimN];
  double r[dimM][dimN];
  double r_gold[dimM][dimN];

  // Initialize variables with random floats
  srand(0xDEADBEEF);
  alpha = RANDOM;
  beta = RANDOM;
  for (int i = 0; i < dimM; i++) {
    for (int j = 0; j < dimK; j++) {
      A[i][j] = RANDOM;
    }
  }
  for (int i = 0; i < dimK; i++) {
    for (int j = 0; j < dimN; j++) {
      B[i][j] = RANDOM;
    }
  }
  for (int i = 0; i < dimM; i++) {
    for (int j = 0; j < dimN; j++) {
      C[i][j] = RANDOM;
    }
  }

  // Compute the correct result to compare against
  for (int i = 0; i < dimM; i++) {
    for (int j = 0; j < dimN; j++) {
      r_gold[i][j] = beta * C[i][j];
      for (int k = 0; k < dimK; k++) {
        r_gold[i][j] += alpha * A[i][k] * B[j][k];
      }
    }
  }
  

  // Make call to kernel
  dgemm_rc(alpha, A, B, beta, C, r);

  // Verify results. Due to potential floating point error, we need to use an approximate comparison
  int failed_index = -1;
  for (int i = 0; i < dimM; i++) {
    for (int j = 0; j < dimN; j++) {
      if (!approximatelyEqual(r[i][j], r_gold[i][j], 1e-9)) {
        failed_index = i * dimN + j;
        break;
      }
    }
  }

  if (failed_index > -1) {
    int i = failed_index / dimN;
    int j = failed_index % dimN;
    std::cout << "FAILED TEST" << std::endl;
    std::cout << "r[" << i << "][" << j << "] (" << r[i][j] << ") != "
              << "r_gold[" << i << "][" << j << "] (" << r_gold[i][j] << ")" << std::endl;
    return -1;
  } else {
    std::cout << "PASSED TEST" << std::endl;
    return 0;
  }
}
