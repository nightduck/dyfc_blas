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
#include "transpose.hpp"

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
  // Row major means mxn is rows x cols
  // Col major means mxn is cols x rows
  double A[dimN][dimM];
  double AT[dimM][dimN];    // Transpose of A, computed on host
  double ArTc[dimN][dimM];  // Read as row-major, transposed, and written as col-major
  double ArTr[dimM][dimN];  // Read as row-major, transposed, and written as row-major
  double AcTr[dimN][dimM];  // Read as col-major, transposed, and written as row-major
  double AcTc[dimM][dimN];  // Read as col-major, transposed, and written as col-major

  // Initialize variables with random floats
  srand(0xDEADBEEF);
  for (int i = 0; i < dimN; i++) {
    for (int j = 0; j < dimM; j++) {
      A[i][j] = RANDOM;
      AT[j][i] = A[i][j];
    }
  }

  // Make call to kernel
  transpose_krnl(A, ArTc, ArTr, AcTr, AcTc);

  // Verify results. Due to potential floating point error, we need to use an approximate comparison
  int fail = -1;
  for (int i = 0; i < dimN; i++) {
    for (int j = 0; j < dimM; j++) {
      if (!approximatelyEqual(A[i][j], ArTc[i][j], 1e-9)) {
        std::cout << "FAILED TEST" << std::endl;
        std::cout << "A/ArTc[" << i << "][" << j << "] = " << A[i][j] << " != " << ArTc[i][j] << std::endl;
        fail = 1;
        return -1;
      }
      // if (!approximatelyEqual(AT[j][i], ArTr[j][i], 1e-9)) {
      //   fail = 1;
      //   std::cout << "FAILED TEST" << std::endl;
      //   std::cout << "AT/ArTr[" << j << "][" << i << "] = " << AT[j][i] << " != " << ArTr[j][i] << std::endl;
      //   return -1;
      // }
      if (!approximatelyEqual(A[i][j], AcTr[i][j], 1e-9)) {
        fail = 1;
        std::cout << "FAILED TEST" << std::endl;
        std::cout << "A/AcTr[" << i << "][" << j << "] = " << A[i][j] << " != " << AcTc[i][j] << std::endl;
        return -1;
      }
      // if (!approximatelyEqual(AT[j][i], AcTc[j][i], 1e-9)) {
      //   fail = 1;
      //   std::cout << "FAILED TEST" << std::endl;
      //   std::cout << "AT/AcTc[" << j << "][" << i << "] = " << AT[j][i] << " != " << AcTr[j][i] << std::endl;
      //   return -1;
      // }
    }
  }

  if (fail < 0) {
    std::cout << "PASSED TEST" << std::endl;
    return 0;
  }
}
