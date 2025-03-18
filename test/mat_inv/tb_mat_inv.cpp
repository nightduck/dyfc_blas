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
#include "mat_inv.hpp"

#define RANDOM (double)(rand() % 100 - 50) / (double)(rand() % 100 + 1)

bool approximatelyEqual(double a, double b, double epsilon) {
  return (a == b) || std::abs(a/b - 1) <= epsilon;
}

int main(int argc, char** argv) {
  double x[dimN][dimN];
  double r[dimN][dimN];
  double r_gold[dimN][dimN];

  // Initialize variables with random floats
  srand(0xDEADBEEF);
  for (int i = 0; i < dimN; i++) {
    for (int j = 0; j < dimN; j++) {
      x[i][j] = RANDOM;
      if (i == j) x[i][j]; // Little trick to guarantee it's invertible
    }
  }

  // Create a load a matrix that is left half x and right half identity
  double gjordan[dimN][2*dimN];
  for (int i = 0; i < dimN; i++) {
    for (int j = 0; j < dimN; j++) {
      gjordan[i][j] = x[i][j];
      if (i == j) {
        gjordan[i][j + dimN] = 1;
      } else {
        gjordan[i][j + dimN] = 0;
      }
    }
  }

  // Compute the correct result with gaussian elimination
  for (int i = 0; i < dimN; i++) {
  std::cout << "Gold answer: " << std::endl;
  for (int i = 0; i < dimN; i++) {
    for (int j = 0; j < 2*dimN; j++) {
        std::cout << std::setw(12) << gjordan[i][j] << " ";
    }
    std::cout << std::endl;
  }
    for (int j = 0; j < dimN; j++) {
      double alpha = gjordan[j][i];
      double beta = gjordan[i][i];
      for (int k = i; k < 2*dimN; k++) {
        if (j < i) {
          gjordan[j][k] -= (gjordan[i][k] * alpha / beta);
        } else if (i == j) {
          gjordan[j][k] /= alpha;
        } else {
          gjordan[j][k] -= (gjordan[i][k] * alpha);
        }
      }
    }
  }

  // Copy the right half of the matrix to the gold result
  for (int i = 0; i < dimN; i++) {
    for (int j = 0; j < dimN; j++) {
      r_gold[i][j] = gjordan[i][j + dimN];
    }
  }

  // Make call to kernel
  mat_inv(x, r);

  // Verify results. Due to potential floating point error, we need to use an approximate comparison
  int failed_index = -1;
  for (int i = 0; i < dimN; i++) {
    for (int j = 0; j < dimN; j++) {
      if (!approximatelyEqual(r[i][j], r_gold[i][j], 1e-9)) {
        failed_index = i * dimN + j;
        break;
      }
    }
  }

  std::cout << "Input: " << std::endl;
  for (int i = 0; i < dimN; i++) {
    for (int j = 0; j < dimN; j++) {
        std::cout << std::setw(12) << x[i][j] << " ";
    }
    std::cout << std::endl;
  }
  std::cout << "Computed: " << std::endl;
  for (int i = 0; i < dimN; i++) {
    for (int j = 0; j < dimN; j++) {
        std::cout << std::setw(12) << r[i][j] << " ";
    }
    std::cout << std::endl;
  }
  std::cout << "Gold answer: " << std::endl;
  for (int i = 0; i < dimN; i++) {
    for (int j = 0; j < dimN; j++) {
        std::cout << std::setw(12) << r_gold[i][j] << " ";
    }
    std::cout << std::endl;
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
