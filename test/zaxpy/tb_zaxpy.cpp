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
#include "zaxpy.hpp"

#define COMPLEX_RANDOM                                                    \
  ComplexDouble((double)(rand() % 100 - 50) / (double)(rand() % 100 + 1), \
                (double)(rand() % 100 - 50) / (double)(rand() % 100 + 1))

bool approximatelyEqual(ComplexDouble a, ComplexDouble b, double epsilon) {
  bool real_close = (a.real == b.real) || std::abs(a.real / b.real - 1) <= epsilon;
  bool imag_close = (a.imag == b.imag) || std::abs(a.imag / b.imag - 1) <= epsilon;

  return real_close && imag_close;
}

int main(int argc, char** argv) {
  ComplexDouble alpha;
  ComplexDouble x[dimN];
  ComplexDouble y[dimN];
  ComplexDouble r[dimN];
  ComplexDouble r_gold[dimN];

  // Initialize variables with random floats
  srand(0xDEADBEEF);
  alpha = COMPLEX_RANDOM;
  for (int i = 0; i < dimN; i++) {
    x[i] = COMPLEX_RANDOM;
    y[i] = COMPLEX_RANDOM;
  }

  // Compute the correct result to compare against (this fictitious function is an element-wise
  // add along the diagonal)
  for (int i = 0; i < dimN; i++) {
    r_gold[i] = alpha * x[i] + y[i];
  }

  // Make call to kernel
  zaxpy(alpha, x, y, r);

  // Verify results. Due to potential floating point error, we need to use an approximate comparison
  int failed_index = -1;
  for (int i = 0; i < dimN; i++) {
    if (!approximatelyEqual(r[i], r_gold[i], 1e-9)) {
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
