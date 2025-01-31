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

#include <cmath>
#include "daxpy.hpp"
#include "axpy.hpp"
#include "blas.hpp"
#include <cstdlib>
#include <stdio.h>


bool approximatelyEqual(double a, double b, double epsilon) {
    if (a > b) {
        return (a / b) - 1 <= epsilon;
    } else {
        return (b / a) - 1 <= epsilon;
    }
}

int main(int argc, char** argv) {
    double x[128];
    double y[128];
    double r[128];
    double r_gold[128];
    double alpha;

    // Initialize variables
    srand(0xDEADBEEF);
    alpha = (rand() % 100) / (rand() % 100);
    for(int i = 0; i < 128; i++) {
        x[i] = (rand() % 100) / (rand() % 100);
        y[i] = (rand() % 100) / (rand() % 100);
        r_gold[i] = alpha * x[i] + y[i];
    }

    // Make call to kernel
    daxpy(128, alpha, x, y, r);

    // Verify results
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
