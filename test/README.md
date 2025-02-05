# Tests

This contains a set of tests for the BLAS library. Each test corresponds to a particular
specialization of a blas function. For instance the axpy function, that adds two vectors, has four
specializations: saxpy, daxpy, caxpy, and zaxpy for single precision real, double precision real,
single precision complex, and double precision complex numbers. Each of these four specializations
will have it's own unique test.

## Unit Tests

The primary goal of the tests is to verify the correctness of the BLAS library. This is done for
each test with the following (executed from the build directory)

  vitis-run --mode hls --csim --config ../test/TEST_NAME/hls_config.cfg --work_dir TEST_NAME

The program will return a zero if the C simulation passed. This confirms programmatic correctness

## Benchmarking

These tests also serve to benchmark the BLAS library functions by synthesizing the tests for
different input sizes

IN PROGRESS: The input sizes can only be specified in the hls_config file. See the two lines
passing the dimN, dimM, and dimK defines as cflags. We need some way to automatically modify the
config file between successive tests. Each test also needs to extract the latency information from
csynth.rtpt file. Preferably both the unit as a whole, as well as just the call to the API, to give
us a sense of memory overhead.

## Development

An example test is provided purely to serve as a template for implementing future tests.

It consists of 4 files:

* EXAMPLE.hpp - Stores the function declaration for the kernel and defines size info if not done already
* EXAMPLE.cpp - The kernel function definition
* tb_EXAMPLE.cpp - The host code to run the test
* hls_config.cpp - The HLS config file

Required changes to the template when making a new test:

Generally
* Update the names of the files to reflect the new test

EXAMPLE.hpp
* Update the names of the include headers, matching the format of the new test
* Update the kernel function prototype

EXAMPLE.cpp
* Update the name of the include file
* Update the function signature
* Modify the Vector and Matrix objects to reflect the arguments being loaded in
* Change the call to invoke the right BLAS API function
* Update the write-back code, if relevant

tb_EXAMPLE.cpp
* Update the name of the include file
* Change the local variables to reflect what's being passed to the BLAS call
* Modify the initialization code appropriately to seed initial values
* Change the computation of a correct gold standard
* Change the call to the kernel function.
* Modify the test verification, if relevant

hls_config.cfg
* Change tb.file and syn.file
* Change filepath in two cflags lines
* Change syn.top to match function name of kernel
