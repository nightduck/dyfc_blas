// File was copied from https://github.com/UofT-HPRC/hlstl, which didn't have a license
// at the time. Double check there in case that's changed.

#ifndef ASSERT_HPP
#define ASSERT_HPP

#include <execinfo.h>

#include <cassert>
#include <cstdlib>
#include <iostream>
// #include <stacktrace>

#ifdef __RTL_SIMULATION__
#define RTL_SIMULATION_DEFINED 1
#else
#define RTL_SIMULATION_DEFINED 0
#endif
#ifdef __SYNTHESIS__
#define SYNTHESIS_DEFINED 1
#else
#define SYNTHESIS_DEFINED 0
#endif

#if (SYNTHESIS_DEFINED == 0)
#define ASSERT(condition, message)                             \
  if (!(condition)) {                                          \
    std::cout << "ASSERTION FAILED: " << message << std::endl; \
    assert(false);                                             \
  }

#define ASSERT_IF(condition_when_to_check_assertion, assertion_condition, message) \
  if ((condition_when_to_check_assertion)) {                                       \
    if (!(assertion_condition)) {                                                  \
      std::cout << "ASSERTION FAILED: " << message << std::endl;                   \
      std::cout << message << std::endl;                                           \
      assert(false);                                                               \
    }                                                                              \
  }

#define UNIMPLEMENTED()                               \
  std::cout << "UNIMPLEMENTED FUNCTION" << std::endl; \
  assert(false);

#define UNIMPLEMENTED_QUIET(message)                  \
  std::cout << "UNIMPLEMENTED FUNCTION" << std::endl; \
  std::cout << "\t" << message << std::endl;

#define INFO(message) std::cout << message << std::endl;
#else  // DEBUG
#define ASSERT(condition, message) {}
#define DO_PRAGMA_INNER(x) _Pragma(#x)
#define DO_PRAGMA(x) DO_PRAGMA_INNER(x)
#define ASSERT_IF(condition_when_to_check_assertion, assertion_condition, message) {}

#define INFO(message) {}
#define UNIMPLEMENTED() {}
#define UNIMPLEMENTED_QUIET(messsage) {}
#endif  // DEBUG

#endif