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

#ifndef DYFC_BLAS_TYPES_HPP
#define DYFC_BLAS_TYPES_HPP

#include <ap_int.h>
#include <hls_stream.h>
#include <hls_vector.h>

#define MAX_BITWIDTH 4096

namespace dyfc {
namespace blas {

enum MajorOrder { RowMajor, ColMajor };

enum UpperLower { Upper, Lower };

// Alias to avoid confusing hls vector with blas vector
template <typename T, unsigned int Par> using WideType = hls::vector<T, Par>;

// NOTE: A design decision was made to include the vector and matrix dimensions
// as class members
//    instead of template parameters. This is because the functions that will
//    accept vectors and matrices as arguments would also need to specify the
//    dimensions as template arguments, which creates multiple compiled
//    implementations of the same function. While different implementations may
//    be needed for different levels of parallelism, a block capable of
//    processing a 128x128 matrix should also be able to process a 1024x1024
//    matrix. for the same operation. Removing the template parameters from said
//    function allows for a single implementation to be used for all argument
//    sizes.

constexpr size_t log2(size_t n) { return ((n < 2) ? 0 : 1 + log2(n / 2)); }

/**
 * A wrapper for a stream of data representing a vector.
 *
 * @tparam T The type of the elements in the vector. Supports any type with
 * defined arithmetic ops.
 * @tparam Par Number of elements retrieved in one read operation. Must be a
 * power of 2.
 */
template <typename T, const unsigned int Par = MAX_BITWIDTH / 8 / sizeof(T)>
class Vector {
public:
  using StreamType = typename hls::stream<WideType<T, Par>>;

protected:
  StreamType stream_;
  T *buffer_;

#ifndef __SYNTHESIS__
  const unsigned int length_;
  unsigned int num_readers_;
  unsigned int num_writers_;
#endif

public:
  // Constructors
  /**
   * Creates a vector with a given length. Performs checks to validate the
   * length of parallelism
   *
   * @param length The length of the vector
   */
#ifndef __SYNTHESIS__
  Vector(const unsigned int length)
      : stream_(), buffer_(nullptr), length_(length), num_readers_(0),
        num_writers_(0) {
#else
  Vector(const unsigned int length) : stream_(), buffer_(nullptr) {
#endif
#pragma HLS INLINE
    static_assert(Par % 1 << log2(Par) == 0, "Par must be a power of 2");
#ifndef __SYNTHESIS__
    assert(("length must be greater than 0", length > 0));
    assert(("length must be a multiple of Par", length % Par == 0));
#endif
  }

  /**
   * Creates a vector and fills it with a single value
   *
   * @param p_Val The value to fill the vector with.
   * @param length The length of the vector
   */
#ifndef __SYNTHESIS__
  Vector(T p_Val, const unsigned int length)
      : stream_(), buffer_(nullptr), length_(length), num_readers_(0),
        num_writers_(1) {
#else
  Vector(T p_Val, const unsigned int length) : stream_(), buffer_(nullptr) {
#endif
#pragma HLS INLINE
    static_assert(Par % 1 << log2(Par) == 0, "Par must be a power of 2");
#ifndef __SYNTHESIS__
    assert(("length must be greater than 0", length > 0));
    assert(("length must be a multiple of Par", length % Par == 0));
#endif
    for (size_t i = 0; i < length; i += Par) {
#pragma HLS PIPELINE
      WideType<T, Par> value;
      for (size_t j = 0; j < Par; j++) {
#pragma HLS UNROLL
        value[j] = p_Val;
      }
      stream_.write(value);
    }
  }

  /**
   * Creates a vector associated with input buffer. Internal stream is disabled
   *
   * @param in_array The array of vector values
   * @param length The length of the vector
   */
#ifndef __SYNTHESIS__
  Vector(T *in_array, const unsigned int length)
      : stream_(), buffer_(in_array), length_(length), num_readers_(0),
        num_writers_(1){
#else
  Vector(T *in_array, const unsigned int length)
      : stream_(), buffer_(in_array) {
#endif
#pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable = in_array type = cyclic factor = Par
#pragma HLS STREAM variable = stream_ depth = 0
            static_assert(Par % 1 << log2(Par) == 0,
                          "Par must be a power of 2");
#ifndef __SYNTHESIS__
  assert(("length must be greater than 0", length > 0));
  assert(("length must be a multiple of Par", length % Par == 0));
#endif
}

~Vector() {
#ifndef __SYNTHESIS__
  assert(("Vector stream is not empty", stream_.empty()));
#endif
}

#ifndef __SYNTHESIS__
/**
 * Registers a writer for the vector. Does nothing if the vector has a buffer
 */
bool write_lock() {
#pragma HLS INLINE
  if (buffer_ == nullptr) {
    if (num_writers_ == 0) {
      num_writers_++;
      return true;
    } else {
      return false;
    }
  }
  return true;
}

/**
 * Registers a reader for the vector. Does nothing if the vector has a buffer
 */
bool read_lock() {
#pragma HLS INLINE
  if (buffer_ == nullptr) {
    if (num_readers_ == 0) {
      num_readers_++;
      return true;
    } else {
      return false;
    }
  }
  return true;
}
#endif

/**
 * Writes to the underlying stream
 *
 * @param value The value to write to the stream.
 */
void write(WideType<T, Par> value) {
#pragma HLS INLINE
#ifndef __SYNTHESIS__
  assert(("An input buffer has been provided for this vector. No additional "
          "input is accepted",
          buffer_ == nullptr));
#endif
  stream_.write(value);
}

// TODO: Explore providing a buffer to store the vector when repeat_vector is
// nonsingular
/**
 * Reads from the underlying stream
 *
 * @param Stream The stream to put read data into
 * @param repeat_elements The number of times to repeat each element into the
 * provided stream
 * @param repeat_vector The number of times to repeat the vector into the
 * provided stream
 *
 * @return A WideType containing the next Par elements from the stream.
 */
void read(StreamType &stream, int repeat_elements = 1, int repeat_vector = 1) {
// TODO: Explore getting rid of repeat elements. It kind fucks up performance
// unless we can guarantee it's a power of 2
#ifndef __SYNTHESIS__
  assert(("repeat_elements must be at least 1", repeat_elements > 0));
  assert(("repeat_vector must be at least 1", repeat_vector > 0));

  // TODO: For now only sequential reads are supported
  assert(("repeat_elements is not supported yet, must be left default",
          repeat_elements == 1));
#endif
  if (buffer_ == nullptr) {
    StreamType repeat_stream;
#pragma HLS STREAM variable = repeat_stream depth = length_ / Par
    for (int i = 0; i < repeat_vector; i++) {
      for (int j = 0; j < length_; j += Par) {
#pragma HLS PIPELINE
        WideType<T, Par> value;
        if (i == 0) {
          value = stream_.read();
        } else {
          value = repeat_stream.read();
        }
        if (i < repeat_vector - 1) {
          repeat_stream.write(value);
        }
        stream.write(value);
      }
    }
  } else {
    for (int i = 0; i < repeat_vector; i++) {
      for (int j = 0; j < length_; j += Par) {
#pragma HLS PIPELINE
        WideType<T, Par> value;
        for (int k = 0; k < Par; k++) {
#pragma HLS UNROLL
          value[k] = buffer_[j + k];
        }
        stream.write(value);
      }
    }
  }
}

/**
 * Writes the stream to memory
 *
 * @param value The pointer to memory to write the stream to.
 */
void to_memory(T *out_array) {
#pragma HLS ARRAY_PARTITION variable = out_array type = cyclic factor = Par
  for (size_t i = 0; i < length_; i += Par) {
#pragma HLS PIPELINE
    WideType<T, Par> value = stream_.read();
    for (size_t j = 0; j < Par; j++) {
#pragma HLS UNROLL
      out_array[i + j] = value[j];
    }
  }
}

/**
 * Checks if the underlying stream is empty
 *
 * @return True if the stream is empty, false otherwise.
 */
bool empty() {
#pragma HLS INLINE
  return stream_.empty();
}

/**
 * Returns the number of elements in the underlying stream
 *
 * @return The number of elements in the stream. Should be less than or equal
 * to Length
 */
unsigned int size() {
#pragma HLS INLINE
  return stream_.size();
}

/**
 * Returns the size of the vector, used for dimension checks during behavioral C
 * synthesis
 *
 * @return The size of the vector
 */
unsigned int length() {
#pragma HLS INLINE
  return length_;
}

/**
 * Alias for length(), to match the equivalent method in the Matrix class
 *
 * @return The size of the vector
 */
unsigned int shape() {
#pragma HLS INLINE
  return length_;
}

// TODO: Add support for reshaping and slicing. With the former returning a
// matrix potentially
}; // namespace blas

/**
 * A wrapper for a stream of data representing a general matrix.
 *
 * @tparam T The type of the elements in the matrix. Supports any type with
 * defined arithmetic ops.
 * @tparam Order The major order of the matrix. Can be either RowMajor or
 * ColMajor.
 * @tparam Par Number of elements retrieved in one read operation. Must be a
 * power of 2.
 */
template <typename T, MajorOrder Order = RowMajor,
          const unsigned int Par = MAX_BITWIDTH / 8 / sizeof(T)>
class Matrix {
public:
  using StreamType = typename hls::stream<WideType<T, Par>>;

protected:
  hls::stream<WideType<T, Par>> stream_;
  T *buffer_;

#ifndef __SYNTHESIS__
  const unsigned int rows_;
  const unsigned int cols_;
  unsigned int num_readers_;
  unsigned int num_writers_;
#endif

public:
  // Constructors
  /**
   * Creates a matrix with a given number of rows and columns. Performs checks
   * to validate the dimensions of the matrix and the length of parallelism
   *
   * @param Rows The number of rows in the matrix
   * @param Cols The number of columns in the matrix
   */
#ifndef __SYNTHESIS__
  Matrix(const unsigned int Rows, const unsigned int Cols)
      : stream_(), buffer_(nullptr), rows_(Rows), cols_(Cols), num_readers_(0),
        num_writers_(0) {
#else
  Matrix(const unsigned int Rows, const unsigned int Cols)
      : stream_(), buffer_(nullptr) {
#endif
#pragma HLS INLINE
    static_assert(Par % 1 << log2(Par) == 0, "Par must be a power of 2");
#ifndef __SYNTHESIS__
    assert(("Rows must be greater than 0", Rows > 0));
    assert(("Cols must be greater than 0", Cols > 0));
    if (Order == RowMajor) {
      assert(("Cols must be a multiple of Par", Cols % Par == 0));
    } else {
      assert(("Rows must be a multiple of Par", Rows % Par == 0));
    }
#endif
  }

  /**
   * Creates a matrix and fills it with a single value
   *
   * @param p_Val The value to fill the matrix with.
   * @param Rows The number of rows in the matrix
   * @param Cols The number of columns in the matrix
   */
#ifndef __SYNTHESIS__
  Matrix(T p_Val, const unsigned int Rows, const unsigned int Cols)
      : stream_(), buffer_(nullptr), rows_(Rows), cols_(Cols), num_readers_(0),
        num_writers_(1) {
#else
  Matrix(T p_Val, const unsigned int Rows, const unsigned int Cols)
      : stream_(), buffer_(nullptr) {
#endif
    static_assert(Par % 1 << log2(Par) == 0, "Par must be a power of 2");
#ifndef __SYNTHESIS__
    assert(("Rows must be greater than 0", Rows > 0));
    assert(("Cols must be greater than 0", Cols > 0));
    if (Order == RowMajor) {
      assert(("Cols must be a multiple of Par", Cols % Par == 0));
    } else {
      assert(("Rows must be a multiple of Par", Rows % Par == 0));
    }
#endif
    for (size_t i = 0; i < Rows; i++) {
      for (size_t j = 0; j < Cols; j += Par) {
#pragma HLS PIPELINE
        WideType<T, Par> value;
        for (size_t k = 0; k < Par; k++) {
#pragma HLS UNROLL
          value[k] = p_Val;
        }
        stream_.write(value);
      }
    }
  }
/**
 * Creates a matrix and fills it with provided buffer. Internal stream is
 * disabled
 *
 * @param buffer The array to fill the vector with. Memory is assumed to be
 * arranaged in order matching Order specified in the class template
 * @param Rows The number of rows in the matrix
 * @param Cols The number of columns in the matrix
 */
#ifndef __SYNTHESIS__
  Matrix(T *buffer, const unsigned int Rows, const unsigned int Cols)
      : stream_(), buffer_(buffer), rows_(Rows), cols_(Cols), num_readers_(0),
        num_writers_(1) {
#else
  Matrix(T *buffer, const unsigned int Rows, const unsigned int Cols)
      : stream_(), buffer_(buffer) {
#endif
#pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable = buffer type = cyclic factor = Par
#pragma HLS STREAM variable = stream_ depth = 0
    static_assert(Par % 1 << log2(Par) == 0, "Par must be a power of 2");
#ifndef __SYNTHESIS__
    assert(("Rows must be greater than 0", Rows > 0));
    assert(("Cols must be greater than 0", Cols > 0));
    if (Order == RowMajor) {
      assert(("Cols must be a multiple of Par", Cols % Par == 0));
    } else {
      assert(("Rows must be a multiple of Par", Rows % Par == 0));
    }
#endif
    buffer_ = buffer;
  }

  // TODO: Revisit these in the context of the new type design
  // Matrix can't be copied.
  Matrix(const Matrix &other) = delete;
  Matrix &operator=(const Matrix &other) = delete;

  // Matrix can be moved, but this consumes it
  Matrix(Matrix &&other) noexcept
      : stream_(std::move(other.stream_)), rows_(other.rows_),
        cols_(other.cols_) {
    // TODO: Set consume flag when that's implemented
  }
  Matrix &operator=(Matrix &&other) noexcept {
    if (this != &other) {
      stream_ = other.stream_;
      rows_ = other.rows_;
      cols_ = other.cols_;
    }
    return *this;
  }

  ~Matrix() {
#ifndef __SYNTHESIS__
    assert(("Matrix stream is not empty", stream_.empty()));
#endif
  }

#ifndef __SYNTHESIS__
  /**
   * Registers a writer for the matrix. Does nothing if the matrix has a buffer
   */
  bool write_lock() {
#pragma HLS INLINE
    if (buffer_ == nullptr) {
      if (num_writers_ == 0) {
        num_writers_++;
        return true;
      } else {
        return false;
      }
    }
    return true;
  }

  /**
   * Registers a reader for the matrix. Does nothing if the matrix has a buffer
   */
  bool read_lock() {
#pragma HLS INLINE
    if (buffer_ == nullptr) {
      if (num_readers_ == 0) {
        num_readers_++;
        return true;
      } else {
        return false;
      }
    }
    return true;
  }
#endif

  /**
   * Writes to the underlying stream
   *
   * @param value The value to write to the stream.
   */
  void write(WideType<T, Par> value) {
#pragma HLS INLINE
#ifndef __SYNTHESIS__
    assert(("An input buffer has been provided for this matrix. No additional "
            "input is accepted",
            buffer_ == nullptr));
#endif
    stream_.write(value);
  }

  // TODO: Explore providing a buffer to store the vector when repeat_row or
  // repeat_matrix are nonsingular
  /**
   * Reads from the underlying stream
   *
   * @param Stream The stream to put read data into
   * @param repeat_elements If true, repeats each element Par times. Only
   * supported with tiling
   * @param repeat_row Number of times to repeat each row in a tile
   * @param repeat_matrix Number of times to repeat the entire matrix
   */
  void read(StreamType &stream, const bool repeat_elements = false,
            const int repeat_row = 1, const int repeat_matrix = 1) {
#ifndef __SYNTHESIS__
    assert(("repeat_elements is not supported yet, must be left default",
            repeat_elements == false));
    assert(("repeat_row must be at least 1", repeat_row > 0));
    assert(("repeat_matrix must be at least 1", repeat_matrix > 0));
#endif
    if (buffer_ == nullptr) {
      // TODO: Implement reordering from sequential stream. For now it ignores
      // the parameters and does a sequential read

#ifndef __SYNTHESIS__
      assert(("Pure stream matrices only support sequential reads for now. "
              "repeat_elements must remain "
              "default value",
              repeat_elements == false));
      assert(("Pure stream matrices only support sequential reads for now. "
              "repeat_row must remain "
              "default value",
              repeat_row == 1));
      assert(("Pure stream matrices only support sequential reads for now. "
              "repeat_matrix must remain "
              "default value",
              repeat_matrix == 1));
#endif
      for (int i = 0; i < rows_; i++) {
        for (int j = 0; j < cols_; j += Par) {
          stream.write(stream_.read());
        }
      }
    } else {
      if (Order == RowMajor) {
        for (int i = 0; i < repeat_matrix; i++) {
          for (int j = 0; j < rows_; j++) {
            for (int k = 0; k < repeat_row; k++) {
              for (int l = 0; l < cols_; l += Par) {
#pragma HLS PIPELINE
                WideType<T, Par> value;
                for (int m = 0; m < Par; m++) {
#pragma HLS UNROLL
                  value[m] = buffer_[j * cols_ + l + m];
                }
                stream.write(value);
              }
            }
          }
        }
      } else { // Col Major Order
        for (int i = 0; i < repeat_matrix; i++) {
          for (int j = 0; j < cols_; j++) {
            for (int k = 0; k < repeat_row; k++) {
              for (int l = 0; l < rows_; l += Par) {
#pragma HLS PIPELINE
                WideType<T, Par> value;
                for (int m = 0; m < Par; m++) {
#pragma HLS UNROLL
                  if (repeat_elements) {
                    value[m] = buffer_[j * rows_ + l + m];
                  } else {
                    value[m] = buffer_[j * rows_ + l + m];
                  }
                }
                stream.write(value);
              }
            }
          }
        }
      }

#ifndef __SYNTHESIS__

      assert(
          ("Output stream is unexpected length",
           stream.size() == rows_ * cols_ * repeat_matrix * repeat_row / Par));
#endif
    }
  }

  /**
   * Writes the stream to memory.
   *
   * @param value The pointer to memory to write the stream to.
   */
  void to_memory(T *out_array) {
#pragma HLS ARRAY_PARTITION variable = out_array type = cyclic factor = Par
    if (Order == RowMajor) {
      for (size_t i = 0; i < rows_; i++) {
        for (size_t j = 0; j < cols_; j += Par) {
#pragma HLS PIPELINE
#pragma HLS LOOP_FLATTEN
          WideType<T, Par> value = stream_.read();
          for (size_t k = 0; k < Par; k++) {
#pragma HLS UNROLL
            // if (OutputOrder == RowMajor) {
            out_array[i * cols_ + j + k] = value[k];
            // } else {
            //   out_array[(j + k) * rows_ + i] = value[k];
            // }
          }
        }
      }
    } else {
      for (size_t i = 0; i < cols_; i++) {
        for (size_t j = 0; j < rows_; j += Par) {
#pragma HLS PIPELINE
#pragma HLS LOOP_FLATTEN
          WideType<T, Par> value = stream_.read();
          for (size_t k = 0; k < Par; k++) {
#pragma HLS UNROLL
            // if (OutputOrder == RowMajor) {
            //   out_array[(j + k) * cols_ + i] = value[k];
            // } else {
            out_array[i * rows_ + j + k] = value[k];
            // }
          }
        }
      }
    }
  }

  /** Inverts this matrix and writes it to provided matrix
   *
   * r = A^-1
   *
   * @param result The matrix to write the inverted matrix to
   * @param buffer A buffer to use for intermediate calculations. Must be of
   * size 2*N*N
   */
  bool invert(Matrix<T, Order, Par> &result, T *buffer) {
#ifndef __SYNTHESIS__
    assert(("Matrix is not square", rows_ == cols_));
    assert(("Provided matrix does not match dimensions of this matrix",
            result.rows() == rows_ && result.cols() == cols_));
    assert(("Must provide buffer of size 2*M*N", buffer != nullptr));
#endif
    int i = 0;
    T alpha;
    if (buffer_ ==
        nullptr) { // Initial iteration of outer loop if Matrix is pure-stream
      for (int j = 0; j < rows_; j++) {
#pragma HLS PIPELINE
        for (int k = 0; k < rows_; k += Par) {
          WideType<T, Par> value = stream_.read();
          for (int l = 0; l < Par; l++) {
#pragma HLS UNROLL
            if (k == 0) {
              alpha = value[0];
              if (alpha == 0 && j == 0) {
                return false;
              }
            }
            if (j == 0) {
              buffer[j * 2 * rows_ + k + l] = value[l] / alpha;
              if (j == k + l) {
                buffer[j * 2 * rows_ + k + l + rows_] = 1 / alpha;
              } else {
                buffer[j * 2 * rows_ + k + l + rows_] = 0;
              }
            } else {
              buffer[j * 2 * rows_ + k + l] =
                  value[l] - alpha * buffer[i * 2 * rows_ + k + l];
              if (j == k + l) {
                buffer[j * 2 * rows_ + k + l + rows_] =
                    1 - alpha * buffer[i * 2 * rows_ + k + l];
              } else {
                buffer[j * 2 * rows_ + k + l + rows_] =
                    -alpha * buffer[i * 2 * rows_ + k + l];
              }
            }
          }
        }
        for (int k = rows_; k < 2 * rows_; k++) {
          if (j == 0) {
            if (j == k) {
              buffer[j * 2 * rows_ + k + rows_] = 1 / alpha;
            } else {
              buffer[j * 2 * rows_ + k + rows_] = 0;
            }
          } else {
            if (j == k) {
              buffer[j * 2 * rows_ + k + rows_] =
                  1 - alpha * buffer[i * 2 * rows_ + k];
            } else {
              buffer[j * 2 * rows_ + k + rows_] =
                  -alpha * buffer[i * 2 * rows_ + k];
            }
          }
        }
      }
    } else {    // If matrix was created as a buffer
      for (int j = 0; j < rows_; j++) {
#pragma HLS PIPELINE
        alpha = buffer_[j * rows_];
        if (alpha == 0 && j == 0) {
          return false;
        }
        for (int k = 0; k < rows_; k++) {
          if (i == j) {
            buffer[j * 2 * rows_ + k] = buffer_[j * rows_ + k] / alpha;
          } else {
            buffer[j * 2 * rows_ + k] =
                buffer_[j * rows_ + k] - alpha * buffer[i * 2 * rows_ + k];
          }
        }
        for (int k = rows_; k < 2 * rows_; k++) {
          if (i == j) {
            if (j == (k - rows_)) {
              buffer[j * 2 * rows_ + k] = 1 / alpha;
            } else {
              buffer[j * 2 * rows_ + k] = 0;
            }
          } else {
            if (j == (k - rows_)) {
              buffer[j * 2 * rows_ + k] =
                  1 - alpha * buffer[i * 2 * rows_ + k];
            } else {
              buffer[j * 2 * rows_ + k] =
                  -alpha * buffer[i * 2 * rows_ + k];
            }
          }
        }
      }
    }

    for (int i = 1; i < rows_; i++) {
      for (int j = 0; j < rows_; j++) {
#pragma HLS PIPELINE
        T alpha;
        if (j < i) {
          alpha = buffer[j * 2 * rows_ + i] / buffer[i * 2 * rows_ + i];
        } else {
          alpha = buffer[j * 2 * rows_ + i];
        }
        if (alpha == 0 && i == j) {
          return false;
        }
        for (int k = i; k < 2 * rows_; k++) {
          if (i == j) {
            buffer[j * 2 * rows_ + k] /= alpha;
          } else {
            buffer[j * 2 * rows_ + k] -= alpha * buffer[i * 2 * rows_ + k];
          }
        }
      }
    }

    for (int i = 0; i < rows_; i++) {
      for (int j = rows_; j < 2*rows_; j += Par) {
#pragma HLS PIPELINE
        WideType<T, Par> value;
        for (int k = 0; k < Par; k++) {
#pragma HLS UNROLL
          value[k] = buffer[i * 2 * rows_ + j + k];
        }
        result.write(value);
      }
    }

#ifndef __SYNTHESIS__
    assert(("Internal stream isn't empty after matrix inversion",
            stream_.empty()));
#endif
    return true;
  }

  /**
   * Checks if the underlying stream is empty
   *
   * @return True if the stream is empty, false otherwise.
   */
  bool empty() {
#pragma HLS INLINE
    return stream_.empty();
  }

  /**
   * Returns the number of elements in the underlying stream
   *
   * @return The number of elements in the stream. Should be less than or equal
   * to Rows * Cols
   */
  unsigned int size() {
#pragma HLS INLINE
    return stream_.size();
  }

  /**
   * Returns the number of rows in the matrix, used for dimension checks during
   * behavioral C
   *
   * @return The number of rows in the matrix
   */
  unsigned int rows() {
#pragma HLS INLINE
    return rows_;
  }

  /**
   * Returns the number of columns in the matrix, used for dimension checks
   * during behavioral C
   *
   * @return The number of columns in the matrix
   */
  unsigned int cols() {
#pragma HLS INLINE
    return cols_;
  }

  /**
   * Returns the shape of the matrix, used for dimension checks during
   * behavioral C synthesis
   *
   * @return The shape of the matrix
   */
  std::pair<unsigned int, unsigned int> shape() {
#pragma HLS INLINE
    return {rows_, cols_};
  }

  // TODO: Add support for reshaping and slicing. With the former returning a
  // vector potentially
};

/**
 * A wrapper for a stream of data representing a general matrix stored in tiled
 * order.asum
 *
 * A 4x4 matrix in row-major order with a Par level of 4 is read in the
 * following order
 *
 *  00 01 04 05
 *  02 03 06 07
 *  08 09 12 13
 *  10 11 14 15
 *
 * Each stream read returns an entire tile, which is square and has Par elements
 */
template <typename T, MajorOrder Order = RowMajor,
          const unsigned int Par = MAX_BITWIDTH / 8 / sizeof(T)>
class TiledMatrix : public Matrix<T, Order, Par> {};
// TODO: Par must be a square number. Modify the default parameter to ensure
// this is the case

/**
 * A wrapper for a stream of data representing a banded matrix.
 *
 * @tparam T The type of the elements in the matrix. Supports any type with
 * defined arithmetic ops.
 * @tparam SubDiagonals The number of subdiagonals in the matrix.
 * @tparam SupDiagonals The number of superdiagonals in the matrix.
 * @tparam Par Number of elements retrieved in one read operation. Must be a
 * power of 2.
 */
template <typename T, const unsigned int SubDiagonals,
          const unsigned int SupDiagonals,
          const unsigned int Par = MAX_BITWIDTH / 8 / sizeof(T)>
class BandedMatrix : public Matrix<T, RowMajor, Par> {};

/**
 * A wrapper for a stream of data representing a diagonal matrix. This is
 * functionally equivalent to a banded matrix with 0 subdiagonals and 0
 * superdiagonals.
 *
 * @tparam T The type of the elements in the matrix. Supports any type with
 * defined arithmetic ops.
 * @tparam Par Number of elements retrieved in one read operation. Must be a
 * power of 2.
 */
template <typename T, unsigned const int Par = MAX_BITWIDTH / 8 / sizeof(T)>
using DiagonalMatrix = BandedMatrix<T, 0, 0, Par>;

/**
 * A wrapper for a stream of data representing a triangular matrix.
 * NOTE: Store packed by default?
 *
 * @tparam T The type of the elements in the matrix. Supports any type with
 * defined arithmetic ops.
 * @tparam Order The major order of the matrix. Can be either RowMajor or
 * ColMajor.
 * @tparam UpLo Whether the matrix is upper or lower triangular.
 * @tparam Par Number of elements retrieved in one read operation. Must be a
 * power of 2.
 */
template <typename T, const MajorOrder Order = RowMajor,
          const UpperLower UpLo = Upper,
          const unsigned int Par = MAX_BITWIDTH / 8 / sizeof(T)>
class TriangularMatrix : public Matrix<T, Order, Par> {};

/**
 * A wrapper for a stream of data representing a triangular banded matrix.
 * NOTE: Upper triangular is always row major?
 *
 * @tparam T The type of the elements in the matrix. Supports any type with
 * defined arithmetic ops.
 * @tparam Diagonals The number of subdiagonals in the matrix.
 * @tparam UpLo Whether the matrix is upper or lower triangular.
 * @tparam Par Number of elements retrieved in one read operation. Must be a
 * power of 2.
 */
template <typename T, const unsigned int Diagonals,
          const unsigned int Par = MAX_BITWIDTH / 8 / sizeof(T),
          const UpperLower UpLo = Upper>
class TriangularBandedMatrix
    : public BandedMatrix<T, (UpLo == Upper) ? 0 : Diagonals,
                          (UpLo == Lower) ? 0 : Diagonals, Par> {};

// TODO: Add support for unit triangular matrices and their banded equivalents

/**
 * A wrapper for a stream of data representing a symmetric matrix.
 * NOTE: Store packed by default?
 *
 * @tparam T The type of the elements in the matrix. Supports any type with
 * defined arithmetic ops.
 * @tparam Order The major order of the matrix. Can be either RowMajor or
 * ColMajor.
 * @tparam UpLo Whether the matrix is upper or lower triangular.
 * @tparam Par Number of elements retrieved in one read operation. Must be a
 * power of 2.
 */
template <typename T, const MajorOrder Order = RowMajor,
          const UpperLower UpLo = Upper,
          const unsigned int Par = MAX_BITWIDTH / 8 / sizeof(T)>
class SymmetricMatrix : public Matrix<T, Order, Par> {};

/**
 * A wrapper for a stream of data representing a symmetric banded matrix.
 *
 * @tparam T The type of the elements in the matrix. Supports any type with
 * defined arithmetic ops.
 * @tparam Diagonals The number of superdiagonals/subdiagonals in the matrix
 * (they are equal).
 * @tparam UpLo Whether the matrix is upper or lower triangular.
 * @tparam Par Number of elements retrieved in one read operation. Must be a
 * power of 2.
 */
template <typename T, const unsigned int Diagonals,
          const UpperLower UpLo = Upper,
          const unsigned int Par = MAX_BITWIDTH / 8 / sizeof(T)>
class SymmetricBandedMatrix
    : public BandedMatrix<T, Diagonals, Diagonals, Par> {};

/**
 * A wrapper for a stream of data representing a Hermitian matrix.
 * NOTE: Store packed by default?
 * NOTE: Throw warnings when T is not complex (because that's just a symmetric
 * matrix)
 *
 * @tparam T The type of the elements in the matrix. Supports any type with
 * defined arithmetic ops.
 * @tparam Order The major order of the matrix. Can be either RowMajor or
 * ColMajor.
 * @tparam Par Number of elements retrieved in one read operation. Must be a
 * power of 2.
 */
template <typename T, const MajorOrder Order = RowMajor,
          const UpperLower UpLo = Upper,
          const unsigned int Par = MAX_BITWIDTH / 8 / sizeof(T)>
class HermitianMatrix : public Matrix<T, Order, Par> {};

/**
 * A wrapper for a stream of data representing a Hermitian banded matrix.
 * NOTE: Throw warnings when T is not complex (because that's just a banded
 * symmetric matrix)
 *
 * @tparam T The type of the elements in the matrix. Supports any type with
 * defined arithmetic ops.
 * @tparam Diagonals The number of superdiagonals/subdiagonals in the matrix
 * (they are equal).
 * @tparam UpLo Whether the matrix is upper or lower triangular.
 * @tparam Par Number of elements retrieved in one read operation. Must be a
 * power of 2.
 */
template <typename T, const unsigned int Diagonals,
          const UpperLower UpLo = Upper,
          const unsigned int Par = MAX_BITWIDTH / 8 / sizeof(T)>
class HermitianBandedMatrix
    : public BandedMatrix<T, Diagonals, Diagonals, Par> {};

/**
 * Transposes a column-major matrix into a row-major matrix.
 * Doesn't move any values, just changes the type. This is analogous to the
 * TRANSPOSE flag used in the BLAS standard.
 *
 * @tparam T The type of the elements in the matrix. Supports any type with
 * defined arithmetic ops.
 * @tparam Par Number of elements retrieved in one read operation. Must be a
 * power of 2.
 *
 * @param A The column-major matrix to read from
 * @param AT The row-major matrix to write to
 */
template <typename T, const unsigned int Par = MAX_BITWIDTH / 8 / sizeof(T)>
void transpose(Matrix<T, ColMajor, Par> &A, Matrix<T, RowMajor, Par> &AT) {
#ifndef __SYNTHESIS__
  assert(("Dimensions of A and AT must match",
          A.rows() == AT.cols() && A.cols() == AT.rows()));
#endif
#pragma HLS DATAFLOW
  typename Matrix<T, ColMajor, Par>::StreamType stream;
  A.read(stream);
  for (size_t i = 0; i < A.rows(); i++) {
    for (size_t j = 0; j < A.cols(); j += Par) {
      AT.write(stream.read());
    }
  }
}

/**
 * Transposes a row-major matrix into a column-major matrix.
 * Doesn't move any values, just changes the type. This is analogous to the
 * TRANSPOSE flag used in
 *
 * @tparam T The type of the elements in the matrix. Supports any type with
 * defined arithmetic ops.
 * @tparam Par Number of elements retrieved in one read operation. Must be a
 * power of 2.
 *
 * @param A The row-major matrix to read from
 * @param AT The column-major matrix to write to
 */
template <typename T, const unsigned int Par = MAX_BITWIDTH / 8 / sizeof(T)>
void transpose(Matrix<T, RowMajor, Par> &A, Matrix<T, ColMajor, Par> &AT) {
#ifndef __SYNTHESIS__
  assert(("Dimensions of A and AT must match",
          A.rows() == AT.cols() && A.cols() == AT.rows()));
#endif
#pragma HLS DATAFLOW
  typename Matrix<T, RowMajor, Par>::StreamType stream;
  A.read(stream);
  for (size_t i = 0; i < A.cols(); i++) {
    for (size_t j = 0; j < A.rows(); j += Par) {
      AT.write(stream.read());
    }
  }
}

} // namespace dyfc
} // namespace dyfc

#endif // DYFC_BLAS_TYPES_HPP
