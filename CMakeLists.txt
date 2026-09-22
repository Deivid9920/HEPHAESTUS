cmake_minimum_required(VERSION 3.16)
project(hephaestus CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
enable_testing()

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# POSIX-only build: getrusage/sys/resource.h and GCC/Clang flags.
# Run on Linux or WSL2 (see README limits).
add_compile_options(-O3 -march=native -Wall -Wextra)

add_executable(heph
  src/main.cpp
  src/core/tensor.cpp
  src/core/sha256.cpp
  src/core/json.cpp
  src/loader/loader.cpp
  src/kernels/kernels.cpp
  src/model/transformer.cpp
  src/kv/kv_cache.cpp
  src/quant/quantize.cpp
  src/tokenizer/bpe.cpp
  src/bench/bench.cpp
)
target_include_directories(heph PRIVATE src)

add_executable(test_kernels
  tests/cpp/test_kernels.cpp
  src/core/tensor.cpp
  src/core/sha256.cpp
  src/kernels/kernels.cpp
  src/kv/kv_cache.cpp
  src/quant/quantize.cpp
)
target_include_directories(test_kernels PRIVATE src)
add_test(NAME kernel_parity COMMAND test_kernels)

add_executable(test_kv
  tests/cpp/test_kv_int8.cpp
  src/core/tensor.cpp
  src/core/sha256.cpp
  src/core/json.cpp
  src/loader/loader.cpp
  src/kernels/kernels.cpp
  src/model/transformer.cpp
  src/kv/kv_cache.cpp
  src/quant/quantize.cpp
)
target_include_directories(test_kv PRIVATE src)
add_test(NAME kv_int8_paged COMMAND test_kv
         WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
