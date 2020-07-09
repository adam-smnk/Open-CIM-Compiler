cmake_minimum_required(VERSION 3.4.3)

#################################################
# Compilation Settings
#################################################
set(CMAKE_C_COMPILER "clang")
set(CMAKE_CXX_COMPILER "clang++")

set(CMAKE_C_FLAGS  "${CMAKE_C_FLAGS} -O2 -fexceptions -fno-rtti")
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -O2 -fexceptions -fno-rtti")

set(CMAKE_BUILD_TYPE "")

project("GemmInMemory")

include_directories(${CMAKE_SOURCE_DIR}/include)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

#################################################
# LLVM Project Settings
#################################################

set(LLVM_ENABLE_PROJECTS "" CACHE STRING
  "Semicolon-separated list of projects to build,
  or \"all\". The mlir project is added by default.")
list(APPEND LLVM_ENABLE_PROJECTS "mlir")
set(LLVM_TARGETS_TO_BUILD "X86;NVPTX")
set(LLVM_ENABLE_ASSERTIONS "ON")
set(LLVM_ENABLE_LLD "ON")
set(LLVM_CCACHE_BUILD "ON")

# MLIR links some whole-archive libraries by name and they
# must be on the path. This may not be necessary in future versions of MLIR.
# link_directories(${CMAKE_BINARY_DIR}/llvm-project/llvm/lib)

#################################################
# MLIR Settings
#################################################
set(MLIR_OPT_BIN "$ENV{HOME}/llvm/llvm-project/build/bin/mlir-opt")
set(MLIR_TRANSLATE_BIN "$ENV{HOME}/llvm/llvm-project/build/bin/mlir-translate")
set(LLC_BIN "llc")

#################################################
# Teckyl Settings
#################################################
set(TECKYL_BIN "$ENV{HOME}/llvm/teckyl/build/bin/teckyl")

#################################################
# Cmake includes
#################################################
include(${CMAKE_SOURCE_DIR}/utils/cmake/file_generation.cmake)

#################################################
# Subdirectories
#################################################
add_subdirectory(lib)
add_subdirectory(tests)