cmake_minimum_required(VERSION 3.4.3)

#################################################
# Compilation Settings
#################################################
set(CMAKE_C_COMPILER "clang")
set(CMAKE_CXX_COMPILER "clang++")

set(CMAKE_C_FLAGS  "${CMAKE_C_FLAGS} -O2 -fexceptions -fno-rtti -fPIC")
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -O2 -fexceptions -fno-rtti -fPIC")
SET(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS}")

set(CMAKE_BUILD_TYPE "")

project("GemmInMemory")

include_directories(${CMAKE_SOURCE_DIR}/include)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

#################################################
# Teckyl Settings
#################################################
set(TECKYL_BIN "$ENV{HOME}/llvm/teckyl/build/bin/teckyl")

separate_arguments(TECKYL_FLAGS UNIX_COMMAND "--emit=mlir")

#################################################
# MLIR Settings
#################################################
set(MLIR_OPT_BIN "$ENV{HOME}/llvm/llvm-project/build/bin/mlir-opt")
set(MLIR_TRANSLATE_BIN "$ENV{HOME}/llvm/llvm-project/build/bin/mlir-translate")
set(LLC_BIN "llc")

# Intermediate file extensions
set(MLIR_GEN_EXTENSION ".mlir")
set(MLIR_CIM_EXTENSION "_cim.mlir")
set(OBJ_GEN_EXTENSION ".o")

# Linalg to CIM lowering flags
separate_arguments(MLIR_OPT_CIM_FLAGS UNIX_COMMAND "--convert-linalg-to-cim --convert-cim-to-std --cse")

# MLIR to .obj compilation flags
separate_arguments(MLIR_OPT_FLAGS UNIX_COMMAND "--convert-linalg-to-cim --convert-cim-to-std --convert-linalg-to-loops --convert-loop-to-std --convert-std-to-llvm=\'emit-c-wrappers=1\' --cse")
separate_arguments(MLIR_TRANSLATE_FLAGS UNIX_COMMAND "--mlir-to-llvmir")
separate_arguments(LLC_FLAGS UNIX_COMMAND "--filetype=obj --relocation-model=pic")

#################################################
# Cmake includes
#################################################
include(${CMAKE_SOURCE_DIR}/utils/cmake/file_generation.cmake)

#################################################
# Subdirectories
#################################################
add_subdirectory(lib)
add_subdirectory(tests)