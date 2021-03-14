# OCC: The Open CIM Compiler

This repository contains MLIR-based compilation infrastructure and a suite of tests for a Computing-In-Memory device simulator.
This branch contains extra set of benchmarks that can be run on an ARM-based CIM simulator.

## End-to-end CIM compilation tests

The provided tests are based on the following steps:

1. **MLIR** code is generated from the functions defined in a **Tensor Comprehensions (TC)** file
2. **MLIR** is lowered through **CIM dialect** (and others) and compiled to an **object file**
3. A **test application** is compiled together with the MLIR **object file** and linked against **CIM runtime library**
4. The **test application** calls the compiled **TC functions** through their **MLIR C interfaces**

## Initializing submodules
```
git submodule init && git submodule update
```

## Building tests
Ensure that paths to the following executables are set correctly at the project root level ``CMakeLists.txt``:
* ``MLIR_OPT_BIN``
* ``MLIR_TRANSLATE_BIN``
* ``LLC_BIN``
* ``TECKYL_BIN``

Tests can then be built by (optionally) creating a build directory, invoking ``cmake`` and running ``make`` as follows:
* ``mkdir build``
* ``cd build``
* ``cmake ..``
* ``make``

The built tests application can be found in ``build/bin`` directory.<br/>
Other build artifacts, e.g. generated MLIR files, can be found in ``build/tests/TEST_NAME`` subdirectories.

## Build tools
Build tested with:
* ``cmake`` version 3.10.2
* ``clang`` version 10.0.0
* ``clang++`` version 10.0.0
* ``llc`` version 10.0.0
* ``ld`` version 2.30
* ``mlir_opt`` built from llvm-project submodule
* ``mlir_translate`` built from llvm-project submodule
* ``teckyl`` built from teckyl submodule
