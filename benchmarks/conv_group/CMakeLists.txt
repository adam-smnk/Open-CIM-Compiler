#################################################
# Transform TC to MLIR
#################################################
set(TC_SOURCES
  tc/conv_group.tc
)

tc_to_mlir(
  MLIR_FILES
  "${TC_SOURCES}"
)

#################################################
# Compile benchmark application
#################################################
set(SOURCES
  conv_group.cpp
)

# Overwrite default flags to compile convolution correctly
set(MLIR_CONV_FLAGS "--convert-linalg-to-loops --convert-loop-to-std --convert-linalg-to-llvm --convert-std-to-llvm=\'emit-c-wrappers=1\'")

make_all_benchmarks(
  conv_group
  ${SOURCES}
  ${MLIR_FILES}
)
