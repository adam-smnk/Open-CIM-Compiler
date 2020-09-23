#################################################
# Transform TC to MLIR
#################################################
set(TC_SOURCES
  tc/kronecker3.tc
)

tc_to_mlir(
  MLIR_FILES
  "${TC_SOURCES}"
)

#################################################
# Compile benchmark application
#################################################
set(SOURCES
  kronecker3.cpp
)

make_all_benchmarks(
  kronecker3
  ${SOURCES}
  ${MLIR_FILES}
)
