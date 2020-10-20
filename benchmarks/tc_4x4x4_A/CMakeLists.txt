#################################################
# Transform TC to MLIR
#################################################
set(TC_SOURCES
  tc/tc_4x4x4_A.tc
)

tc_to_mlir(
  MLIR_FILES
  "${TC_SOURCES}"
)

#################################################
# Compile benchmark application
#################################################
set(SOURCES
  tc_4x4x4_A.cpp
)

make_all_benchmarks(
  tc_4x4x4_A
  ${SOURCES}
  ${MLIR_FILES}
)
