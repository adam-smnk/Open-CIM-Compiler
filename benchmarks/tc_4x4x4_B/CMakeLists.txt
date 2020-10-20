#################################################
# Transform TC to MLIR
#################################################
set(TC_SOURCES
  tc/tc_4x4x4_B.tc
)

tc_to_mlir(
  MLIR_FILES
  "${TC_SOURCES}"
)

#################################################
# Compile benchmark application
#################################################
set(SOURCES
  tc_4x4x4_B.cpp
)

make_all_benchmarks(
  tc_4x4x4_B
  ${SOURCES}
  ${MLIR_FILES}
)
