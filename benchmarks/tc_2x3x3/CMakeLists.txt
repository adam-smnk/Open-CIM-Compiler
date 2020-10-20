#################################################
# Transform TC to MLIR
#################################################
set(TC_SOURCES
  tc/tc_2x3x3.tc
)

tc_to_mlir(
  MLIR_FILES
  "${TC_SOURCES}"
)

#################################################
# Compile benchmark application
#################################################
set(SOURCES
  tc_2x3x3.cpp
)

make_all_benchmarks(
  tc_2x3x3
  ${SOURCES}
  ${MLIR_FILES}
)
