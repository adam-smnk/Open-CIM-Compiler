#################################################
# Transform TC to MLIR
#################################################
set(TC_SOURCES
  tc/tc_3x3x2.tc
)

tc_to_mlir(
  MLIR_FILES
  "${TC_SOURCES}"
)

#################################################
# Compile benchmark application
#################################################
set(SOURCES
  tc_3x3x2.cpp
)

make_all_benchmarks(
  tc_3x3x2
  ${SOURCES}
  ${MLIR_FILES}
)
