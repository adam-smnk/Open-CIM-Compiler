#################################################
# Transform TC to MLIR
#################################################
set(TC_SOURCES
  tc/lstm.tc
)

tc_to_mlir(
  MLIR_FILES
  "${TC_SOURCES}"
)

#################################################
# Compile benchmark application
#################################################
set(SOURCES
  lstm.cpp
)

make_all_benchmarks(
  lstm
  ${SOURCES}
  ${MLIR_FILES}
)
