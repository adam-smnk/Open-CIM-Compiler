#################################################
# Transform TC to MLIR
#################################################
set(TC_SOURCES
  tc/tmm.tc
)

tc_to_mlir(
  MLIR_FILES
  "${TC_SOURCES}"
)

#################################################
# Compile benchmark application
#################################################
set(SOURCES
  tmm.cpp
)

make_all_benchmarks(
  tmm
  ${SOURCES}
  ${MLIR_FILES}
)
