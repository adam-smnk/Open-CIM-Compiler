#################################################
# Transform TC to MLIR
#################################################
set(TC_SOURCES
  tc/tbmm.tc
)

tc_to_mlir(
  MLIR_FILES
  "${TC_SOURCES}"
)

#################################################
# Compile benchmark application
#################################################
set(SOURCES
  tbmm.cpp
)

make_all_benchmarks(
  tbmm
  ${SOURCES}
  ${MLIR_FILES}
)
