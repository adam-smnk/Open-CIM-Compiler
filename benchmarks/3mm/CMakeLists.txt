#################################################
# Transform TC to MLIR
#################################################
set(TC_SOURCES
  tc/3mm.tc
)

tc_to_mlir(
  MLIR_FILES
  "${TC_SOURCES}"
)

#################################################
# Compile benchmark application
#################################################
set(SOURCES
  3mm.cpp
)

make_all_benchmarks(
  3mm
  ${SOURCES}
  ${MLIR_FILES}
)
