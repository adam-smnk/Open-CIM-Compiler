#################################################
# Transform TC to MLIR
#################################################
set(TC_SOURCES
  tc/2mm.tc
)

tc_to_mlir(
  MLIR_FILES
  "${TC_SOURCES}"
)

#################################################
# Compile benchmark application
#################################################
set(SOURCES
  2mm.cpp
)

make_all_benchmarks(
  2mm
  ${SOURCES}
  ${MLIR_FILES}
)
