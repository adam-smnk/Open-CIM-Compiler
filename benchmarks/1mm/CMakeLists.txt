#################################################
# Transform TC to MLIR
#################################################
set(TC_SOURCES
  tc/1mm.tc
)

tc_to_mlir(
  MLIR_FILES
  "${TC_SOURCES}"
)

#################################################
# Compile benchmark application
#################################################
set(SOURCES
  1mm.cpp
)

make_all_benchmarks(
  1mm
  ${SOURCES}
  ${MLIR_FILES}
)
