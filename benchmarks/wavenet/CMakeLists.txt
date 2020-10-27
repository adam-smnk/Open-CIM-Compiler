#################################################
# Transform TC to MLIR
#################################################
set(TC_SOURCES
  tc/wavenet.tc
)

tc_to_mlir(
  MLIR_FILES
  "${TC_SOURCES}"
)

#################################################
# Compile benchmark application
#################################################
set(SOURCES
  wavenet.cpp
)

make_all_benchmarks(
  wavenet
  ${SOURCES}
  ${MLIR_FILES}
)
