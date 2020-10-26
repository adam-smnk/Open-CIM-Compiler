#################################################
# Transform TC to MLIR
#################################################
set(TC_SOURCES
  tc/gather.tc
)

tc_to_mlir(
  MLIR_FILES
  "${TC_SOURCES}"
)

#################################################
# Compile benchmark application
#################################################
set(SOURCES
  gather.cpp
)

make_all_benchmarks(
  gather
  ${SOURCES}
  ${MLIR_FILES}
)
