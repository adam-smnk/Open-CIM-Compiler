#################################################
# Transform TC to MLIR
#################################################
set(TC_SOURCES
  tc/mv.tc
)

tc_to_mlir(
  MLIR_FILES
  "${TC_SOURCES}"
)

#################################################
# Compile benchmark application
#################################################
set(SOURCES
  mv.cpp
)

make_all_benchmarks(
  mv
  ${SOURCES}
  ${MLIR_FILES}
)
