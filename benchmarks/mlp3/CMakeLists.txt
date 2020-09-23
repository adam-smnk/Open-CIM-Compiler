#################################################
# Transform TC to MLIR
#################################################
set(TC_SOURCES
  tc/mlp3.tc
)

tc_to_mlir(
  MLIR_FILES
  "${TC_SOURCES}"
)

#################################################
# Compile benchmark application
#################################################
set(SOURCES
  mlp3.cpp
)

make_all_benchmarks(
  mlp3
  ${SOURCES}
  ${MLIR_FILES}
)
