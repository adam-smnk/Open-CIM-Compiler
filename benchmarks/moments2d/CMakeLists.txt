#################################################
# Transform TC to MLIR
#################################################
set(TC_SOURCES
  tc/moments2d.tc
)

tc_to_mlir(
  MLIR_FILES
  "${TC_SOURCES}"
)

#################################################
# Compile benchmark application
#################################################
set(SOURCES
  moments2d.cpp
)

make_all_benchmarks(
  moments2d
  ${SOURCES}
  ${MLIR_FILES}
)
