include(${CMAKE_SOURCE_DIR}/utils/cmake/benchmark_gen.cmake)

if(BUILD_BENCHMARKS)
  add_subdirectory(1mm)
  add_subdirectory(2mm)
  add_subdirectory(mlp3)
endif()
