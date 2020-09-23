include(${CMAKE_SOURCE_DIR}/utils/cmake/benchmark_gen.cmake)

if(BUILD_BENCHMARKS)
  add_subdirectory(1mm)
endif()
