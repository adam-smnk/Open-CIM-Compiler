set(SOURCES
  src/hello_world.cpp
)

# Hello world executable to test compiler
add_executable(hello-world
  ${SOURCES}
)

set_target_properties(hello-world
  PROPERTIES
  ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
