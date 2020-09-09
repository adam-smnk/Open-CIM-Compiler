set(SIMULATOR_INTERFACE_SOURCES
  cim_sim.c
)

add_library(simulatorInterface STATIC
  ${SIMULATOR_INTERFACE_SOURCES}
  ${SIM_OBJ}
)
