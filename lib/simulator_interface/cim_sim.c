#include "simulator_interface/cim_sim.h"

#include "libs/cim.h"
#include "libs/gic.h"
#include "libs/m5ops.h"

void simulator_init(void) {
  enable_caches();

#ifdef ENABLE_INTERRUPTS
  gic_init();
  gic_enable_interrupt(131);
#endif

  printf("\n\nMain starts\n\n");
}

void simulator_terminate(void) { M5_EXIT(); }
