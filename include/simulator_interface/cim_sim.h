#ifndef _SIMULATOR_INTERFACE__CIM_SIM_H_
#define _SIMULATOR_INTERFACE__CIM_SIM_H_

#ifdef __cplusplus
extern "C" {
#endif

void simulator_init(void);

void simulator_terminate(void);

void simulator_mark_start(void);

void simulator_mark_end(void);

#ifdef __cplusplus
}
#endif

#endif /* _SIMULATOR_INTERFACE__CIM_SIM_H_ */