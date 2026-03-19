/*
 * ddio_helper.cpp — thin C++ wrapper around ddio.h so .cu test files
 * can call DDIO functions without hitting the #ifndef __CUDA_ARCH__ guard
 * that hides ddio.h from NVCC's device compilation pass.
 */
#include "ddio.h"

extern "C" {

uint8_t ddio_get_gpu_bus(void)   { return gpm_find_gpu_bus(); }
void    ddio_disable(uint8_t b)  { gpm_ddio_off(b); }
void    ddio_enable (uint8_t b)  { gpm_ddio_on (b); }

} /* extern "C" */
