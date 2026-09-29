#ifndef CUMO_CUDA_CUSOLVER_H
#define CUMO_CUDA_CUSOLVER_H

#include <ruby.h>
#ifdef CUSOLVER_FOUND
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wundef"
#include <cusolverDn.h>
#pragma GCC diagnostic pop
#endif // CUSOLVER_FOUND

#if defined(__cplusplus)
extern "C" {
#if 0
} /* satisfy cc-mode */
#endif
#endif

extern VALUE cumo_cuda_eCusolverError;

#ifdef CUSOLVER_FOUND

void
cumo_cuda_cusolver_check_status(cusolverStatus_t status);

#endif // CUSOLVER_FOUND

#if defined(__cplusplus)
#if 0
{ /* satisfy cc-mode */
#endif
}  /* extern "C" { */
#endif

#endif /* ifndef CUMO_CUDA_CUSOLVER_H */
