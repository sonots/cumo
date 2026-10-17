#ifndef CUMO_CUDA_CUFFT_H
#define CUMO_CUDA_CUFFT_H

#include <ruby.h>
#ifdef CUFFT_FOUND
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wundef"
#include <cufft.h>
#pragma GCC diagnostic pop
#endif // CUFFT_FOUND

#if defined(__cplusplus)
extern "C" {
#if 0
} /* satisfy cc-mode */
#endif
#endif

extern VALUE cumo_cuda_eCufftError;

void
cumo_cuda_cufft_drop_plans(void);

#ifdef CUFFT_FOUND

void
cumo_cuda_cufft_check_status(cufftResult status);

#endif // CUFFT_FOUND

#if defined(__cplusplus)
#if 0
{ /* satisfy cc-mode */
#endif
}  /* extern "C" { */
#endif

#endif /* ifndef CUMO_CUDA_CUFFT_H */
