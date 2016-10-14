/*
 * scaled_hgemm.h
 *
 *  Created on: Oct 13, 2016
 *      Author: okuchaiev
 */

#ifndef SCALED_HGEMM_H_
#define SCALED_HGEMM_H_
#include <memory>
#include <cuda.h>
#include <cublas_v2.h>

cublasStatus_t CUBLASWINAPI scaled_Hgemm (cublasHandle_t handle,
                                                      cublasOperation_t transa,
                                                      cublasOperation_t transb,
                                                      int m,
                                                      int n,
                                                      int k,
                                                      const float *alpha,
                                                      const __half *A,
                                                      int lda,
                                                      const __half *B,
                                                      int ldb,
                                                      const float *beta,
                                                      __half *C,
                                                      int ldc, bool raw_hgemm);


#endif /* SCALED_HGEMM_H_ */
