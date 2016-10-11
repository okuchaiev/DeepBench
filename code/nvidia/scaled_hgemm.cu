#include <chrono>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <tuple>
#include <vector>
#include <cuda.h>
#include <cublas_v2.h>
#include <curand.h>

#include <thrust/device_ptr.h>
#include <thrust/fill.h>
#include "tensor.h"

cublasStatus_t CUBLASWINAPI scaled_Hgemm (cublasHandle_t handle,
                                                      cublasOperation_t transa,
                                                      cublasOperation_t transb,
                                                      int m,
                                                      int n,
                                                      int k,
                                                      //const __half *alpha, /* host or device pointer */
                                                      const float *alpha,
                                                      const __half *A,
                                                      int lda,
                                                      const __half *B,
                                                      int ldb,
                                                      //const __half *beta, /* host or device pointer */
                                                      const float *beta,
                                                      __half *C,
                                                      int ldc) {


	return cublasSgemmEx(handle,
						   transa,
	                       transb,
	                       m,
	                       n,
	                       k,
	                       alpha, // host or device pointer
	                       A,
	                       CUDA_R_16F,
	                       lda,
	                       B,
	                       CUDA_R_16F,
	                       ldb,
	                       beta, // host or device pointer
	                       C,
	                       CUDA_R_16F,
	                       ldc);


/*return cublasHgemm(handle,
            transa,
            transb,
            m,
            n,
            k,
            alpha,
            A, lda,
            B, ldb,
            beta,
            C, ldc);*/
}
