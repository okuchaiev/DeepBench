#include "scaled_hgemm.h"
#include <iostream>
#include <stdio.h>
#include <cuda.h>
#define IDX2C(i,j,ld) (((j)*(ld))+(i))
#define min_represent 0.0000152663f
#define DivCnst 256



inline void gpuErrchk(cudaError_t code, char *label)
{
   if (code != cudaSuccess)
   {
	  std::cout<<std::endl<<cudaGetErrorString(code)<<"  LABEL: "<<label<<std::endl;
      //exit(code);
   }
}

/**
 * Call like this <<<(rows+255)/256, 256>>> if reduce_cols = true
 * else <<<(cols+255)/256, 256>>>
 * inpt still has ld = rows regardless of reduce_cols
 */

//(k, n, B, Db, Bprime, false, transb==CUBLAS_OP_T);
__global__ void createScalingDiagonal(const int rows, const int cols, const __half *inpt, __half *res, __half* scaled_inpt, bool reduce_cols, bool transpose_input) {

	int id = blockIdx.x*blockDim.x + threadIdx.x; //row index if reduce_cols = true, else column index

	if (reduce_cols && id < rows) { //id is row index
		float mx = (!transpose_input ? fabs(__half2float(inpt[IDX2C(id, 0, rows)])) : fabs(__half2float(inpt[IDX2C(0, id, cols)])));

		for (int j=1; j<cols; ++j) {
			float element = (!transpose_input ? fabs(__half2float(inpt[IDX2C(id, j, rows)])) : fabs(__half2float(inpt[IDX2C(j, id, cols)])));
			if (mx < element)
				mx = element;
		}
		float scale = (mx <= min_represent ? 1.f : mx);
		res[id] = __float2half(scale);

		for (int j=0; j<cols; ++j) {
			float element = (!transpose_input ? __half2float(inpt[IDX2C(id, j, rows)]) : __half2float(inpt[IDX2C(j, id, cols)]));
			scaled_inpt[IDX2C(id, j, rows)] = __float2half(element/scale);
		}


	} else if (!reduce_cols && id < cols) { //id is column index
		float mx = (!transpose_input ? fabs(__half2float(inpt[IDX2C(0, id, rows)])) : fabs(__half2float(inpt[IDX2C(id, 0, cols)])) );

		for (int i=1; i<rows; ++i) {
			float element = (!transpose_input ? fabs(__half2float(inpt[IDX2C(i, id, rows)])) : fabs(__half2float(inpt[IDX2C(id, i, cols)])));
			if (mx < element)
				mx = element;
			}
		float scale = (mx <= min_represent ? 1.f : mx);
		res[id] = __float2half(scale);

		for (int i=0; i<rows; ++i) {
			float element = (!transpose_input ? __half2float(inpt[IDX2C(i, id, rows)]) : __half2float(inpt[IDX2C(id, i, cols)]));
			scaled_inpt[IDX2C(i, id, rows)] = __float2half(element/scale);
		}
	}
}

//* Call like this <<<(rows+255)/256, 256>>> if left
//  else <<<(cols+255)/256, 256>>>
// Left means scales*data, right means data*scales
__global__ void do_scaling(const int rows, const int cols, __half *data, const __half* scales, bool left, bool inv_scale) {

	int id = blockIdx.x*blockDim.x + threadIdx.x; //row index in data if left, else this is a column index
	float scale_factor = (!inv_scale ? __half2float(scales[id]) : 1.f/__half2float(scales[id]));

	if (left && id < rows) {
		for (int j=0; j<cols; ++j)
			data[IDX2C(id, j, rows)] = __float2half(__half2float(data[IDX2C(id, j, rows)]) * scale_factor);
	} else if (!left && id < cols) {
		for (int i=0; i<rows; ++i)
			data[IDX2C(i, id, rows)] = __float2half(__half2float(data[IDX2C(i, id, rows)]) * scale_factor);
	}
}

//Does out=out*beta+arg
__global__ void scale_add(const int rows, const int cols, const __half *arg, __half *out, float beta) {
	int id = blockIdx.x*blockDim.x + threadIdx.x;
	if (id<rows*cols) {
		out[id] = __float2half((beta != 0.f ? __half2float(out[id])*beta + __half2float(arg[id]) : __half2float(arg[id])));
	}
}

__global__ void coefConverterFloat2Half(const float *src, __half *out) {
	(*out)=__float2half(*src);
};

static const char *_cudaGetErrorEnum(cublasStatus_t error)
{
    switch (error)
    {
        case CUBLAS_STATUS_SUCCESS:
            return "CUBLAS_STATUS_SUCCESS";

        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "CUBLAS_STATUS_NOT_INITIALIZED";

        case CUBLAS_STATUS_ALLOC_FAILED:
            return "CUBLAS_STATUS_ALLOC_FAILED";

        case CUBLAS_STATUS_INVALID_VALUE:
            return "CUBLAS_STATUS_INVALID_VALUE";

        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "CUBLAS_STATUS_ARCH_MISMATCH";

        case CUBLAS_STATUS_MAPPING_ERROR:
            return "CUBLAS_STATUS_MAPPING_ERROR";

        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "CUBLAS_STATUS_EXECUTION_FAILED";

        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "CUBLAS_STATUS_INTERNAL_ERROR";
    }

    return "<unknown>";
}

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
                                                      int ldc, bool raw_hgemm) {

	cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);
	float *d_alpha, *d_beta;
	gpuErrchk(cudaMalloc((void **)&d_alpha, sizeof(float)), "1");
	gpuErrchk(cudaMalloc((void **)&d_beta, sizeof(float)), "2");
	cudaMemcpy(d_alpha, alpha, sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_beta, beta, sizeof(float), cudaMemcpyHostToDevice);

	__half *d_h_alpha, *d_h_beta;
	gpuErrchk(cudaMalloc((void **)&d_h_alpha, sizeof(__half)), "3");
	gpuErrchk(cudaMalloc((void **)&d_h_beta, sizeof(__half)), "4");

	coefConverterFloat2Half<<<1,1>>>(d_alpha, d_h_alpha);
	gpuErrchk( cudaPeekAtLastError() , "5");
	gpuErrchk( cudaDeviceSynchronize(), "6" );

	coefConverterFloat2Half<<<1,1>>>(d_beta, d_h_beta);
	gpuErrchk( cudaPeekAtLastError(), "7" );
	gpuErrchk( cudaDeviceSynchronize(), "8" );

	cublasStatus_t status;
	if (raw_hgemm) {
		status = cublasHgemm(handle,
				transa, transb,
				m, n, k,
				d_h_alpha,
				A, lda,
				B, ldb,
				d_h_beta,
				C, ldc);
		//cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);

	} else { //do outside scaling algorithm
		__half *Da, *Db, *Aprime, *Bprime, *Cprime;
		gpuErrchk(cudaMalloc((void **)&Da, sizeof(__half)*m), "9");//
		gpuErrchk(cudaMalloc((void **)&Db, sizeof(__half)*n), "10");//
		gpuErrchk(cudaMalloc((void **)&Aprime, m*k*sizeof(__half)), "11");//
		gpuErrchk(cudaMalloc((void **)&Bprime, k*n*sizeof(__half)), "12");//
		gpuErrchk(cudaMalloc((void **)&Cprime, m*n*sizeof(__half)), "13");//

		//bookkeeping for coefficients
		float sp_beta = 0.f; //host
		float  *sp_d_beta; //device
		gpuErrchk(cudaMalloc((void **)&sp_d_beta, sizeof(float)), "14");
		gpuErrchk(cudaMemcpy(sp_d_beta, &sp_beta, sizeof(float), cudaMemcpyHostToDevice), "15");
		__half *sp_d_h_beta;
		gpuErrchk(cudaMalloc((void **)&sp_d_h_beta, sizeof(__half)), "16");
		coefConverterFloat2Half<<<1,1>>>(sp_d_beta, sp_d_h_beta);
		gpuErrchk( cudaPeekAtLastError(), "17" );
		gpuErrchk( cudaDeviceSynchronize(), "18" );

		//Step 1&2
		createScalingDiagonal<<<(m+DivCnst-1)/DivCnst, DivCnst>>>(m, k, A, Da, Aprime, true, transa==CUBLAS_OP_T);
		gpuErrchk( cudaPeekAtLastError(), "19" );
		gpuErrchk( cudaDeviceSynchronize(), "20" );
		//Step 3&4
		createScalingDiagonal<<<(n+DivCnst-1)/DivCnst, DivCnst>>>(k, n, B, Db, Bprime, false, transb==CUBLAS_OP_T);
		gpuErrchk( cudaPeekAtLastError(), "21" );
		gpuErrchk( cudaDeviceSynchronize(), "22" );

		//Step 5
		status = cublasHgemm(handle,
						CUBLAS_OP_N, CUBLAS_OP_N,//transa, transb,
						m, n, k,
						d_h_alpha,
						Aprime, lda,
						//Aprime, m,
						//Bprime, k,
						Bprime, ldb,
						sp_d_h_beta,
						//Cprime, m);
						Cprime, ldc);

		//cudaFree(sp_d_alpha);//
		cudaFree(sp_d_beta);//
		//cudaFree(sp_d_h_alpha);//
		cudaFree(sp_d_h_beta);//
		cudaFree(Aprime);//
		cudaFree(Bprime);//
		do_scaling<<<(m+DivCnst-1)/DivCnst, DivCnst>>>(m, n, Cprime, Da, true, false);
		cudaDeviceSynchronize();
		do_scaling<<<(n+DivCnst-1)/DivCnst, DivCnst>>>(m, n, Cprime, Db, false, false);
		cudaDeviceSynchronize();
		scale_add<<<(m*n+DivCnst-1)/DivCnst, DivCnst>>>(m, n, Cprime, C, *beta);
		//cleanup
		cudaFree(Da);//
		cudaFree(Db);//
		cudaFree(Cprime);//
	}//end of outside scaling
	cudaFree(d_alpha);//
	cudaFree(d_beta);//
	cudaFree(d_h_alpha);//
	cudaFree(d_h_beta);//
	cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);
	//checkCudaErrors();
	if (status!=CUBLAS_STATUS_SUCCESS) {
		std::cout << std::endl << std::endl;
		std::cout << _cudaGetErrorEnum(status) << std::endl;

	}
	return status;
	//return CUBLAS_STATUS_SUCCESS;
}
