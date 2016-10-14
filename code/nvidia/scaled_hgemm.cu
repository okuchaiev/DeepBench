#include "scaled_hgemm.h"

#define IDX2C(i,j,ld) (((j)*(ld))+(i))
#define min_represent 0.00001526624f

/**
 * Call like this <<<(rows+255)/256, 256>>> if reduce_cols = true
 * <<<(cols+255)/256, 256>>> else
 * inpt still has ld = rows regardless of reduce_cols
 */
__global__ void createScalingDiagonal(const int rows, const int cols, const __half *inpt, __half *res, bool reduce_cols) {
	int id = blockIdx.x*blockDim.x + threadIdx.x;
	if (reduce_cols && id < rows) {
		float mx = fabs(__half2float(inpt[IDX2C(id, 0, rows)]));
		for (int j=1; j<cols; ++j) {
			if (mx < fabs(__half2float(inpt[IDX2C(id, j, rows)])))
				mx = fabs(__half2float(inpt[IDX2C(id, j, rows)]));
		}
		res[id] = (mx <= min_represent ? __float2half(1.f) :__float2half(mx));

	} else if (!reduce_cols && id < cols) {
		float mx = fabs(__half2float(inpt[IDX2C(0, id, rows)]));
		for (int i=1; i<rows; ++i) {
			if (mx < fabs(__half2float(inpt[IDX2C(i, id, rows)])))
				mx = fabs(__half2float(inpt[IDX2C(i, id, rows)]));
			}
		res[id] = (mx <= min_represent ? __float2half(1.f) :__float2half(mx));
	}
}

//* Call like this <<<(rows+255)/256, 256>>> u
__global__ void do_scaling(const int rows, const int cols, __half *data, const __half* scales, bool left, bool inv_scale) {
	int id = blockIdx.x*blockDim.x + threadIdx.x;
	float scale_factor = (!inv_scale ? __half2float(scales[id]) : 1.f/__half2float(scales[id]));
	if (left && id < rows) {
		for (int j=0; j<cols; ++j)
			data[IDX2C(id, j, rows)] = __float2half(__half2float(data[IDX2C(id, j, rows)])* scale_factor);
	} else if (!left && id < cols) {
		for (int i=0; i<rows; ++i)
			data[IDX2C(i, id, rows)] = __float2half(__half2float(data[IDX2C(i, id, rows)])* scale_factor);
	}
}

//Does out=out*beta+arg
__global__ void scale_add(const int rows, const int cols, const __half *arg, __half *out, float beta) {
	int id = blockIdx.x*blockDim.x + threadIdx.x;
	if (id<rows*cols) {
		out[id] = __float2half((beta != 0.f ? __half2float(out[id])*beta + __half2float(arg[id]) : __half2float(arg[id])));
	}
}


/*
__global__ void createDAScalingMatrices(const int rows, const int cols, const __half *A, __half *DA, __half *invDA) {
	int cur_row_id = blockIdx.x*blockDim.x + threadIdx.x;
	if (cur_row_id<rows){
		//half mx = A[IDX2C(cur_row_id, 0, rows)];
		float mx = fabs(__half2float(A[IDX2C(cur_row_id, 0, rows)]));
		for (int j=1; j<cols; ++j) {
			if (mx< fabs(__half2float(A[IDX2C(cur_row_id, j, rows)])))
				mx = fabs(__half2float(A[IDX2C(cur_row_id, j, rows)]));
		}
		DA[IDX2C(cur_row_id, cur_row_id, rows)] = (mx <= min_represent ? __float2half(1.f) :__float2half(mx));
		invDA[IDX2C(cur_row_id, cur_row_id, rows)] = (mx <= min_represent ? __float2half(1.f) : __float2half(1.f/mx));
	}
};

__global__ void createDBScalingMatrices(const int rows, const int cols, const __half *B, __half *DB, __half *invDB) {
	int cur_col_id = blockIdx.x*blockDim.x + threadIdx.x;
	if (cur_col_id<cols){
		float mx = fabs(__half2float(B[IDX2C(0, cur_col_id, rows)]));
		for (int i=1; i<rows; ++i) {
			if (mx < fabs(__half2float(B[IDX2C(i, cur_col_id, rows)])))
				mx = fabs(__half2float(B[IDX2C(i, cur_col_id, rows)]));
		}
		DB[IDX2C(cur_col_id, cur_col_id, cols)] = (mx <= min_represent ? __float2half(1.f) :__float2half(mx));;
		invDB[IDX2C(cur_col_id, cur_col_id, cols)] = (mx <= min_represent ? __float2half(1.f) : __float2half(1.f/mx));
	}
};

__global__ void zeroInit(int n, __half* data) {
	int ind = blockIdx.x*blockDim.x + threadIdx.x;
	if (ind<n) {
		data[ind] = __float2half(0.f);
	}
}*/

__global__ void coefConverterFloat2Half(const float *src, __half *out) {
	(*out)=__float2half(*src);
};


__global__ void transposeKernel(int rows, int cols, const __half *src, __half *dst) {
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i<rows) {
		for (int j=0;j<cols;++j) {
			dst[IDX2C(j,i,cols)] = src[IDX2C(i,j,rows)];
		}
	}
}

__half* get_super_slow_transpose(int rows, int cols, const __half *src) {
	__half *fdata;
	cudaMalloc(&fdata,sizeof(__half)*rows*cols);
	transposeKernel<<<(rows + 255)/256, 256>>>(rows, cols, src, fdata);
	return fdata;
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


	float *d_alpha, *d_beta;
	cudaMalloc(&d_alpha, sizeof(float)); //
	cudaMalloc(&d_beta, sizeof(float));  //
	cudaMemcpy(d_alpha, alpha, sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_beta, beta, sizeof(float), cudaMemcpyHostToDevice);

	__half *d_h_alpha, *d_h_beta;
	cudaMalloc(&d_h_alpha, sizeof(__half)); //
	cudaMalloc(&d_h_beta, sizeof(__half));  //

	coefConverterFloat2Half<<<1,1>>>(d_alpha, d_h_alpha);
	coefConverterFloat2Half<<<1,1>>>(d_beta, d_h_beta);

	cublasStatus_t status;
	if (raw_hgemm) {
		cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);
		status = cublasHgemm(handle,
				transa, transb,
				m, n, k,
				d_h_alpha,
				A, lda,
				B, ldb,
				d_h_beta,
				C, ldc
				);
		cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);

	} else { //do outside scaling algorithm
		__half *inptA;
		__half *inptB;
		__half *Da, *Db;
		cudaMalloc(&Da, sizeof(__half)*m);
		cudaMalloc(&Db, sizeof(__half)*n);

		__half *Aprime;
		cudaMalloc(&Aprime, m*k*sizeof(__half));
		__half *Bprime;
		cudaMalloc(&Bprime, k*n*sizeof(__half));

		//bookkeeping for coefficients
		float sp_alpha = 1.f, sp_beta = 0.f;
		float *sp_d_alpha, *sp_d_beta;
		cudaMalloc(&sp_d_alpha, sizeof(float)); //
		cudaMalloc(&sp_d_beta, sizeof(float));  //
		cudaMemcpy(sp_d_alpha, &sp_alpha, sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(sp_d_beta, &sp_beta, sizeof(float), cudaMemcpyHostToDevice);

		__half *sp_d_h_alpha, *sp_d_h_beta;
		cudaMalloc(&sp_d_h_alpha, sizeof(__half));//
		cudaMalloc(&sp_d_h_beta, sizeof(__half));//

		coefConverterFloat2Half<<<1,1>>>(sp_d_alpha, sp_d_h_alpha);
		coefConverterFloat2Half<<<1,1>>>(sp_d_beta, sp_d_h_beta);
		cudaDeviceSynchronize();
		//end of coef bookkeeping

		if (transa==CUBLAS_OP_T) {
			//inptA = get_super_slow_transpose(k, m, A);
			inptA = get_super_slow_transpose(k, m, A);
			cudaDeviceSynchronize();
			createScalingDiagonal<<<(m+255)/256, 256>>>(m, k, inptA, Da, true);
			cudaDeviceSynchronize();
			cudaMemcpy(Aprime, inptA, sizeof(__half)*m*k, cudaMemcpyDeviceToDevice);
		} else {
			createScalingDiagonal<<<(m+255)/256, 256>>>(m, k, A, Da, true);
			cudaDeviceSynchronize();
			cudaMemcpy(Aprime, A, sizeof(__half)*m*k, cudaMemcpyDeviceToDevice);
		}

		if (transb==CUBLAS_OP_T) {
			//inptB = get_super_slow_transpose(n, k, B);
			inptB = get_super_slow_transpose(n, k, B);
			cudaDeviceSynchronize();
			createScalingDiagonal<<<(n+255)/256, 256>>>(k, n, inptB, Db, false);
			cudaDeviceSynchronize();
			cudaMemcpy(Bprime, inptB, sizeof(__half)*n*k, cudaMemcpyDeviceToDevice);
		} else {
			createScalingDiagonal<<<(n+255)/256, 256>>>(k, n, B, Db, false);
			cudaDeviceSynchronize();
			cudaMemcpy(Bprime, B, sizeof(__half)*n*k, cudaMemcpyDeviceToDevice);
		}
		cudaDeviceSynchronize();

		do_scaling<<<(m+255)/256, 256>>>(m, k, Aprime, Da, true, true);
		cudaDeviceSynchronize();
		do_scaling<<<(n+255)/256, 256>>>(k, n, Bprime, Db, false, true);

		__half *Cprime;
		cudaMalloc(&Cprime, m*n*sizeof(__half));
		cudaDeviceSynchronize();
		cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);
		status = cublasHgemm(handle,
						CUBLAS_OP_N, CUBLAS_OP_N,//transa, transb,
						m, n, k,
						sp_d_h_alpha,
						Aprime, m,
						Bprime, k,
						sp_d_h_beta,
						Cprime, m);
		cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);
		cudaDeviceSynchronize();
		do_scaling<<<(m+255)/256, 256>>>(m, n, Cprime, Da, true, false);
		cudaDeviceSynchronize();
		do_scaling<<<(n+255)/256, 256>>>(m, n, Cprime, Db, false, false);
		cudaDeviceSynchronize();
		scale_add<<<(m*n)/256, 256>>>(m, n, Cprime, C, *beta);

		//cleanup
		cudaFree(sp_d_alpha);//
		cudaFree(sp_d_beta);//
		cudaFree(sp_d_h_alpha);//
		cudaFree(sp_d_h_beta);//
		cudaFree(Da);//
		cudaFree(Db);//
		cudaFree(Aprime);//
		cudaFree(Bprime);//
		cudaFree(Cprime);//
		if (transa==CUBLAS_OP_T) {
			cudaFree(inptA);
		}
		if (transb==CUBLAS_OP_T) {
			cudaFree(inptB);
		}
	}//end of outside scaling
	cudaFree(d_alpha);//
	cudaFree(d_beta);//
	cudaFree(d_h_alpha);//
	cudaFree(d_h_beta);//
	return status;
}
