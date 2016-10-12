#include <memory>
#include <cuda.h>
#include <cublas_v2.h>

#define IDX2C(i,j,ld) (((j)*(ld))+(i))
cublasStatus_t CUBLASWINAPI scaled_Hgemm (cublasHandle_t handle,
                                                      cublasOperation_t transa,
                                                      cublasOperation_t transb,
                                                      int m,
                                                      int n,
                                                      int k,
                                                      //const __half *alpha, /* host or device pointer */
                                                      const float *alpha,
                                                      __half *A, //need to figure out way to change this to const
                                                      int lda,
                                                      __half *B, //need to figure out way to change this to const
                                                      int ldb,
                                                      //const __half *beta, /* host or device pointer */
                                                      const float *beta,
                                                      __half *C,
                                                      int ldc, bool raw_hgemm);

__global__ void createDAScalingMatrices(const int rows, const int cols, const half *A, half *DA, half *invDA) {
	int cur_row_id = blockIdx.x*blockDim.x + threadIdx.x;
	if (cur_row_id<rows){
		//half mx = A[IDX2C(cur_row_id, 0, rows)];
		float mx = fabs(__half2float(A[IDX2C(cur_row_id, 0, rows)]));
		for (int j=1; j<cols; ++j) {
			if (mx< fabs(__half2float(A[IDX2C(cur_row_id, j, rows)])))
				mx = fabs(__half2float(A[IDX2C(cur_row_id, j, rows)]));
		}
		DA[IDX2C(cur_row_id, cur_row_id, rows)] = __float2half(mx);
		invDA[IDX2C(cur_row_id, cur_row_id, rows)] = (mx==0.f ? __float2half(0.f) : __float2half(1.f/mx));
	}
};

__global__ void createDBScalingMatrices(const int rows, const int cols, const half *B, half *DB, half *invDB) {
	int cur_col_id = blockIdx.x*blockDim.x + threadIdx.x;
	if (cur_col_id<cols){
		float mx = fabs(__half2float(B[IDX2C(0, cur_col_id, rows)]));
		for (int i=1; i<rows; ++i) {
			if (mx < fabs(__half2float(B[IDX2C(i, cur_col_id, rows)])))
				mx = fabs(__half2float(B[IDX2C(i, cur_col_id, rows)]));
		}
		DB[IDX2C(cur_col_id, cur_col_id, cols)] = __float2half(mx);
		invDB[IDX2C(cur_col_id, cur_col_id, cols)] = (mx==0.f ? __float2half(0.f) : __float2half(1.f/mx));
	}
};

__global__ void zeroInit(int n, half* data) {
	int ind = blockIdx.x*blockDim.x + threadIdx.x;
	if (ind<n) {
		data[ind] = __float2half(0.f);
	}
}

__global__ void coefConverterFloat2Half(float *src, half *out) {
	(*out)=__float2half(*src);
};


__global__ void transposeKernel(int rows, int cols, half *src, half *dst) {
	/*int n = blockIdx.x;
	int m = threadIdx.x;
	dst[IDX2C(m, n, cols)] = src[IDX2C(n, m, rows)];*/
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i<rows) {
		for (int j=0;j<cols;++j) {
			dst[IDX2C(j,i,cols)] = src[IDX2C(i,j,rows)];
		}
	}
}

half* get_super_slow_transpose(int rows, int cols, half *src) {
	half *fdata;
	cudaMalloc(&fdata,sizeof(float)*rows*cols/2);
	//transposeKernel<<<rows, cols>>>(rows, cols, src, fdata);
	transposeKernel<<<(rows + 255)/256, 256>>>(rows, cols, src, fdata);
	return fdata;
}

cublasStatus_t CUBLASWINAPI scaled_Hgemm (cublasHandle_t handle,
                                                      cublasOperation_t transa,
                                                      cublasOperation_t transb,
                                                      int m,
                                                      int n,
                                                      int k,
                                                      //const __half *alpha, /* host or device pointer */
                                                      const float *alpha,
                                                      __half *A, //need to figure out way to change this to const
                                                      int lda,
                                                      __half *B, //need to figure out way to change this to const
                                                      int ldb,
                                                      //const __half *beta, /* host or device pointer */
                                                      const float *beta,
                                                      __half *C,
                                                      int ldc, bool raw_hgemm) {


	float *d_alpha, *d_beta;
	cudaMalloc(&d_alpha, sizeof(float));
	cudaMalloc(&d_beta, sizeof(float));
	cudaMemcpy(d_alpha, alpha, sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_beta, beta, sizeof(float), cudaMemcpyHostToDevice);

	half *d_h_alpha, *d_h_beta;
	cudaMalloc(&d_h_alpha, sizeof(float)/2);
	cudaMalloc(&d_h_beta, sizeof(float)/2);

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
		half *Da, *invDa, *Db, *invDb;

		__half *inptA;
		__half *inptB;

		if (transa==CUBLAS_OP_T) {
			//print_GPU_data(k, m, A);
			inptA = get_super_slow_transpose(k, m, A);
			//print_GPU_data(m, k, inptA);
			//exit(0);
		} else {
			inptA = A;
		}

		if (transb==CUBLAS_OP_T) {
			inptB = get_super_slow_transpose(n, k, B);
		} else {
			inptB = B;
		}

		//Da and invDa
		cudaMalloc(&Da, m*m*sizeof(float)/2);
		cudaMalloc(&invDa, m*m*sizeof(float)/2);
		zeroInit<<<(m*m+255)/256, 256>>>(m*m, Da);
		zeroInit<<<(m*m+255)/256, 256>>>(m*m, invDa);

		//Db and invDb
		cudaMalloc(&Db, n*n*sizeof(float)/2);
		cudaMalloc(&invDb, n*n*sizeof(float)/2);
		zeroInit<<<(n*n+255)/256, 256>>>(n*n, Db);
		zeroInit<<<(n*n+255)/256, 256>>>(n*n, invDb);

		//bookkeeping for coefficients
		float sp_alpha = 1.f, sp_beta = 0.f;
		float *sp_d_alpha, *sp_d_beta;
		cudaMalloc(&sp_d_alpha, sizeof(float));
		cudaMalloc(&sp_d_beta, sizeof(float));
		cudaMemcpy(sp_d_alpha, &sp_alpha, sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(sp_d_beta, &sp_beta, sizeof(float), cudaMemcpyHostToDevice);

		half *sp_d_h_alpha, *sp_d_h_beta;
		cudaMalloc(&sp_d_h_alpha, sizeof(float)/2);
		cudaMalloc(&sp_d_h_beta, sizeof(float)/2);

		coefConverterFloat2Half<<<1,1>>>(sp_d_alpha, sp_d_h_alpha);
		coefConverterFloat2Half<<<1,1>>>(sp_d_beta, sp_d_h_beta);
		//end of coef bookkeeping

		cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);

		//Step1&2: Aprime=invDa*A
		createDAScalingMatrices<<<(m+255)/256,256>>>(m, k, inptA, Da, invDa);

		///DEBUG
		/*print_GPU_data(m, m, Da);
		print_GPU_data(m, m, invDa);*/
		///

		half *Aprime;
		cudaMalloc(&Aprime, m*k*sizeof(float)/2);
		//std::cout<<"Step1"<<std::endl;
		status = cublasHgemm(handle,
						CUBLAS_OP_N, CUBLAS_OP_N,//transa, transb,
						m, k, m,
						sp_d_h_alpha,
						invDa, m,
						inptA, m,
						sp_d_h_beta,
						Aprime, m
						);
		//cudaDeviceSynchronize();
		//Steps3&4: Bprime=B*invDb
		createDBScalingMatrices<<<(n+255)/256,256>>>(k, n, inptB, Db, invDb);

		///DEBUG
		/*print_GPU_data(n, n, Db);
		print_GPU_data(n, n, invDb);*/
		///

		half *Bprime;
		cudaMalloc(&Bprime, k*n*sizeof(float)/2);
		//std::cout<<"Step2"<<std::endl;
		status = cublasHgemm(handle,
						CUBLAS_OP_N, CUBLAS_OP_N,//transa, transb,
						k, n, n,
						sp_d_h_alpha,
						inptB, k,
						invDb, n,
						sp_d_h_beta,
						Bprime, k
						);
		//cudaDeviceSynchronize();
		//Step 5
		half *Cprime;
		cudaMalloc(&Cprime, m*n*sizeof(float)/2);
		//std::cout<<"Step5"<<std::endl;
		status = cublasHgemm(handle,
						CUBLAS_OP_N, CUBLAS_OP_N,//transa, transb,
						m, n, k,
						sp_d_h_alpha,
						Aprime, m,
						Bprime, k,
						sp_d_h_beta,
						Cprime, m
						);
		//cudaDeviceSynchronize();
		//Step 6.1. Cprimeprime = Da*Cprime
		half *Cprimeprime;
		cudaMalloc(&Cprimeprime, m*n*sizeof(float)/2);
		//std::cout<<"Step6.1"<<std::endl;
		status = cublasHgemm(handle,
						CUBLAS_OP_N, CUBLAS_OP_N,//transa, transb,
						m, n, m,
						sp_d_h_alpha,
						Da, m,
						Cprime, m,
						sp_d_h_beta,
						Cprimeprime, m
						);
		//cudaDeviceSynchronize();
		//Step 6.2. C = Cprimeprime * Db
		//std::cout<<"Step6.2"<<std::endl;
		status = cublasHgemm(handle,
						CUBLAS_OP_N, CUBLAS_OP_N,//transa, transb,
						m, n, n,
						sp_d_h_alpha,
						Cprimeprime, m,
						Db, n,
						sp_d_h_beta,
						C, m
						);

		//cleanup
		cudaFree(sp_d_h_alpha);
		cudaFree(sp_d_h_beta);
		cudaFree(Da);
		cudaFree(invDa);
		cudaFree(Db);
		cudaFree(invDb);
		cudaFree(Aprime);
		cudaFree(Bprime);
		cudaFree(Cprime);
		cudaFree(Cprimeprime);
		if (transa==CUBLAS_OP_T) {
			cudaFree(inptA);
		}
		if (transb==CUBLAS_OP_T) {
			cudaFree(inptB);
		}
		cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);
	}

/*	auto status = cublasSgemmEx(handle,
						   transa,
	                       transb,
	                       m,
	                       n,
	                       k,
	                       d_alpha, // host or device pointer
	                       A,
	                       CUDA_R_16F,
	                       lda,
	                       B,
	                       CUDA_R_16F,
	                       ldb,
	                       d_beta, // host or device pointer
	                       C,
	                       CUDA_R_16F,
	                       ldc);*/
	cudaFree(d_alpha);
	cudaFree(d_beta);
	cudaFree(d_h_alpha);
	cudaFree(d_h_beta);
	return status;
}
