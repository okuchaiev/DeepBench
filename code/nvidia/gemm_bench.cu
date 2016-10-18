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
//#include "scaled_hgemm.cu"
#include "scaled_hgemm.h"



int time_Hgemm(Tensor<half> A, Tensor<half> B, Tensor<half> C, bool a_t, bool b_t, cublasHandle_t cublas_handle, bool true_hgemm) {
	//const float alpha = 1.f;// / static_cast<float>(A.dims()[1]);
	//const float beta =  1.f;
	const float alpha = 1.f;
	const float beta = 0.f; //1.f;

	int m = C.dims()[0];
	int k = a_t ? A.dims()[0] : A.dims()[1];
	int n = C.dims()[1];

	int numRepeats = std::max(std::ceil(1e11 / (m * k * n)), 10.);

	    // Warm up
	cublasStatus_t stat = scaled_Hgemm(cublas_handle,
	                a_t ? CUBLAS_OP_T : CUBLAS_OP_N,
	                b_t ? CUBLAS_OP_T : CUBLAS_OP_N,
	                m,
	                n,
	                k,
	                &alpha,
	                A.begin(), A.dims()[0],
	                B.begin(), B.dims()[0],
	                &beta,
	                C.begin(), C.dims()[0], true_hgemm);
	//return 1;

	if (stat != CUBLAS_STATUS_SUCCESS) {
		throw std::runtime_error("hgemm failed");
	}
	//return 0;

	cudaDeviceSynchronize();
	auto start = std::chrono::steady_clock::now();

	for (int i = 0; i < numRepeats; ++i) {
		cublasStatus_t stat = scaled_Hgemm(cublas_handle,
				a_t ? CUBLAS_OP_T : CUBLAS_OP_N,
	            b_t ? CUBLAS_OP_T : CUBLAS_OP_N,
	            m,
	            n,
	            k,
	            &alpha,
	            A.begin(), A.dims()[0],
	            B.begin(), B.dims()[0],
	            &beta,
	            C.begin(), C.dims()[0], true_hgemm);
		if (stat != CUBLAS_STATUS_SUCCESS) {
	            throw std::runtime_error("hgemm failed");
	    }
	}
	cudaDeviceSynchronize();

	auto end = std::chrono::steady_clock::now();
	return static_cast<int>(std::chrono::duration<double, std::micro>(end - start).count() / numRepeats);
}


int time_Sgemm(Tensor<float> A, Tensor<float> B, Tensor<float> C, bool a_t, bool b_t, cublasHandle_t cublas_handle) {
    const float alpha = 1.f;// / static_cast<float>(A.dims()[1]);
    const float beta  = 0.f;//1.f;

    int m = C.dims()[0];
    int k = a_t ? A.dims()[0] : A.dims()[1];
    int n = C.dims()[1];

    int numRepeats = std::max(std::ceil(1e11 / (m * k * n)), 10.);

    // Warm up
    cublasStatus_t stat = cublasSgemm(cublas_handle,
                a_t ? CUBLAS_OP_T : CUBLAS_OP_N,
                b_t ? CUBLAS_OP_T : CUBLAS_OP_N,
                m,
                n,
                k,
                &alpha,
                A.begin(), A.dims()[0],
                B.begin(), B.dims()[0],
                &beta,
                C.begin(), C.dims()[0]);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("sgemm failed");
    }

    cudaDeviceSynchronize();

    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < numRepeats; ++i) {
        cublasStatus_t stat = cublasSgemm(cublas_handle,
                    a_t ? CUBLAS_OP_T : CUBLAS_OP_N,
                    b_t ? CUBLAS_OP_T : CUBLAS_OP_N,
                    m,
                    n,
                    k,
                    &alpha,
                    A.begin(), A.dims()[0],
                    B.begin(), B.dims()[0],
                    &beta,
                    C.begin(), C.dims()[0]);
        if (stat != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("sgemm failed");
        }
    }
    cudaDeviceSynchronize();

    auto end = std::chrono::steady_clock::now();

    return static_cast<int>(std::chrono::duration<double, std::micro>(end - start).count() / numRepeats);

}

int main(int argc, char **argv) {
    cudaFree(0);
	cublasHandle_t cublas_handle;
	cublasStatus_t status = cublasCreate(&cublas_handle);
	if (status != CUBLAS_STATUS_SUCCESS) {
	   std::cout << "CUBLAS init failed" << std::endl;
	}
    curandGenerator_t curand_gen;
    curandCreateGenerator(&curand_gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(curand_gen, 123ULL);

    std::vector<std::tuple<int, int, int, bool, bool>> problems  = {
    	//std::make_tuple(16, 16, 8, true, false)
    	std::make_tuple(3000, 6000, 20, true, false),
    	std::make_tuple(1760, 32, 1760, false, false),
        std::make_tuple(1760, 64, 1760, false, false),
        std::make_tuple(1760, 128, 1760, false, false),
        std::make_tuple(1760, 7000, 1760, false, false),
        std::make_tuple(2048, 16, 2048, false, false),
        std::make_tuple(2048, 32, 2048, false, false),
        std::make_tuple(2048, 64, 2048, false, false),
        std::make_tuple(2048, 128, 2048, false, false),
        std::make_tuple(2048, 7000, 2048, false, false),
        std::make_tuple(2560, 16, 2560, false, false),
        std::make_tuple(2560, 32, 2560, false, false),
        std::make_tuple(2560, 64, 2560, false, false),
        std::make_tuple(2560, 128, 2560, false, false),
        std::make_tuple(2560, 7000, 2560, false, false),
        std::make_tuple(4096, 16, 4096, false, false),
        std::make_tuple(4096, 32, 4096, false, false),
        std::make_tuple(4096, 64, 4096, false, false),
        std::make_tuple(4096, 128, 4096, false, false),
        std::make_tuple(4096, 7000, 4096, false, false),
        std::make_tuple(1760, 16, 1760, true, false),
        std::make_tuple(1760, 32, 1760, true, false),
        std::make_tuple(1760, 64, 1760, true, false),
        std::make_tuple(1760, 128, 1760, true, false),
        std::make_tuple(1760, 7000, 1760, true, false),
        std::make_tuple(2048, 16, 2048, true, false),
        std::make_tuple(2048, 32, 2048, true, false),
        std::make_tuple(2048, 64, 2048, true, false),
        std::make_tuple(2048, 128, 2048, true, false),
        std::make_tuple(2048, 7000, 2048, true, false),
        std::make_tuple(2560, 16, 2560, true, false),
        std::make_tuple(2560, 32, 2560, true, false),
        std::make_tuple(2560, 64, 2560, true, false),
        std::make_tuple(2560, 128, 2560, true, false),
        std::make_tuple(2560, 7000, 2560, true, false),
        std::make_tuple(4096, 16, 4096, true, false),
        std::make_tuple(4096, 32, 4096, true, false),
        std::make_tuple(4096, 64, 4096, true, false),
        std::make_tuple(4096, 128, 4096, true, false),
        std::make_tuple(4096, 7000, 4096, true, false),
        std::make_tuple(1760, 7133, 1760, false, true),
        std::make_tuple(2048, 7133, 2048, false, true),
        std::make_tuple(2560, 7133, 2560, false, true),
        std::make_tuple(4096, 7133, 4096, false, true),
        std::make_tuple(5124, 9124, 1760, false, false),
        std::make_tuple(35, 8457, 1760, false, false),
        std::make_tuple(5124, 9124, 2048, false, false),
        std::make_tuple(35, 8457, 2048, false, false),
        std::make_tuple(5124, 9124, 2560, false, false),
        std::make_tuple(35, 8457, 2560, false, false),
        std::make_tuple(5124, 9124, 4096, false, false),
        std::make_tuple(35, 8457, 4096, false, false),
        std::make_tuple(5124, 9124, 1760, true, false),
        std::make_tuple(35, 8457, 1760, true, false),
        std::make_tuple(5124, 9124, 2048, true, false),
        std::make_tuple(35, 8457, 2048, true, false),
        std::make_tuple(5124, 9124, 2560, true, false),
        std::make_tuple(35, 8457, 2560, true, false),
        std::make_tuple(5124, 9124, 4096, true, false),
        std::make_tuple(35, 8457, 4096, true, false),
        std::make_tuple(7680, 16, 2560, false, false),
        std::make_tuple(7680, 32, 2560, false, false),
        std::make_tuple(7680, 64, 2560, false, false),
        std::make_tuple(7680, 128, 2560, false, false),
        std::make_tuple(7680, 16, 2560, true, false),
        std::make_tuple(7680, 32, 2560, true, false),
        std::make_tuple(7680, 64, 2560, true, false),
        std::make_tuple(7680, 128, 2560, true, false),
        std::make_tuple(3072, 16, 1024, false, false),
        std::make_tuple(3072, 32, 1024, false, false),
        std::make_tuple(3072, 64, 1024, false, false),
        std::make_tuple(3072, 128, 1024, false, false),
        std::make_tuple(3072, 16, 1024, true, false),
        std::make_tuple(3072, 32, 1024, true, false),
        std::make_tuple(3072, 64, 1024, true, false),
        std::make_tuple(3072, 128, 1024, true, false),
        std::make_tuple(3072, 7435, 1024, false, true),
        std::make_tuple(7680, 5481, 2560, false, true)

    };

    std::cout << std::setw(30) << "Times for gemm" << std::endl;
    std::cout << std::setfill('-') << std::setw(88) << "-" << std::endl;
    std::cout << std::setfill(' ');
    std::cout << "    m       n      k      a_t     b_t      time (usec)      timeHScaled (usec)      timeHRaw (usec)      N(c)      N(c16)      N(cc16)      SandHnormDiff      SandRawDiff " << std::endl;
    for (const auto &problem : problems) {
        int m, n, k;
        bool a_t, b_t;
        std::tie(m, n, k, a_t, b_t) = problem;

        auto a = rand({m, k}, curand_gen);
        auto b = rand({a_t ? m : (b_t ? n : k), b_t ? k : n}, curand_gen);
        auto c = zeros({a_t ? k : m, n});

        std::cout << std::setw(7) << m;
        std::cout << std::setw(7) << n;
        std::cout << std::setw(7) << k;
        std::cout << std::setw(7) << a_t ? "t" : "n";
        std::cout << std::setw(7) << b_t ? "t" : "n";
        std::cout << std::setw(13) << std::setprecision(6) << time_Sgemm(a, b, c, a_t, b_t, cublas_handle);
        auto a16 = floatTensor2half(a);
        auto b16 = floatTensor2half(b);
        auto c16 = floatTensor2half(c);
        auto cc16 = floatTensor2half(c);
        std::cout << std::setw(16) << std::setprecision(6) << time_Hgemm(a16, b16, c16, a_t, b_t, cublas_handle, false);
        std::cout << std::setw(24) << std::setprecision(6) << time_Hgemm(a16, b16, cc16, a_t, b_t, cublas_handle, true);
        std::cout << std::setw(24) << std::setprecision(6) << frobeniusNorm(c, cublas_handle);
        std::cout << std::setw(12) << std::setprecision(6) << frobeniusNorm(halfTensor2float(c16), cublas_handle);
        std::cout << std::setw(12) << std::setprecision(6) << frobeniusNorm(halfTensor2float(cc16), cublas_handle);
        std::cout << std::setw(16) << std::setprecision(6) << frobeniusNorm(minus(c, halfTensor2float(c16), cublas_handle), cublas_handle)/frobeniusNorm(c, cublas_handle);
        std::cout << std::setw(12) << std::setprecision(6) << frobeniusNorm(minus(c, halfTensor2float(cc16), cublas_handle), cublas_handle)/frobeniusNorm(c, cublas_handle);
        std::cout << std::endl;

        //DEBUG
        //exit(0);
    }
    cublasDestroy(cublas_handle);
    curandDestroyGenerator(curand_gen);

    return 0;
}
