#pragma once

#include <vector>
#include <numeric>
#include <memory>

#include <curand.h>

#include <thrust/device_ptr.h>
#include <thrust/fill.h>

/**
 * Kernel from on device conversion from float to half arrays
 */
__global__
void copyFloatArray2HalfArray(int n, float *src, half *dst) {
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < n) dst[i] = __float2half(src[i]);
}

/**
 * Kernel from on device conversion from half to float arrays
 */
__global__
void copyHalfArray2FloatArray(int n, half *src, float *dst) {
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < n) dst[i] = __half2float(src[i]);
}

__global__
void shiftAndScale(int n, float *src) {
	float scale = 0.01;
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < n) src[i] = src[i]*(2.f*scale) - scale;
}


template <typename T>
class Tensor {

    struct deleteCudaPtr {
        void operator()(T *p) const {
            cudaFree(p);
        }
    };

    std::shared_ptr<T> ptr_;

public:
    std::vector<int> dims_;
    int size_;

    Tensor() {}

    Tensor(std::vector<int> dims) : dims_(dims) {
        T* tmp_ptr;
        size_ = std::accumulate(dims_.begin(), dims_.end(), 1, std::multiplies<int>());
        cudaMalloc(&tmp_ptr, sizeof(T) * size_);
        ptr_.reset(tmp_ptr, deleteCudaPtr());
    }

    T* begin() const { return ptr_.get(); }
    T* end()   const { return ptr_.get() + size_; }
    int size() const { return size_; }
    std::vector<int> dims() const { return dims_; }
};

Tensor<half> floatTensor2half(Tensor<float> const &x) {
	Tensor<half> res(x.dims_);
	copyFloatArray2HalfArray<<<(x.size_+255)/256, 256>>>(x.size_, x.begin(), res.begin());
	return res;
}

Tensor<float> halfTensor2float(Tensor<half> const &x) {
	Tensor<float> res(x.dims_);
	copyHalfArray2FloatArray<<<(x.size_+255)/256, 256>>>(x.size_, x.begin(), res.begin());
	return res;
}

Tensor<float> minus(Tensor<float> const &a, Tensor<float> const &b, cublasHandle_t const cublas_handle) {
	assert(a.dims()==b.dims());
	Tensor<float> res(a.dims());
	float alpha = 1.f;
	float beta = -1.f;
	cublasStatus_t status = cublasSgeam(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
			a.size_, 1, &alpha, a.begin(), a.size_, &beta, b.begin(), b.size_, res.begin(), res.size_);
	return res;
}

float frobeniusNorm(Tensor<float> const &x, cublasHandle_t const cublas_handle) {
	float* res = (float*)malloc(sizeof(float));
	cublasStatus_t stat = cublasSnrm2(cublas_handle, x.size_, x.begin(), 1, res);
	return *res;
}


Tensor<float> fill(std::vector<int> dims, float val) {
     Tensor<float> tensor(dims);
     thrust::fill(thrust::device_ptr<float>(tensor.begin()),
                  thrust::device_ptr<float>(tensor.end()), val);
     return tensor;
}

Tensor<float> zeros(std::vector<int> dims) {
    Tensor<float> tensor(dims);
    thrust::fill(thrust::device_ptr<float>(tensor.begin()),
                 thrust::device_ptr<float>(tensor.end()), 0.f);
    return tensor;
}

Tensor<float> rand(std::vector<int> dims, curandGenerator_t curand_gen) {
    Tensor<float> tensor(dims);
    curandGenerateUniform(curand_gen, tensor.begin(), tensor.size());
    shiftAndScale<<<(tensor.size_+255)/256, 256>>>(tensor.size(),tensor.begin());
    return tensor;
}
