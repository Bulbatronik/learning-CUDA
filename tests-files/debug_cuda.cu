#include <cuda_runtime.h>
#include <cublasLt.h>
#include <iostream>

int main() {
    std::cout << "=== CUDA Debugging Information ===" << std::endl;
    
    // 1. Check device count
    int deviceCount;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    std::cout << "cudaGetDeviceCount result: " << cudaGetErrorString(error) << std::endl;
    std::cout << "Device count: " << deviceCount << std::endl;
    
    if (error != cudaSuccess) {
        return 1;
    }
    
    // 2. Try to set device
    error = cudaSetDevice(0);
    std::cout << "cudaSetDevice(0) result: " << cudaGetErrorString(error) << std::endl;
    
    // 3. Try to allocate minimal memory
    void* ptr;
    error = cudaMalloc(&ptr, 4);
    std::cout << "cudaMalloc result: " << cudaGetErrorString(error) << std::endl;
    
    if (error == cudaSuccess) {
        cudaFree(ptr);
        std::cout << "Memory allocation successful!" << std::endl;
    }
    
    // 4. Try to create cuBLAS handle
    cublasLtHandle_t handle;
    cublasStatus_t status = cublasLtCreate(&handle);
    std::cout << "cublasLtCreate result: " << status << std::endl;
    
    if (status == CUBLAS_STATUS_SUCCESS) {
        std::cout << "cuBLAS handle created successfully!" << std::endl;
        cublasLtDestroy(handle);
    }
    
    return 0;
}
