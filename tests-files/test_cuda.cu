#include <cuda_runtime.h>
#include <iostream>

int main() {
    int deviceCount;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    
    if (error != cudaSuccess) {
        std::cout << "CUDA error: " << cudaGetErrorString(error) << std::endl;
        return 1;
    }
    
    std::cout << "Found " << deviceCount << " CUDA device(s)" << std::endl;
    
    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        std::cout << "Device " << i << ": " << prop.name << std::endl;
    }
    
    return 0;
}
