#include <cuda_runtime.h>
#include <stdio.h>
#include <time.h>

// Function to measure execution time
double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// Million threads (total)
#define NUM_THREADS 1000 // per block
#define NUM_BLOCKS 1000 // blocks in the grid

// Kernel without atomics (incorrect)
__global__ void incrementCounterNonAtomic(int* counter) {
    // not locked (you are supposed to)
    int old = *counter; // value from dereference
    int new_value = old + 1;
    // not unlocked (you are supposed to)
    *counter = new_value;
}

// An atomic operation ensures that a particular operation on a memory location is 
// completed entirely by one thread before another thread can access or modify the 
// same memory location. This prevents race conditions.

// Kernel with atomics (correct)
__global__ void incrementCounterAtomic(int* counter) {
    int a = atomicAdd(counter, 1);
}

int main() {
    int h_counterNonAtomic = 0, h_counterAtomic = 0; // Initialize counters to zero
    int *d_counterNonAtomic, *d_counterAtomic; // Device pointers for counters

    // Allocate device memory
    cudaMalloc((void**)&d_counterNonAtomic, sizeof(int));
    cudaMalloc((void**)&d_counterAtomic, sizeof(int));

    // Copy initial counter values to device
    cudaMemcpy(d_counterNonAtomic, &h_counterNonAtomic, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_counterAtomic, &h_counterAtomic, sizeof(int), cudaMemcpyHostToDevice);

    // Warm-up runs, because CUDA kernels can take time to launch and initialize
    // Launch kernels
    printf("Performing warm-up runs...\n");
    incrementCounterNonAtomic<<<NUM_BLOCKS, NUM_THREADS>>>(d_counterNonAtomic); // Million threads will try to update the same counter
    incrementCounterAtomic<<<NUM_BLOCKS, NUM_THREADS>>>(d_counterAtomic);

    // Copy results back to host
    cudaMemcpy(&h_counterNonAtomic, d_counterNonAtomic, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_counterAtomic, d_counterAtomic, sizeof(int), cudaMemcpyDeviceToHost);

    // Print results
    printf("Non-atomic counter value: %d\n", h_counterNonAtomic);
    printf("Atomic counter value: %d\n", h_counterAtomic);

    // Benchmark non-atomic implementation
    printf("Benchmarking non-atomic implementation...\n");
    double non_atom_total_time = 0.0;
    for (int i = 0; i < 20; i++) {
        // Reset counter to zero for each run
        h_counterNonAtomic = 0;
        cudaMemcpy(d_counterNonAtomic, &h_counterNonAtomic, sizeof(int), cudaMemcpyHostToDevice);
        
        double start_time = get_time();
        incrementCounterNonAtomic<<<NUM_BLOCKS, NUM_THREADS>>>(d_counterNonAtomic);
        cudaDeviceSynchronize(); // Wait for the kernel to finish
        double end_time = get_time();
        non_atom_total_time += (end_time - start_time);

        // Copy results back to host
        // cudaMemcpy(&h_counterNonAtomic, d_counterNonAtomic, sizeof(int), cudaMemcpyDeviceToHost);    
        // Print results
        // printf("Non-atomic counter value: %d\n", h_counterNonAtomic);
    }
    double non_atom_avg_time = non_atom_total_time / 20.0;
    printf("Average time for non-atomic implementation: %.6f seconds\n", non_atom_avg_time);

    // Benchmark atomic implementation
    printf("Benchmarking atomic implementation...\n");
    double atom_total_time = 0.0;
    for (int i = 0; i < 20; i++) {
        // Reset counter to zero for each run
        h_counterAtomic = 0;
        cudaMemcpy(d_counterAtomic, &h_counterAtomic, sizeof(int), cudaMemcpyHostToDevice);
        
        double start_time = get_time();
        incrementCounterAtomic<<<NUM_BLOCKS, NUM_THREADS>>>(d_counterAtomic);
        cudaDeviceSynchronize(); // Wait for the kernel to finish
        double end_time = get_time();
        atom_total_time += (end_time - start_time);

        // Copy results back to host
        // cudaMemcpy(&h_counterAtomic, d_counterAtomic, sizeof(int), cudaMemcpyDeviceToHost);
        // Print results
        // printf("Atomic counter value: %d\n", h_counterAtomic);
    }
    double atom_avg_time = atom_total_time / 20.0;
    printf("Average time for atomic implementation: %.6f seconds\n", atom_avg_time);

    // Free device memory
    cudaFree(d_counterNonAtomic);
    cudaFree(d_counterAtomic);

    return 0;
}