#include <stdio.h>
#include <stdlib.h>

/**
 * In this section, we will discover concurrent operation in CUDA
 *  1) blocks in grid: concurrent tasks, no gurantee their order of execution (no synchronization)
 *  2) warp in blocks: concurrent threads, explicitly synchronizable (it will be discussed in next section)
 *  3) thread in warp: implicitly synchronized
 */

__global__ void idx_print()
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_idx = threadIdx.x / warpSize;
    int lane_idx = threadIdx.x & (warpSize - 1); // the index of the thread in the warp
    
    //if ((lane_idx & (warpSize/2 - 1)) == 0) // print every other thread
    //    printf(" %5d\t%5d\t %2d\t%2d\n", idx, blockIdx.x, warp_idx, lane_idx); // thread, block, warp, lane
    printf(" %5d\t%5d\t %2d\t%2d\n", idx, blockIdx.x, warp_idx, lane_idx); // thread, block, warp, lane
}

//int main(int argc, char* argv[]) {
//    int gridSize = atoi(argv[1]);
//    int blockSize = atoi(argv[2]);
int main() {
    int gridSize = 4;
    int blockSize = 128;
    puts("thread, block, warp, lane");
    idx_print<<<gridSize, blockSize>>>();
    cudaDeviceSynchronize();
}
