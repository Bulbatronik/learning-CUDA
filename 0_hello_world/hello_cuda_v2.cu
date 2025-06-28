#include <stdio.h>

__global__ void hello_cuda(){
	printf("Hello Cuda\n");
	printf("Block Index X:%d, Block Index Y:%d, Thread Index X:%d, Thread Index Y:%d\n",
	       blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y);
}

int main(){
	hello_cuda<<<2,3>>>(); // grid shape (num_blocks), block shape
	cudaDeviceSynchronize();
	return 0;
}
