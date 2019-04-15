#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

__global__ void testKernel(int val, int b)
{
    printf("[%d, %d]:\t\tValue is:%d\n",\
            blockIdx.y*gridDim.x+blockIdx.x,\
            threadIdx.z*blockDim.x*blockDim.y+threadIdx.y*blockDim.x+threadIdx.x,\
            val, b);
}

int main(int argc, char **argv)
{
    int nDev; CUcontext ctx;

    cuInit(0);
    cuDeviceGetCount(&nDev);
    assert(nDev > 0);

    cuDevicePrimaryCtxRetain(&ctx, 0);
    cuCtxSetCurrent(ctx);

    CUstream stream;
    cuStreamCreate(&stream, 0);
    CUdeviceptr dptr = 0;
    int count = 64;
    int val[64] = {};
    memset(val, 0xff, sizeof(val));
    cuMemAlloc_v2(&dptr, count * sizeof(int));
    for (int i = 0; i < count; ++i)
        cuMemsetD32_v2(dptr + i * sizeof(int), i, 1);
    assert(CUDA_SUCCESS == cuStreamSynchronize(stream));
    // cuMemcpyDtoH(val, dptr, count * sizeof(int));
    cublasGetMatrix(4, 2, sizeof(int), (void*)dptr, 8, val, 8);
    /*dim3 dimGrid(2, 2);
    dim3 dimBlock(2, 2, 2);
    testKernel<<<dimGrid, dimBlock>>>(10, 1);*/
    for (int i = 0; i < 4; ++i)
       assert(val[i] == i);
    for (int i = 4; i < 8; ++i)
       assert(val[i] == -1);
    for (int i = 8; i < 12; ++i)
       assert(val[i] == i);
    for (int i = 12; i < count; ++i)
       assert(val[i] == -1);
    puts("PASS");
    return 0;
}

