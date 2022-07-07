#include <stdio.h>
#include <math.h>
#include <malloc.h>
#include <time.h>
#include <chrono>

#include <cuda.h>
#include <cufft.h>
#include <cufftXt.h>

#define SIN sin
#define COS cos

typedef double  Real;
typedef double2 Complex;

const unsigned int N = 4;

typedef struct node{
    cufftHandle plan;
    Complex *twid;
    int n;
    int sign;
    int DataSize;
    int MemSize;
}ffth_plan;

#define checkCudaErrors(err) __checkCudaErrors(err, __FILE__, __LINE__)

inline void __checkCudaErrors(cudaError err, const char *file, const int line) {
  if (cudaSuccess != err) {
    fprintf(stderr, "%s(%i) : CUDA Runtime API error %d: %s.\n", file, line,
            (int)err, cudaGetErrorString(err));
    exit(1);
  }
}


__global__ void twid_fac_2d(Complex* twid, int n, int sign){
    int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
    Real x, fac = 6.2831853071795864769252867665590057683943387987502/((Real)(3*n));

    if(xIndex < 3*n){
        x = (xIndex-n+1) * fac;
        twid[xIndex].x = COS(x);
        twid[xIndex].y = SIN(x) * (-sign);
    }
}

__global__ void Radix_3_fft(Complex *input, Complex *twid, int n){
    int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
    int yIndex = blockIdx.y * blockDim.y + threadIdx.y;
    if(xIndex < n && yIndex < n){
        int a = yIndex * n + xIndex;
        int b = a + n * n;
        int c = b + n * n;

        Complex tmpa, tmpb, tmpc;
        tmpa = input[a];
        tmpb = input[b];
        tmpc = input[c];

        int s ,t;
        s = yIndex - xIndex + n - 1;
        t = n - (yIndex - xIndex) + n - 1;

        Complex tmp;
        tmp.x = tmpa.x + tmpc.x;
        tmp.y = tmpa.y + tmpc.y;

        tmpa.x -= tmpc.x;
        tmpa.y -= tmpc.y;
        tmpc.x -= tmpb.x;
        tmpc.y -= tmpb.y;
        input[b].x = tmpb.x + tmp.x;
        input[b].y = tmpb.y + tmp.y;

        tmp.x = -twid[t].y * tmpa.y - twid[s].y * tmpc.y;
        tmp.y =  twid[t].y * tmpa.x + twid[s].y * tmpc.x;
        tmpc.x = twid[t].x * tmpa.x - twid[s].x * tmpc.x;
        tmpc.y = twid[t].x * tmpa.y - twid[s].x * tmpc.y;
        input[a].x = tmpc.x - tmp.x;
        input[a].y = tmpc.y - tmp.y;
        input[c].x = tmpc.x + tmp.x;
        input[c].y = tmpc.y + tmp.y;
    }
    return ;
}

__device__ void sorting(void* dataout,
                        size_t offset,
                        cufftDoubleComplex element,
                        void* callerInfo,
                        void* sharePtr){
    size_t xIndex = offset % N;
    size_t yIndex = offset / N;

    int n = yIndex / N - 1;
    int m1 = xIndex;
    int m2 = yIndex % N;
    int p1 = 2*m1 - m2 - n;
    int p2 = 2*m2 - m1 + n;

    if(p1 < 0){
        p1 += N;
        p2 += N;
    }
    else if(p1 >= N){
        p1 -= N;
        p2 += 2*N;
    }
    p2 = (p2+N)%(3*N);

    ((cufftDoubleComplex*)dataout)[p2*N+p1].x = element.x;
    ((cufftDoubleComplex*)dataout)[p2*N+p1].y = element.y;
    return ;
}

__device__ cufftCallbackStoreZ d_storeCallbackPtr = sorting;

void ffthPlan(ffth_plan *ffthplan, int N, int sign){
    ffthplan->N = N;
    ffthplan->DataSize = 3 * N * N;
    ffthplan->MemSize = 3 * N * N * sizeof(Complex);
    int length[2] = {N, N};
    int batch = 3;

    cufftPlanMany(&(ffthplan->plan), 2, length, NULL, 1, N*N, NULL);
    cufftCallbackStoreZ h_storeCallbackPtr;
    cudaMemcpyFromSymbol(&h_storeCallbackPtr, d_storeCallbackPtr, sizeof(h_storeCallbackPtr));
    cufftXtSetCallback(plan, (void**)&h_storeCallbackPtr, CUFFT_CB_ST_COMPLEX_DOUBLE, 0);

    dim3 dimblockt(768);
    dim3 dimgridt((3*N+dimblockt.x-1)/dimblockt.x);

    Complex* twiddles;
    cudaMalloc((void**)&twiddles, 3*N*sizeof(Complex));
    twid_fac_2d<<<dimgridt, dimblockt>>>(twiddles, N, sign);

    return ;
}

void ffthDestroy(ffth_plan *ffthplan){
    cufftDestroy(ffthplan->plan);
    cudaFree(plan->twid);
    free(ffthplan);
    return ;
}

int HexagonsFFT_run(ffth_plan* ffthplan, Complex* input, Complex* output, int sign){
    
    int DataSize = ffthplan->DataSize;
    int MemSize = ffthplan->MemSize;
    Complex* d_input;
    Complex* d_output;
    cudaMalloc((void**)&d_input, MemSize);
    cudaMalloc((void**)&d_output, MemSize);
    cudaMemcpy(d_input, input, MemSize, cudaMemcpyHostToDevice);

    cufftHandle plan;
    int length[2] = {N, N};
    cufftPlanMany(&plan, 2, length, NULL, 1, N*N, NULL, 1, N*N, CUFFT_Z2Z, 3);

    cufftCallbackStoreZ h_storeCallbackPtr;
    cudaMemcpyFromSymbol(&h_storeCallbackPtr, d_storeCallbackPtr, sizeof(h_storeCallbackPtr));
    cufftXtSetCallback(plan, (void**)&h_storeCallbackPtr, CUFFT_CB_ST_COMPLEX_DOUBLE, 0);

    dim3 dimblock3(48, 16);
    dim3 dimgrid3((N+dimblock3.x - 1)/dimblock3.x, (N+dimblock3.y - 1)/dimblock3.y);
    dim3 dimblockt(768);
    dim3 dimgridt((3*N+dimblockt.x-1)/dimblockt.x);

    Complex* twiddles;
    cudaMalloc((void**)&twiddles, 3*N*sizeof(Complex));
    twid_fac_2d<<<dimgridt, dimblockt>>>(twiddles, N, sign);

    Radix_3_fft<<<dimgrid3, dimblock3>>>(d_input, twiddles, N);

    cufftExecZ2Z(plan, d_input, d_output, sign);

    cudaMemcpy(output, d_output, MemSize, cudaMemcpyDeviceToHost);
    cufftDestroy(plan);
    cudaFree(twiddles);
    cudaFree(d_input);
    cudaFree(d_output);
    return 0;
}

int main(){
    Complex* input;
    Complex* output;

    cudaHostAlloc((void **)&input, MemSize, cudaHostAllocDefault);
    cudaHostAlloc((void **)&output, MemSize, cudaHostAllocDefault);

    for(int i=0; i<3*N; i++){
        for(int j=0; j<N; j++){
            input[i*N + j].x = (Real)(i*N + j);
            input[i*N + j].y = (Real)(i*N + j);
        }
    }

    HexagonsFFT_run(input, output, N, -1);

    for(int i=0; i<3*N; i++){
        for(int j=0; j<N; j++){
            printf("(%2d, %3d), (%20.16f + i*%20.16f)\n", i, j, output[i*N+j].x, output[i*N+j].y);
        }
    }
    cudaFreeHost(input);
    cudaFreeHost(output);

    return 0;
}