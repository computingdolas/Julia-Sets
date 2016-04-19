#include <cuda_runtime.h>

#include <cstddef>
#include <sys/time.h>
#include <iostream>
#include <vector>

void checkError (cudaError_t err)
{
    if(err != cudaSuccess )
    {
        std::cout<< cudaGetErrorString(err) <<std::endl ;
        exit(-1);
    }
}

double getSeconds()
{
    struct timeval tp ;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1e-6) ;
}

__global__ void sum(int *A , int *B, int *C, long long N)
{
    long long idx = blockIdx.x * blockDim.x + threadIdx.x ;
    if(idx  < N)
    {
        C[idx] = A[idx] + B[idx] ;
    }
}

int main() {

    const long long nElem = 1<<20 ;

    std::vector<int> A(nElem,1);
    std::vector<int> B(nElem,1);
    std::vector<int> C(nElem,0);

    const long long  nBytes = nElem * sizeof(int); // Negative memory
    std::cout<< nBytes * 1e-6 <<std::endl ;
    int * d_A ;
    int * d_B ;
    int * d_C ;

    cudaMalloc(&d_A , nBytes);
    cudaMalloc(&d_B , nBytes);
    cudaMalloc(&d_C , nBytes);

    cudaMemcpy(d_A, &A[0],nBytes,cudaMemcpyHostToDevice );
    cudaMemcpy(d_B, &B[0],nBytes,cudaMemcpyHostToDevice );

    double start = getSeconds();
    sum <<< (1<< 11) ,( 1<<9) >>> (d_A, d_B, d_C, nElem);
    checkError(cudaDeviceSynchronize());
    double stop = getSeconds();

    std::cout<< (stop - start) *1e3 <<std::endl ;
    cudaMemcpy(&C[0], d_A,nBytes,cudaMemcpyDeviceToHost );


    for(int i = 0 ; i < nElem ; ++i)
    {
        if(C.at(i) !=1 )
        {
            std::cout<<"error"<<i<< " "
                    << C.at(i)<<std::endl ;
            exit(-1);
        }
    }

    checkError(cudaFree(d_A));
    checkError(cudaFree(d_B));
    checkError( cudaFree(d_C));

    // The warning can be atrributed to the fact that we need to specify the architecture for which nvidia ..

    return 0 ;
}
