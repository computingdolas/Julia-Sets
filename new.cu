//
//  main.cpp
//  High End simulation in Practice
//
//  Created by Sagar Dolas on 19/04/16.
//  Copyright © 2016 Sagar Dolas. All rights reserved.
//

#include <cuda_runtime.h>
#include <cstddef>
#include <sys/time.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <string>
#include "lodepng.h"

//#define GRID_SIZE = 1<<11 ; 

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

// Device Function //

__global__ void juliaImage(unsigned char * color_bit_device, long long N, const double mesh, const double threshold) {
    
    long long idx = blockIdx.x * blockDim.x + threadIdx.x ;
    double real = (mesh * (idx % N)) - 2.0;
    double imag = (mesh * (idx / N)) - 2.0 ;
    double temp_real = 0.0 ; 
    double temp_imag = 0.0 ; 
    const double c_real = -0.4 ; 
    const double c_imag = 0.6 ;
    double modulus = sqrt( (real * real) + (imag * imag) ) ;     
    unsigned int itr = 0 ; 
    
    while (modulus <= threshold) { 
	
      temp_real = (real * real) - (imag * imag) ; 
      temp_imag = (2 * real * imag) ; 
      
      real = temp_real ; 
      imag = temp_imag ; 
      
      real += c_real ;
      imag += c_imag ; 
      
      temp_real = 0.0 ; 
      temp_imag = 0.0 ; 
      
      modulus = sqrt( (real * real) + (imag * imag) ) ; 
      
      ++itr ;  
    }
    unsigned int numIter_ = itr*10 +10000;
    unsigned i = idx % N ;
    unsigned j = idx / N ;

    color_bit_device[4*N*j+4*i+3] = (numIter_  & 255);
    color_bit_device[4*N*j+4*i+2] = (numIter_ >> 8) & 255;
    color_bit_device[4*N*j+4*i+1] = (numIter_ >> 16) & 255;
    color_bit_device[4*N*j+4*i+0] = (numIter_ >> 24) & 255;

}

//Encode from raw pixels to disk with a single function call
//The image argument has width * height RGBA pixels or width * height * 4 bytes
void encodeImage(const char* filename, std::vector<unsigned char>& image, unsigned width, unsigned height)
{
  //Encode the image
  unsigned error = lodepng::encode(filename, image, width, height);
  //if there's an error, display it
  if(error) std::cout << "encoder error " << error << ": "<< lodepng_error_text(error) << std::endl;
}

int main(int argc , char *argv[]) {


    unsigned int numTHreadsPerBlocks_ = 1024;         //std::stol(argv[1]);
    unsigned numblocks_ =  2048 *2048 / numTHreadsPerBlocks_ ;

    const long long numberOfGridPoints_ = (1<<11)* (1<<11 ) ;
    const long long bytes_ = numberOfGridPoints_ * sizeof(unsigned char ) * 4 ;
    const long num = 2048 ;  
    const double mesh = 4.0 / ((1<<11 )) ;
    const double threshold = 500.0 ; 
    
    std::cout<<"the mesh is "<<mesh<<std::endl ; 
    std::cout<<"The Total Memory in MB allocated for the program is :="<<bytes_ * 1e-6<<std::endl ;
    
    // Allocating Vector on Host
    std::vector <unsigned char> colourBit(numberOfGridPoints_*4);

    // Pointer on device
    unsigned char * color_bit_device  ;
    
    // Allocating memory on device
    checkError(cudaMalloc(&color_bit_device,bytes_)) ;
     
    double start = getSeconds() ;
    juliaImage<<< numblocks_ ,numTHreadsPerBlocks_>>>(color_bit_device,num,mesh,threshold) ;
    checkError(cudaDeviceSynchronize());
    double end = getSeconds() ;
    std::cout<< (end - start) *1e3 <<std::endl ;
    
    // Copying data back to Host
    checkError(cudaMemcpy(&colourBit[0],color_bit_device,bytes_,cudaMemcpyDeviceToHost));

    // Freeing the memory on the device
    checkError(cudaFree(color_bit_device)) ;
      
    encodeImage("Julia.png", colourBit, num, num);    
    return 0 ;
}
