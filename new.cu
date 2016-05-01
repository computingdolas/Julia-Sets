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

class Complex {
    
    private :
    
    double real ;
    double imag ;
    
public:
    
    __host__ __device__ Complex(const double _real, const double _imag): real(_real), imag(_imag){}
    __host__ __device__ Complex(){
        this->real = 0.0 ;
        this->imag = 0.0 ;
    }
    __host__ __device__ ~Complex(){} ;
    
    // Access for the real part
    
    __host__ __device__ const double& realpart() const {
        return this->real ;
    }
    
    __host__ __device__ double & realpart() {
        return this->real ;
    }
    
    // Access for the imag part
    
    __host__ __device__ const double & imagpart() const {
        return this->imag ;
    }
    
    __host__ __device__ double & imagpart() {
        return this->imag ;
    }
    
    __host__ __device__ const Complex square() {
        Complex temp ;
        
        temp.realpart() = (this->realpart() * this->realpart()) - (this->imagpart() * this->imagpart()) ;
        temp.imagpart() = (2 * this->realpart() * this->imagpart() ) ;
        
        this->realpart() = temp.realpart() ;
        this->imagpart() = temp.imagpart() ;
        
        return *(this) ;
        
    }
    
    __host__ __device__ const double  modulus() const{
        return std::sqrt((this->real * this->real) + (this->imag * this->imag)) ;
    }
    
    __host__ __device__ Complex operator+ (const Complex & obj){
        Complex temp ;
        
        double real = this->real + obj.realpart() ;
        double imag = this->imag + obj.imagpart() ;
        
        return Complex(real,imag) ;
    }
    
    __host__ __device__ Complex& operator= (const Complex & obj){
        this->real = obj.realpart() ;
        this->imag = obj.imagpart() ;
        
        return *(this) ;
    }
};

// Device Function //

__global__ void juliaImage(unsigned int * color_bit_device, long long N, const double mesh, const double threshold) {
    
    long long idx = blockIdx.x * blockDim.x + threadIdx.x ;
    double real = (mesh * (idx % N)) - 2.0;
    double imag = (mesh * (idx / N)) - 2.0 ;
    const int numIter = 25 ;
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
    color_bit_device[idx] = itr ; 
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


int main() {
    
    
    const long long numberOfGridPoints_ = (1<<11)* (1<<11 ) ;
    const long long bytes_ = numberOfGridPoints_ * sizeof(unsigned int ) ;
    const long num = 2048 ;  
    const double mesh = 4.0 / ((1<<11 )) ;
    const double threshold = 500.0 ; 
    
    std::cout<<"the mesh is "<<mesh<<std::endl ; 
    std::cout<<"The Total Memory in MB allocated for the program is :="<<bytes_ * 1e-6<<std::endl ;
    
    // Allocating Vector on Host
    std::vector<unsigned int> color_bit(numberOfGridPoints_,0) ;
    std::vector <unsigned char> colourBit(numberOfGridPoints_*4);

    
    // Pointer on device
    unsigned int * color_bit_device  ;
    
    // Allocating memory on device
    checkError(cudaMalloc(&color_bit_device,bytes_)) ;
    
    // Copying Data from host to device
    checkError(cudaMemcpy(color_bit_device,&color_bit[0],bytes_,cudaMemcpyHostToDevice)) ;
     
    double start = getSeconds() ;
    juliaImage<<<(1<<14),(1<<8)>>>(color_bit_device,num,mesh,threshold) ;
    checkError(cudaDeviceSynchronize());
    double end = getSeconds() ;
    std::cout<< (end - start) *1e3 <<std::endl ;
    
    // Copying data back to Host
    checkError(cudaMemcpy(&color_bit[0],color_bit_device,bytes_,cudaMemcpyDeviceToHost));
    
    /*
    for(unsigned int i =0 ; i < numberOfGridPoints_ ; ++i ){
	
      // checking if the value is edited or not 
      std::cout<<color_bit[i]<<std::endl ;
      
    }
    */
    // Freeing the memory on the device
    checkError(cudaFree(color_bit_device)) ;

    unsigned int num_iter = 0;
    //Generation of Image
      for(int j=0;j<num;j++){
        for(int i=0;i<num;i++){
	  num_iter = color_bit[j*num+i]*10+100000;
           /* if(color_bit[j*num+i] > 10.0){
       
                colourBit[4*num*j+4*i+0] = 255;
                colourBit[4*num*j+4*i+1] = 0;
                colourBit[4*num*j+4*i+2] = 0;
                colourBit[4*num*j+4*i+3] = 0;
            }
            else{
      
                colourBit[4*num*j+4*i+0] = 0;
                colourBit[4*num*j+4*i+1] = 0;
                colourBit[4*num*j+4*i+2] = 0;
                colourBit[4*num*j+4*i+3] = 255;
            }*/
		colourBit[4*num*j+4*i+3] = (num_iter  & 255);
                colourBit[4*num*j+4*i+2] = (num_iter >> 8) & 255;
                colourBit[4*num*j+4*i+1] = (num_iter >> 16) & 255;
                colourBit[4*num*j+4*i+0] = (num_iter >> 24) & 255;
        }
    }
      
    encodeImage("Julia.png", colourBit, num, num);    
    return 0 ;
}
