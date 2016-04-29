
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
    
    Complex(const double _real, const double _imag): real(_real), imag(_imag){}
    Complex(){
        this->real = 0.0 ;
        this->imag = 0.0 ;
    }
    ~Complex(){} ;
    
    // Access for the real part
    
    const double& realpart() const {
        return this->real ;
    }
    
    double & realpart() {
        return this->real ;
    }
    
    // Access for the imag part
    
    const double & imagpart() const {
        return this->imag ;
    }
    
    double & imagpart() {
        return this->imag ;
    }
    
    const Complex square() {
        Complex temp ;
        
        temp.realpart() = (this->realpart() * this->realpart()) - (this->imagpart() * this->imagpart()) ;
        temp.imagpart() = (2 * this->realpart() * this->imagpart() ) ;
        
        this->realpart() = temp.realpart() ;
        this->imagpart() = temp.imagpart() ;
        
        return *(this) ;
        
    }
    
    const double modulus() const{
        return std::sqrt((this->real * this->real) + (this->imag * this->imag)) ;
    }
    
    Complex operator+ (const Complex & obj){
        Complex temp ;
        
        double real = this->real + obj.realpart() ;
        double imag = this->imag + obj.imagpart() ;
        
        return Complex(real,imag) ;
    }
    
    Complex& operator= (const Complex & obj){
        this->real = obj.realpart() ;
        this->imag = obj.imagpart() ;
        
        return *(this) ;
    }
};

    // Device Function //
    
    __global__ void juliaImage(unsigned int * color_bit_device, long long N, const long double mesh) {
         
	long long idx = blockIdx.x * blockDim.x + threadIdx.x ;
	const double x = (mesh * (idx % N)) - 2.0 ; 
	const double y = (mesh * (idx / N)) - 2.0 ; 
	const int numIter = 200 ; 
	const Complex c(0,-0.8) ;
	Complex z(x,y) ; 
	const int threshold = 10
	
	for(size_t i =0 ; i < numIter; ++i){
	  z = z.square() + c ; 
	}
	
	color_bit_device[idx] = z.modulus() ;
    }
    
    int main() {
        
        
        const long long numberOfGridPoints_ = (1<<11 )* (1<<11) ;
        const long long bytes_ = numberOfGridPoints_ * sizeof(unsigned int) ;
        long double mesh = 1.0 / (N-1) ;
        std::cout<<"The Total Memory in MB allocated for the program is :="<<bytes_ * 1e-6<<std::endl ;
        
        // Allocating Vector on Host
        std::vector<unsigned int> color_bit(numberOfGridPoints_,0) ;
        
        // Pointer on device
        unsigned int * color_bit_device  ;
        
        // Allocating memory on device 
        checkError(cudaMalloc(&color_bit_device,bytes_)) ; 
        
        // Copying Data from host to device 
        checkError(cudaMemcpy(color_bit_device,&color_bit[0],bytes_,cudaMemcpyHostToDevice)) ;
        
        double start = getSeconds() ; 
	juliaImage<<<(1<<12),(1<<10)>>>(color_bit_device,numberOfGridPoints_,mesh) ; 
	double end = getSeconds() ; 
	
	// Copying data back to Host
	checkError(cudaMemcpy(color_bit[0],&color_bit_device,cudaMemcpyDeviceToHost)) ; 
	
	// Freeing the memory on the device 
	checkError(cudaFree(color_bit_device)) ; 
	
        return 0 ;
    }