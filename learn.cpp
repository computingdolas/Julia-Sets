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
  Complex(){ this->real = 0.0 ; this->imag = 0.0 ;}
  ~Complex() ; 
  
  // Access for the real part 
  
  const double& real() const {
    return this->real ; 
  }
  
  double & real() {
    return this->real ; 
  }
  
  // Access for the imag part 
  
  const double & imag() const {
    return this->imag ; 
  }
    
  double & imag() {
    return this->imag ; 
  }
  
  const Complex square() {
    Complex temp ; 
    
    temp.real() = (this->real() * this->real()) - (this->imag() * this->imag()) ; 
    temp.imag() = (2 * this->real() * this->imag() ) ; 
    
    this->real() = temp.real() ; 
    this->imag() = temp.imag() ; 
    
    return *(this) ; 

  }

  const double modulus() const{
      return std::sqrt((this->real * this->real) + (this->imag * this->imag)) ;
  }

  Complex operator+ (const Complex & obj) {
      Complex temp ;
      temp->real() = this->real + obj->real() ;
      temp->imag() = this->imag + obj->imag() ;

      this->real = temp->real() ;
      this->imag = temp->imag() ;
  }

  Complex& operator= (const Complex & obj) {
      this->real = obj->real() ;
      this->imag = obj->imag() ;
  }

__global__ void julia(unsigned int * _color_bit_device, long long N) {
 
  
  
  
}

int main() {
  
  const long long numberOfGridPoints_ = (1<<11 )* (1<<11) ; 
  const long long bytes_ = numberOfGridPoints_ * sizeof(unsigned int) ; 

  std::cout<<"The Total Memory in MB allocated for the program is :="<<bytes_ * 1e-6<<std::endl ;
  
  // Allocating Vector on Host 
  std::vector<unsigned int> color_bit(numberOfGridPoints_,0) ; 
  
  // Pointer on device 
  unsigned int * color_bit_device  ; 
  
  // Allocating memory on device 
  cudaMalloc(&color_bit_device,bytes_) ; 
  
  // Copying Data from host to device 
  cudaMemcpy(color_bit_device,&color_bit,bytes_,cudaMemcpyHostToDevice) ;
  
  double start = getSeconds() ; 

  return 0 ;
}


  
  
