#include <cuda_runtime.h>
#include <cstddef>
#include <sys/time.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <bitset>
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

    int main() {
	const long long XGridPoints = (1<<11);
	long double mesh = 1/(XGridPoints-1);
	Complex c(0,-0.8);
	const long long numberOfGridPoints = XGridPoints * XGridPoints;
        std::vector <std::bitset<8>> colourBit(numberOfGridPoints);
	std::vector <int> iterations;
	for(int i=0; i< numberOfGridPoints; ++i)
	{
	   double x = mesh*(i%XGridPoints) - 2.0;
	   double y = mesh*(i/XGridPoints) - 2.0;
	   Complex a(x,y);
	   int j=0;
	   while (a.modulus()<10)
	   {
	    a = a.square();
	    a = a + c;
	    ++j;
	   }
	   iteration.pushback(j);
	}


        return 0 ;
    }
:q
