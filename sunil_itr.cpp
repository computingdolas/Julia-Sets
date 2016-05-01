#include <cstddef>
#include <sys/time.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <bitset>
#include <algorithm>
#include "lodepng.h"
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
	const long long XGridPoints = (1<<11);
	long double mesh = 4.0/(XGridPoints);
	Complex c(-0.4,0.6);
	const long long numberOfGridPoints = XGridPoints * XGridPoints;
    std::vector <unsigned char> colourBit(numberOfGridPoints*4);

        std::vector <double> moduli;
	std::vector <int> iteration;
	for(int i=0; i< numberOfGridPoints; ++i)
	{
	   double x = mesh*(i%XGridPoints) - 2.0;
	   double y = mesh*(i/XGridPoints) - 2.0;
	   Complex a(x,y);
	   int j=0;
	   while (a.modulus() <=500)
	   {
	    a = a.square();
	    a = a + c;
	    ++j;
	   }
	
	moduli.push_back(a.modulus());
	iteration.push_back(j);
	}
	std::cout<<*(std::max_element(iteration.begin(), iteration.end()))<<std::endl;
    //Color mapping by modulus
   for(int j=0;j<XGridPoints;j++){
        for(int i=0;i<XGridPoints;i++){
            if(iteration[j*XGridPoints + i]>75){
                colourBit[4*XGridPoints*j+4*i+0] = 255;
                colourBit[4*XGridPoints*j+4*i+1] = 255;
                colourBit[4*XGridPoints*j+4*i+2] = 0;
                colourBit[4*XGridPoints*j+4*i+3] = 0;
            }
            else{
                colourBit[4*XGridPoints*j+4*i+0] = 0;
                colourBit[4*XGridPoints*j+4*i+1] = 0;
                colourBit[4*XGridPoints*j+4*i+2] = 255;
                colourBit[4*XGridPoints*j+4*i+3] = 255;
            }
        }
    }

    encodeImage("sunil_img_col.png", colourBit, XGridPoints, XGridPoints);

    return 0 ;
}
