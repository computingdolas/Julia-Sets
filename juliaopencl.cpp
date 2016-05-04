/*Assignment 1: OpenCL code to generate Julia sets(2048*2048 pixels)*/

#include <iostream>
#include <CL/cl.hpp>
#include <vector>
#include <cassert>
#include <string>
#include "lodepng.h"

#define PIXELRES 2048

std::string ker_code = "\n"\
"__kernel void juliaImage(unsigned char *color_bit_device, long long N, const double mesh, const double theshold)\n"\
"{																				   \n"\
" 	 long long idx = get_global_id(0);											   \n"\
"    const double x = (mesh * (idx % N)) - 2.0 ;                                   \n"\
"    const double y = (mesh * (idx / N)) - 2.0 ;                                   \n"\
"    double temp_real = 0.0 ; 			       									   \n"\
"    double temp_imag = 0.0 ; 													   \n"\
"    const double c_real = -0.4 ; 												   \n"\
"    const double c_imag = 0.6 ;												   \n"\
"    double modulus = sqrt( (real * real) + (imag * imag) ) ;  					   \n"\
"   																			   \n"\
"																				   \n"\
"																				   \n"\
"    unsigned int itr = 0;													       \n"\
"    while(modulus < threshold){                                                   \n"\
"	                                                                               \n"\
"      temp_real = (real * real) - (imag * imag) ;                                 \n"\
"      temp_imag = (2 * real * imag) ;                                             \n"\
"                                                                                  \n"\
"      real = temp_real ;                                                          \n"\
"      imag = temp_imag ;                                                          \n"\
"                                                                                  \n"\
"      real += c_real ;                                                            \n"\
"      imag += c_imag ;                                                            \n"\
"                                                                                  \n"\
"      temp_real = 0.0 ;                                                           \n"\
"      temp_imag = 0.0 ; 														   \n"\
"																				   \n"\
"	   modulus = sqrt( (real * real) + (imag * imag) ) ; 						   \n"\
"	   itr++;                                                                      \n"\
"    }                                                                             \n"\  
"    														                       \n"\
"	unsigned int numIter_ = itr*10 +10000;										   \n"\
"    unsigned i = idx % N ;														   \n"\
"    unsigned j = idx / N ;														   \n"\
"																				   \n"\
"    color_bit_device[4*N*j+4*i+3] = (numIter_  & 255);							   \n"\
"    color_bit_device[4*N*j+4*i+2] = (numIter_ >> 8) & 255;						   \n"\
"    color_bit_device[4*N*j+4*i+1] = (numIter_ >> 16) & 255;					   \n"\
"    color_bit_device[4*N*j+4*i+0] = (numIter_ >> 24) & 255;					   \n"\
"                                            									   \n"\
"}                                                                                 \n";

//Encode from raw pixels to disk with a single function call
//The image argument has width * height RGBA pixels or width * height * 4 bytes
void encodeImage(const char* filename, std::vector<unsigned char>& image, unsigned width, unsigned height)
{
  //Encode the image
  unsigned error = lodepng::encode(filename, image, width, height);

  //if there's an error, display it
  if(error) std::cout << "encoder error " << error << ": "<< lodepng_error_text(error) << std::endl;
}

int main(int argc,char *argv[]){
	if(argc < 2){
		std::cerr<<"Number of arguements to program is less than expected"<<std::endl;
	}

	int items_per_group = std::stoi(argv[1]);

	std::vector<cl::Platform> platforms;
	cl::Platform def_platform;

	//Number of pixels for the picture
	long long numPixels1D = PIXELRES;
	long long numPixels = numPixels1D*numPixels1D;

	int num_groups = numPixels/items_per_group;

	const double mesh = 4.0/(static_cast<double>(PIXELRES));
	const double threshold = 500.0;

	std::vector<unsigned char> color_bit_host(numPixels*4,0);

	//Get available platforms
	cl::Platform::get(&platforms);

	//Select default platform
	if(platforms.size() == 0){
		std::cout<<"No OpenCL platforms found. Please check installation"<<std::endl;
		exit(1);
	}
	else{
		def_platform = platforms[0];
		std::cout<<"Using platform: "<<def_platform.getInfo<CL_PLATFORM_NAME>()<<std::endl;
	}

	std::vector<cl::Device> devices;
	std::vector<cl::Device> used_device;
	cl::Device def_device;
	
	//Get available devices 
	def_platform.getDevices(CL_DEVICE_TYPE_ALL,&devices);

	//Select default device
	if(devices.size() == 0){
		std::cout<<"No devices found"<<std::endl;
		exit(1);
	}
	else{
		def_device = devices[0];
		used_device.push_back(def_device);
		//std::cout<<"Using device: "<<def_device.getInfo<CL_DEVICE_NAME>()<<std::endl;
	}

	//Create context with the devices
	cl::Context context(used_device);

	cl::Program::Sources sources;
	sources.push_back({ker_code.c_str(),ker_code.length()});

	cl::Program program(context,sources);
	if(program.build(used_device)!=CL_SUCCESS){
        std::cout<<" Error building: "<<program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(def_device)<<"\n";
        exit(1);
    }

    //Create buffers on device(Equivalent to cudaMalloc)
    cl::Buffer color_bit_device(context,CL_MEM_READ_WRITE,sizeof(unsigned char)*numPixels*4);

    //Create command queue
    cl::CommandQueue queue(context,def_device);

    //Create kernel object
    cl::Kernel kernel_julia=cl::Kernel(program,"juliaImage");
    kernel_julia.setArg(0,color_bit_device);
    kernel_julia.setArg(1,numPixels1D);
    kernel_julia.setArg(2,mesh);
    kernel_julia.setArg(3,threshold);
    queue.enqueueNDRangeKernel(kernel_julia,cl::NullRange,cl::NDRange(numPixels),cl::NDRange(items_per_group));
    queue.finish();
	


    //Copy the color bits back
    queue.enqueueReadBuffer(color_bit_device,CL_TRUE,0,sizeof(unsigned char)*PIXELRES,&color_bit_host[0]);
	
	//Encode image to juliaOpenCl.png
	encodeImage("juliaOpenCl.png", color_bit_host, PIXELRES, PIXELRES); 
	return 0;
}
