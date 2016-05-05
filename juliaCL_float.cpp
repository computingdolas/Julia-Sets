/*Assignment 1: OpenCL code to generate Julia sets(2048*2048 pixels)*/
#define __CL_ENABLE_EXCEPTIONS
#define CL_USE_DEPRECATED_OPENCL_1_1_APIS

#include <iostream>
#include <CL/cl.hpp>
#include <sys/time.h>
#include <vector>
#include <cassert>
#include <string>
#include "lodepng.h"

#define PIXELRES 2048

//time function
double getSeconds()
{
    struct timeval tp ;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1e-6) ;
}

std::string ker_code = "\n"\
"__kernel void juliaImage(__global unsigned char *color_bit_device, __global unsigned int *N, __global float *mesh, __global float *threshold)\n"\
"{																				   \n"\
" 	 unsigned int idx = get_global_id(0);											   \n"\
"    float real = (mesh[0] * (idx % N[0])) - 2.0 ;                                   \n"\
"    float imag = (mesh[0] * (idx / N[0])) - 2.0 ;                                   \n"\
"    float temp_real = 0.0 ; 			       									   \n"\
"    float temp_imag = 0.0 ; 													   \n"\
"    const float c_real = -0.4 ; 												   \n"\
"    const float c_imag = 0.6 ;												   \n"\
"    float modulus = sqrt( (real * real) + (imag * imag) ) ;  					   \n"\
"   																			   \n"\
"																				   \n"\
"																				   \n"\
"    unsigned int itr = 0;													       \n"\
"    while(modulus < threshold[0]){                                                   \n"\
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
"	unsigned int numIter_ = itr*10 +10000;									   \n"\
"    unsigned i = idx % N[0] ;													   \n"\
"    unsigned j = idx / N[0] ;													   \n"\
"																				   \n"\
"    color_bit_device[4*N[0]*j+4*i+3] = (numIter_  & 255);							   \n"\
"    color_bit_device[4*N[0]*j+4*i+2] = (numIter_ >> 8) & 255;						   \n"\
"    color_bit_device[4*N[0]*j+4*i+1] = (numIter_ >> 16) & 255;					   \n"\
"    color_bit_device[4*N[0]*j+4*i+0] = (numIter_ >> 24) & 255;					   \n"\
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
    unsigned int numPixels1D = PIXELRES;
    unsigned int numPixels = numPixels1D*numPixels1D;

    int num_groups = numPixels/items_per_group;

    const float mesh = 4.0/(static_cast<float>(PIXELRES));
    const float threshold = 500.0;

    std::vector<unsigned char> color_bit_host(numPixels*4,55);

    //Get available platforms
    cl::Platform::get(&platforms);

    //Select default platform
    if(platforms.size() == 0){
        std::cout<<"No OpenCL platforms found. Please check installation"<<std::endl;
        exit(1);
    }
    else{
        def_platform = platforms[1];
     //   std::cout<<"Using platform: "<<def_platform.getInfo<CL_PLATFORM_NAME>()<<std::endl;
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
    cl::Buffer numPixels1D_device(context,CL_MEM_READ_WRITE,sizeof(unsigned int));
    cl::Buffer mesh_device(context,CL_MEM_READ_WRITE,sizeof(float));
    cl::Buffer threshold_device(context,CL_MEM_READ_WRITE,sizeof(float));

    //Create command queue
    cl::CommandQueue queue(context,def_device);

    //Copying data to the buffers
    queue.enqueueWriteBuffer(mesh_device,CL_TRUE,0,sizeof(float),&mesh);
    queue.enqueueWriteBuffer(numPixels1D_device,CL_TRUE,0,sizeof(unsigned int),&numPixels1D);
    queue.enqueueWriteBuffer(threshold_device,CL_TRUE,0,sizeof(float),&threshold);

    //Create kernel object
    cl::Kernel kernel_julia=cl::Kernel(program,"juliaImage");
    kernel_julia.setArg(0,color_bit_device);
    kernel_julia.setArg(1,numPixels1D_device);
    kernel_julia.setArg(2,mesh_device);
    kernel_julia.setArg(3,threshold_device);
   
    double start = getSeconds();
    cl::Event evt;
    queue.enqueueNDRangeKernel(kernel_julia,cl::NullRange,cl::NDRange(numPixels),cl::NDRange(items_per_group),NULL,&evt);
    evt.wait();
    double end = getSeconds() ;
    std::cout<<"The total time for computation is:="<< (end - start) *1e3 <<"milliseconds"<<std::endl ;

    //Copy the color bits back to host
    queue.enqueueReadBuffer(color_bit_device,CL_TRUE,0,sizeof(unsigned char)*numPixels*4,&color_bit_host[0]);

    queue.finish();

    //Encode image to juliaOpenCl.png
    encodeImage("juliaOpenCl.png", color_bit_host, PIXELRES, PIXELRES);
    return 0;
}

