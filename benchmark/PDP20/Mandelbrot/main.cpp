#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <chrono>

#include "common.h"
#include "clDriver.h"

#include "CImg.h"
using namespace cimg_library;
#undef min
#undef max
namespace {

struct mandelbrot_param_t {
    double xmin;
    double xmax;
    double ymin;
    double ymax;
    unsigned int maxiter;
};

template<class T, class A>
std::size_t bytes_in_vector(std::vector<T, A>const& v) {
    return v.size() * sizeof(T);
}

template<class A>
std::size_t bytes_in_vector(std::vector<bool, A> const&) = delete;

}

int main(int argc,const char** argv)
{
	if (argc != 2)
	{
		std::cerr << "Usage:\t" << "./app offload(float <0.0f:1.0f>)" << std::endl;
		return -1;
	}

    const float offload_factor = std::atof(argv[1]);
    const int image_width = 1024;
    const int image_height = 1024;
    const int image_channels = 3;
    constexpr int image_size = image_width*image_height*image_channels;

    const std::string kernel_func =
            R"(
            #pragma cl_khr_fp64 : enable
            typedef struct  mandelbrot_param{
            double xmin;
            double xmax;
            double ymin;
            double ymax;
            unsigned int maxiter;
            }mandelbrot_param_t;

            kernel void mandelbrot(
            mandelbrot_param_t param,
            global unsigned char* buffer,
            constant unsigned char* red,
            constant unsigned char* green,
            constant unsigned char* blue,
            const int width,
            const int height)
            {
                const int xx = get_global_id(0);
                const int yy = get_global_id(1);

                const int size = width*height;
                unsigned int iteration = 0;
                const double x = param.xmin + xx*(param.xmax-param.xmin)/(double)width;
                const double y = param.ymin + yy*(param.ymax-param.ymin)/(double)height;

                double zr, zi, cr, ci;
                zr = 0.; zi = 0.; cr = x; ci = y;

                for (iteration=1; zr*zr + zi*zi<=4 && iteration<=param.maxiter; ++iteration)
                {
                    const double temp = zr*zr - zi*zi + cr;
                    zi = 2*zr*zi + ci;
                    zr = temp;
                }

                if (iteration>param.maxiter)
                {
                    buffer[yy*width+xx] = 0;
                    buffer[yy*width+xx+size] = 0;
                    buffer[yy*width+xx+2*size] = 0;
                }
                else
                {
                    buffer[yy*width+xx] = red[iteration];
                    buffer[yy*width+xx+size] = green[iteration];
                    buffer[yy*width+xx+2*size] = blue[iteration];
                }
            })";

    std::cout << "Execute ... " << std::endl;
    coopcl::virtual_device device;

    std::vector<unsigned char> image(image_size, 0);

    const CImg<unsigned char> colormap =
            CImg<unsigned char>(256, 1, 1, 3, 16 + 120).noise(119, 1).resize(1024, 1, 1, 3, 3).fillC(0, 0, 0, 0, 0, 0);

    CImg<unsigned char> palette_1;
    palette_1.assign(colormap._data, colormap.size() / colormap._spectrum, 1, 1, colormap._spectrum, true);
    palette_1.print();

    std::vector<unsigned char> red(1024);
    std::vector<unsigned char> green(1024);
    std::vector<unsigned char> blue(1024);

    for (int i = 0;i < 1024;++i)
    {
        red[i] = palette_1(i, 0);
        green[i] = palette_1(i, 1);
        blue[i] = palette_1(i, 2);
    }

    auto image_buffer=device.alloc(image,false);
    auto red_buffer = device.alloc(red, true);
    auto green_buffer = device.alloc(green, true);
    auto blue_buffer = device.alloc(blue, true);

    mandelbrot_param_t parameters = { 0.375 - 1.5,0.375 + 1.5,-0.2166 - 1.1,-0.2166 + 1.9,64 };

    int err = 0;
	coopcl::clTask mtask;
	device.build_task(mtask, kernel_func, "mandelbrot");

    double fps = 0.;
    int frames = 0;
    double accumulated_time = 0.;

    CImg<unsigned char> img(image.data(), image_width, image_height, 1, image_channels, true);
    CImgDisplay disp(img);

    while (!(disp.is_closed() || disp.is_keyQ() || disp.is_keyESC()))
    {
        auto start = std::chrono::high_resolution_clock::now();

        err = device.execute(   mtask,  offload_factor,
                                {image_width, image_height, 1},
                                { 16,16,1 },
                                parameters,
                                image_buffer,
                                red_buffer,
                                green_buffer,
                                blue_buffer,
                                image_width,
                                image_height);

        on_error(err);

        auto b = (const unsigned char*)image_buffer->data();

        auto end = std::chrono::high_resolution_clock::now();

        for (size_t i = 0;i < image_size;i++)
            img[i] = b[i];

        ++frames;
        accumulated_time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        if (frames == 50)
        {
            accumulated_time /= 1.e9;
            fps = frames / accumulated_time;
            std::cout<<"Offload: "<< offload_factor<<" average FPS of 50 frames: "<<fps<<std::endl;
			std::cout << "Offload: " << offload_factor << " time of 50 frames: " << accumulated_time <<" sec"<< std::endl;
			accumulated_time = 0.;
			frames = 0;
			return 0;
        }

        //profile(event);
        disp.set_title("[#7] - %s Set : (%g,%g)-(%g,%g), (%u iter.) fps=%g",
                       "Mandelbrot", parameters.xmin, parameters.ymin,
                       parameters.xmax, parameters.ymax,
                       parameters.maxiter, fps);
        disp.display(img);

        const double
                xc = 0.5*(parameters.xmin + parameters.xmax),
                yc = 0.5*(parameters.ymin + parameters.ymax),
                dx = (parameters.xmax - parameters.xmin)*0.99 / 2,
                dy = (parameters.ymax - parameters.ymin)*0.99 / 2;
        parameters.xmin = xc - dx; parameters.ymin = yc - dy; parameters.xmax = xc + dx; parameters.ymax = yc + dy;


        // Do a simple test to check if more/less iterations are necessary for the next step.
        const float value = (float)img.get_norm().get_histogram(256, 0, 255)(0) * 3;
        // std::cout << value << std::endl;
        if (value > img.size() / 6.0f) parameters.maxiter += 16;
        if (parameters.maxiter > 1023) parameters.maxiter = 1023;
        if (value < img.size() / 10.0f) parameters.maxiter -= 4;
        if (parameters.maxiter < 32) parameters.maxiter = 32;

    }

    return 0;
}

