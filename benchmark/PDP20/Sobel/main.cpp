#include "common.h"
#include "clDriver.h"
#include "assert.h"

constexpr auto sobel_filter = R"(
 __kernel void sobel_filter(
const __global uchar* inputImage, 
__global uchar* outputImage)
{
	const uint x = get_global_id(0);
    const uint y = get_global_id(1);

	const uint width = get_global_size(0);
	const uint height = get_global_size(1);

	float Gx = (float)(0);
	float Gy = Gx;
	
	const int c = x + y * width;

	if( x >= 1 && x < (width-1) && y >= 1 && y < height - 1)
	{
		float i00 = convert_float(inputImage[c - 1 - width]);
		float i10 = convert_float(inputImage[c - width]);
		float i20 = convert_float(inputImage[c + 1 - width]);
		float i01 = convert_float(inputImage[c - 1]);
		float i11 = convert_float(inputImage[c]);
		float i21 = convert_float(inputImage[c + 1]);
		float i02 = convert_float(inputImage[c - 1 + width]);
		float i12 = convert_float(inputImage[c + width]);
		float i22 = convert_float(inputImage[c + 1 + width]);

		Gx =   i00 + (float)(2) * i10 + i20 - i02  - (float)(2) * i12 - i22;
		Gy =   i00 - i20  + (float)(2)*i01 - (float)(2)*i21 + i02  -  i22;		
		
		outputImage[c] = convert_uchar(hypot(Gx, Gy)/(float)(2));
	}			
}
)";

static void ref_gold(
	const std::array<size_t, 3> gs,
	const std::array<size_t, 3> ls,
	std::vector<cl_uchar>& inputImage,
	std::vector<cl_uchar>& outputImage,
	const bool verbose)
{
	const auto count_gx = gs[0] / ls[0];
	const auto count_gy = gs[1] / ls[1];
	const auto count_gz = gs[2] / ls[2];

	if (verbose) {
		std::cout << "------------------------------" << std::endl;
		std::cout << "calculate gold/reference..." << std::endl;
	}

	const int width = gs[0];
	const int height = gs[1];

	for (auto gz = 0; gz < count_gz; gz++)
	{
		#pragma omp parallel for
		for (auto gy = 0; gy < count_gy; gy++)
		{
			for (auto gx = 0; gx < count_gx; gx++)
			{
				for (auto lz = 0; lz < ls[2]; lz++)
				{
					for (auto ly = 0; ly < ls[1]; ly++)
					{
						for (auto lx = 0; lx < ls[0]; lx++)
						{
							//kernel_func							
							int x = gx*ls[0] + lx;
							int y = gy*ls[1] + ly;

							float Gx = 0;
							float Gy = Gx;

							int c = x + y * width;

							if (x >= 1 && x < (width - 1) && y >= 1 && y < height - 1)
							{
								auto i00 = (float)(inputImage[c - 1 - width]);
								auto i10 = (float)(inputImage[c - width]);
								auto i20 = (float)(inputImage[c + 1 - width]);
								auto i01 = (float)(inputImage[c - 1]);
								auto i11 = (float)(inputImage[c]);
								auto i21 = (float)(inputImage[c + 1]);
								auto i02 = (float)(inputImage[c - 1 + width]);
								auto i12 = (float)(inputImage[c + width]);
								auto i22 = (float)(inputImage[c + 1 + width]);

								Gx = i00 + 2.0*i10 + i20 - i02 - 2.0*i12 - i22;
								Gy = i00 - i20 + 2.0*i01 - 2.0*i21 + i02 - i22;

								outputImage[c] = (cl_uchar)(hypot(Gx, Gy) / 2.0);

							}
						}
					}
				}
			}
		}
	}
}

using namespace coopcl;

int main(int argc,char** argv)
{
	if (argc != 2)
	{
		std::cerr << "Usage:\t" << "./app.exe <int>array_size>0\n";
		std::cerr << "Example:\t" << "./app.exe 1024\n";
		return -1;
	}
	const bool verbose = false;

	auto width = std::atoi(argv[1]);
	auto height = width;
	const auto step = 0.1f;
	int err = 0;

	virtual_device device;

	std::vector<float> offloads;
	generate_offload_range(offloads, step);

	constexpr auto iter = 10;

	const int M = width;
	const int N = height;

	std::vector<cl_uchar> va(M*N, 1);// init_random(va);
	std::vector<cl_uchar> vb(M*N, 0);	
	std::vector<cl_uchar> vb_ref(M*N, 0);

	auto A = device.alloc(va, true);
	auto B = device.alloc(vb, false);

	//only single channel for host/reference 
	ref_gold({ (size_t)M,(size_t)N,1 }, { 1,1,1 }, va, vb_ref,verbose);

	coopcl::clTask task;
	device.build_task(task, sobel_filter, "sobel_filter");
	
	header();

	for (const auto offload : offloads)
	{
		long duration_execute = 0;
		for (int i = 0; i < iter; i++)
		{
			auto start = std::chrono::system_clock::now();

			err = device.execute(
				task, offload,
				{ (size_t)M,(size_t)N,1 },
				{ 0,0,0 },
				A, B);

			on_error(err);

			auto b = (const cl_uchar*)B->data();

			auto end = std::chrono::system_clock::now();
			const auto et = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
			std::cout << "Pixels_" << (width*height*sizeof(cl_uchar4))*1e-3 << "," << et << "," << label_offload(offload) << "\n";
			duration_execute += et;

			assert(B->items() == vb_ref.size());
			for (auto id = 0; id < vb_ref.size(); id++)
			{
				if (b[id] != vb_ref[id])
				{
					std::cerr << "######### Something wrong at pos:\t" << id << std::endl;
					std::cerr << b[id] << " != " << vb_ref[id] << std::endl;
					return -1;
				}
			}				
		}
		if (iter > 1)
		{
			if (verbose) {
				std::cout << std::endl;
				std::cout << "\t Elapsed mean time:\t" << duration_execute / iter << " ms\n";
			}
		}
	}

	if(verbose)
		std::cout << "Passed! exit..." << std::endl;
	return 0;
}
