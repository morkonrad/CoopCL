#include "common.h"
#include "clDriver.h"
#include "assert.h"
#include <stdlib.h>

constexpr auto tasks = R"(
__kernel
void kAdd( const global float* restrict inputImageA,
			const global float* restrict inputImageB,
            global float* restrict outputImage)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    const int w = get_global_size(0);
    const int pix = mad24(y,w,x);
    outputImage[pix] = inputImageB[pix]+inputImageA[pix];	
}
)";



#define _FIGURE_DUMP_

int main(int argc,char** argv)
{
	if (argc != 3)
	{
		std::cerr << "Usage:\t app.exe vector_mega_items<size_t> offload<float 0:1>" << std::endl;
		std::cerr << "Example:\t app.exe 128 0.0" << std::endl;
		std::exit(-1);
	}
	const int item = std::atoi(argv[1]);
	const int offload = std::atof(argv[2]);

	coopcl::virtual_device device;
	const int w = item * 1e3;
	const int h = 1 * 1e3;
	const int items = w*h;
	
	auto imgBlur1 = device.alloc<float>(items);
	auto imgBlur2 = device.alloc<float>(items);	
	auto imgAdd = device.alloc<float>(items);

	const auto mem_read = 2 * items * sizeof(float)*1e-9;
	const auto mem_write =  items * sizeof(float)*1e-9;
#ifndef _FIGURE_DUMP_
	std::cout << "Memory size read:\t" << mem_read  << " GB" << std::endl;
	std::cout << "Memory size write:\t" << mem_write << " GB" << std::endl;
#else
	std::cout << "Category,Value,Units"<< std::endl;
	std::cout << "Memory size," << mem_write+ mem_read << ",GB" << std::endl;
#endif

	coopcl::clTask task_Add;
	device.build_task(task_Add,{ (size_t)w,(size_t)h,1 }, tasks, "kAdd");

	/*std::vector<float> offloads;
	auto step = 0.1f;
	generate_offload_range(offloads, step);	
	for (const auto offload : offloads)*/
	{
		const int iter = 10;
		long long et_acc = 0;
#ifndef _FIGURE_DUMP_
		std::cout << "Execute " << iter << " times offload: " << offload << std::endl;
#endif
		for (int i = 0;i < iter;i++)
		{
#ifndef _FIGURE_DUMP_
			std::cout << ">" << std::flush;
#endif
			const auto start = std::chrono::system_clock::now();
			device.execute_async(task_Add, offload, { (size_t)w,(size_t)h,1 }, { 0,0,0 }, imgBlur1, imgBlur2, imgAdd);
			const auto err = task_Add.wait();

			if (err != 0) { std::cerr << err << std::endl;return err; }

			const auto end = std::chrono::system_clock::now();
			const auto et = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();			
			if(i>0) et_acc += et;			
#ifndef _FIGURE_DUMP_
			if (i == iter - 1) std::cout << ">" << std::endl;
#endif
		}
		const auto mean_time_sec = (et_acc*1e-9) / iter;
#ifndef _FIGURE_DUMP_
		std::cout << "Elapsed mean time:\t" << mean_time_sec << " sec, offload:\ " << offload << std::endl;
		std::cout << "Elapsed mean bandwidth:\t" << (mem_write + mem_read)/ (mean_time_sec) << " GB/sec offload:\ " << offload << std::endl;
#else
		std::cout << "Memory bandwidth," << (mem_write + mem_read) / (mean_time_sec) << ","<< "GB/sec" << std::endl;
		std::cout << "Offload CPU/GPU," << offload <<","<< (offload==0?"CPU":"GPU")<<std::endl;
#endif
	}

	return 0;
}
