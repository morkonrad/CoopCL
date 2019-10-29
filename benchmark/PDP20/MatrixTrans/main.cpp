#include "common.h"
#include "clDriver.h"
#include "assert.h"


constexpr auto transpose = R"(
kernel void transpose(    const global float* in,
                            global float* out,
                            const int w, const int h)
{	
	const int x = get_global_id(0);
	const int y = get_global_id(1);
	out[(x * h) + y] = in[(y * w) + x];
}
)";

using namespace coopcl;

static void ref_gold(
	virtual_device& device,
	clTask& task,
	const std::array<size_t, 3> gs,
	const std::array<size_t, 3> ls,
	const std::vector<float>& input,
	std::vector<float>& output,
	const int width, const int height,const bool verbose)
{
	auto d_input=device.alloc(input);
	auto d_output = device.alloc(output);
	
	if (verbose) {
		std::cout << "------------------------------" << std::endl;
		std::cout << "calculate gold/reference..." << std::endl;
	}

	//execute only on CPU offload 0
	const auto ok = device.execute(
		task, 0,
		{ gs[0],gs[1],gs[2] },
		{ ls[0],ls[1],ls[2] },
		d_input, d_output,
		width, height);
	assert(ok == 0);
	const auto ptr = (const float*)d_output->data();
	for (size_t i = 0; i < output.size(); i++)
		output[i] = ptr[i];

	return;
}

int main(int argc,char** argv)
{
	if (argc != 2)
	{
		std::cerr << "Usage:\t" << "./app.exe <int>array_size>0\n";
		std::cerr << "Example:\t" << "./app.exe 1024\n";
		return -1;
	}
	const bool verbose = false;

	int err = 0;
	virtual_device device;

	std::vector<float> offloads;
	auto step = 0.1f;
	generate_offload_range(offloads, step);

	constexpr auto iter = 10;
	const int width = std::atoi(argv[1]);
	const int height = width;

	std::vector<float> h_input(width*height); init_random(h_input);
	std::vector<float> h_output(width*height, 0);
	std::vector<float> h_output_ref(width*height, 0);

	auto d_input = device.alloc(h_input, true);

	coopcl::clTask task; device.build_task(task,
	{ (size_t)(width),(size_t)(height),1 },
		transpose, "transpose");

	ref_gold(device, task,
	{ (size_t)width,(size_t)height,1 },
	{ 16,16,1 },
		h_input, h_output_ref,
		width, height,verbose);
	
	header();

	for (const auto offload : offloads)
	{
		long duration_execute = 0;
		for (int i = 0; i < iter; i++)
		{
			auto d_output = device.alloc(h_output);

			auto start = std::chrono::system_clock::now();

			err = device.execute(
				task, offload,
				{ (size_t)width,(size_t)height,1 },
				{ 16,16,1 },
				d_input, d_output,
				width, height);

			on_error(err);

			auto b = (const float*)d_output->data();

			auto end = std::chrono::system_clock::now();
			const auto et = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
			std::cout << "MatrixSize_" << (width*height* sizeof(float))*1e-3 << "," << et << "," << label_offload(offload) << "\n";
			duration_execute += et;

			assert(d_output->items() == h_output_ref.size());
			for (auto id = 0; id < h_output_ref.size(); id++)
			{
				if (!cmpf(b[id], h_output_ref[id]))
				{
					std::cerr << "######### Something wrong at pos:\t" << id << std::endl;
					std::cerr << b[id] << " != " << h_output_ref[id] << std::endl;
					return -1;
				}
			}			
		}
		if (iter > 1) {
			if (verbose) {
				std::cout << std::endl;
				std::cout << "\t Elapsed mean time:\t" << duration_execute / iter << " ms\n";
			}
		}
	}
	if (verbose)
		std::cout << "Passed! exit..." << std::endl;
	return 0;
}
