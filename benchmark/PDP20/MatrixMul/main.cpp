#include "common.h"
#include "clDriver.h"
#include "assert.h"


constexpr auto sgemm = R"(kernel void sgemm(
                              const int M,
                              const int N,
                              const int K,
                              const __global float* A,
                              const __global float* B,
                              __global float* C)
                              {                              
								  // Thread identifiers
								  const int x = get_global_id(0); // Row ID of C (0..M)
								  const int y = get_global_id(1); // Col ID of C (0..N)

								  // Compute a single element (loop over K)
								  float acc = 0.0f;

								  for (int k=0; k<K; k++) 
									acc += A[k*M + x] * B[y*K + k];                             
								 
								  // Store the result								  
								  C[y*M + x] = acc;
                              })";

using namespace coopcl;

static void ref_gold(
	const std::array<size_t, 3> gs,
	const std::array<size_t, 3> ls,
	const int M, const  int N, const  int K,
	const std::vector<float>& A,
	const std::vector<float>& B,
	std::vector<float>& C,
	const bool verbose)
{
	const auto count_gx = gs[0] / ls[0];
	const auto count_gy = gs[1] / ls[1];
	const auto count_gz = gs[2] / ls[2];

	if (verbose)
	{
		std::cout << "------------------------------" << std::endl;
		std::cout << "calculate gold/reference..." << std::endl;
	}

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
							const auto globalRow = gy*ls[1] + ly;
							const auto globalCol = gx*ls[0] + lx;

							// Compute a single element (loop over K)
							float acc = 0.0f;
							for (int k = 0; k < K; k++) {
								acc += A[k*M + globalRow] * B[globalCol*K + k];
							}
							// Store the result
							C[globalCol*M + globalRow] = acc;
						}
					}
				}
			}
		}
	}
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
	const int M = std::atoi(argv[1]);
	const int N = M;
	const int K = M;

	std::vector<float> va(M*K); init_random(va);
	std::vector<float> vb(K*N); init_random(vb);
	std::vector<float> vc(M*N, 0);
	std::vector<float> vc_ref(M*N, 0);

	auto A = device.alloc(va, true);
	auto B = device.alloc(vb, true);	

	ref_gold({ (size_t)M,(size_t)N,1 }, { 1,1,1 }, M, N, K, va, vb, vc_ref,verbose);

	coopcl::clTask task;
	device.build_task(task,{ (size_t)(M),(size_t)(N),(size_t)(1) }, sgemm, "sgemm");
	
	header();

	for (const auto offload : offloads)
	{
		long duration_execute = 0;
		for (int i = 0; i < iter; i++)
		{
			auto C = device.alloc(vc, false);

			auto start = std::chrono::system_clock::now();

			err = device.execute(task,
				offload,
				{ (size_t)M,(size_t)N,1 },
				{ 16,16,1 },
				M, N, K, A, B, C);

			on_error(err);

			auto b = (const float*)C->data();

			auto end = std::chrono::system_clock::now();
			const auto et = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
			std::cout << "MatrixSize_" << (M*N*sizeof(float))*1e-3 << "," << et << "," << label_offload(offload) << "\n";
			duration_execute += et;

			assert(C->items() == vc_ref.size());
			for (auto id = 0; id < vc_ref.size(); id++)
			{
				if (!cmpf(b[id], vc_ref[id]))
				{
					std::cerr << "######### Something wrong at pos:\t" << id << std::endl;
					std::cerr << b[id] << " != " << vc_ref[id] << std::endl;
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
