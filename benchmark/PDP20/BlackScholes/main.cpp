#include "common.h"
#include "clDriver.h"
#include "assert.h"

#define S_LOWER_LIMIT 10.0f
#define S_UPPER_LIMIT 100.0f
#define K_LOWER_LIMIT 10.0f
#define K_UPPER_LIMIT 100.0f
#define T_LOWER_LIMIT 1.0f
#define T_UPPER_LIMIT 10.0f
#define R_LOWER_LIMIT 0.01f
#define R_UPPER_LIMIT 0.05f
#define SIGMA_LOWER_LIMIT 0.01f
#define SIGMA_UPPER_LIMIT 0.10f

void phi(float X, float* phi)
{
	float y;
	float absX;
	float t;
	float result;

	const float c1 = (float)0.319381530f;
	const float c2 = (float)-0.356563782f;
	const float c3 = (float)1.781477937f;
	const float c4 = (float)-1.821255978f;
	const float c5 = (float)1.330274429f;

	const float zero = (float)0.0f;
	const float one = (float)1.0f;
	const float two = (float)2.0f;
	const float temp4 = (float)0.2316419f;

	const float oneBySqrt2pi = (float)0.398942280f;

	absX = fabs(X);
	t = one / (one + temp4 * absX);

	y = one - oneBySqrt2pi * exp(-X*X / two) * t
		* (c1 + t
			* (c2 + t
				* (c3 + t
					* (c4 + t * c5))));

	result = (X < zero) ? (one - y) : y;

	*phi = result;

}

constexpr auto blackScholes = R"(

#define S_LOWER_LIMIT 10.0f
#define S_UPPER_LIMIT 100.0f
#define K_LOWER_LIMIT 10.0f
#define K_UPPER_LIMIT 100.0f
#define T_LOWER_LIMIT 1.0f
#define T_UPPER_LIMIT 10.0f
#define R_LOWER_LIMIT 0.01f
#define R_UPPER_LIMIT 0.05f
#define SIGMA_LOWER_LIMIT 0.01f
#define SIGMA_UPPER_LIMIT 0.10f



void phi(const float X, float* phi)
{
    float y;
    float absX;
    float t;
    float result;

    const float c1 = (float)0.319381530f;
    const float c2 = (float)-0.356563782f;
    const float c3 = (float)1.781477937f;
    const float c4 = (float)-1.821255978f;
    const float c5 = (float)1.330274429f;

    const float zero = (float)0.0f;
    const float one = (float)1.0f;
    const float two = (float)2.0f;
    const float temp4 = (float)0.2316419f;

    const float oneBySqrt2pi = (float)0.398942280f;

    absX = fabs(X);
    t = one/(one + temp4 * absX);

    y = one - oneBySqrt2pi * exp(-X*X/two) * t 
        * (c1 + t
              * (c2 + t
                    * (c3 + t
                          * (c4 + t * c5))));

    result = (X < zero)? (one - y) : y;

    *phi = result;
}

__kernel void
blackScholes(const __global float* restrict randArray,
             const int width,
             __global float* restrict call,
             __global float* restrict put)
{
    float d1, d2;
    float phiD1, phiD2;
    float sigmaSqrtT;
    float KexpMinusRT;
    
    size_t xPos = get_global_id(0);
    size_t yPos = get_global_id(1);	


    float two = (float)2.0f;
    float inRand = randArray[yPos * width + xPos];
    float S = S_LOWER_LIMIT * inRand + S_UPPER_LIMIT * (1.0f - inRand);
    float K = K_LOWER_LIMIT * inRand + K_UPPER_LIMIT * (1.0f - inRand);
    float T = T_LOWER_LIMIT * inRand + T_UPPER_LIMIT * (1.0f - inRand);
    float R = R_LOWER_LIMIT * inRand + R_UPPER_LIMIT * (1.0f - inRand);
    float sigmaVal = SIGMA_LOWER_LIMIT * inRand + SIGMA_UPPER_LIMIT * (1.0f - inRand);

    sigmaSqrtT = sigmaVal * sqrt(T);

    d1 = (log(S/K) + (R + sigmaVal * sigmaVal / two)* T)/ sigmaSqrtT;
    d2 = d1 - sigmaSqrtT;

    KexpMinusRT = K * exp(-R * T);
    phi(d1, &phiD1);
    phi(d2, &phiD2);
	
    call[yPos * width + xPos] = S * phiD1 - KexpMinusRT * phiD2;

    phi(-d1, &phiD1);
    phi(-d2, &phiD2);
    
	put[yPos * width + xPos]  = KexpMinusRT * phiD2 - S * phiD1;
}
)";

static void ref_gold(
	const std::array<size_t, 3> gs,
	const std::array<size_t, 3> ls,
	const std::vector<cl_float>& randArray,
	const int width,
	std::vector<cl_float> &call,
	std::vector<cl_float> &put,
	const bool verbose)
{
	const auto count_gx = gs[0] / ls[0];
	const auto count_gy = gs[1] / ls[1];
	const auto count_gz = gs[2] / ls[2];

	if (verbose) {
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
							cl_float d1, d2;
							cl_float phiD1, phiD2;
							cl_float sigmaSqrtT;
							cl_float KexpMinusRT;

							size_t xPos = gx*ls[0] + lx;
							size_t yPos = gy*ls[1] + ly;

							cl_float two = (cl_float)2.0f;

							cl_float inRand = randArray[yPos * width + xPos];
							cl_float S = S_LOWER_LIMIT * inRand + S_UPPER_LIMIT * (1.0f - inRand);
							cl_float K = K_LOWER_LIMIT * inRand + K_UPPER_LIMIT * (1.0f - inRand);
							cl_float T = T_LOWER_LIMIT * inRand + T_UPPER_LIMIT * (1.0f - inRand);
							cl_float R = R_LOWER_LIMIT * inRand + R_UPPER_LIMIT * (1.0f - inRand);
							cl_float sigmaVal = SIGMA_LOWER_LIMIT * inRand + SIGMA_UPPER_LIMIT * (1.0f - inRand);


							sigmaSqrtT = sigmaVal * sqrt(T);

							d1 = (log(S / K) + (R + sigmaVal * sigmaVal / two)* T) / sigmaSqrtT;
							d2 = d1 - sigmaSqrtT;

							KexpMinusRT = K * exp(-R * T);
							phi(d1, &phiD1);
							phi(d2, &phiD2);

							call[yPos * width + xPos] = S * phiD1 - KexpMinusRT * phiD2;

							phi(-d1, &phiD1);
							phi(-d2, &phiD2);

							put[yPos * width + xPos] = KexpMinusRT * phiD2 - S * phiD1;

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

	int err = 0;
	virtual_device device;
	std::vector<float> offloads;
	auto step = 0.1f;
	generate_offload_range(offloads, step);
	constexpr auto iter = 10;

	const int M = std::atoi(argv[1]);
	const int N = M;

	std::vector<float> va(M*N); init_random(va);
	std::vector<float> vb(M*N, 0);
	std::vector<float> vc(M*N, 0);

	std::vector<float> vb_ref(M*N, 0);
	std::vector<float> vc_ref(M*N, 0);

	auto A = device.alloc(va, true);
	ref_gold({ (size_t)M,(size_t)N,1 }, { 1,1,1 }, va, M, vb_ref, vc_ref,verbose);

	coopcl::clTask task;
	device.build_task(task, blackScholes, "blackScholes");
	header();

	for (const auto offload : offloads)
	{
		long duration_execute = 0;
		for (int i = 0; i < iter; i++)
		{
			auto B = device.alloc(vb);
			auto C = device.alloc(vc);

			auto start = std::chrono::system_clock::now();

			err = device.execute(task,
			offload,
			{ (size_t)M,(size_t)N,1 },
			{ 0,0,0 },
			A, M, B, C);
			on_error(err);

			auto b = (const float*)C->data();

			auto end = std::chrono::system_clock::now();
			const auto et = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
			std::cout <<"Options_"<<M<<","<<et << ","<< label_offload(offload)<<"\n";
			duration_execute += et;

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
		if (iter > 1) {
			if (verbose) {
				std::cout << std::endl;
				std::cout << "\t Elapsed mean time:\t" << duration_execute / iter << " ms\n";
			}
		}			
	}	
	
	if(verbose)
		std::cout<< "Passed! exit..." << std::endl;

	return 0;
}
