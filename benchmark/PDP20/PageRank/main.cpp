#include "common.h"
#include "clDriver.h"
#include "parse.h"

#define ITER 20

int main(int argc, char **argv)
{	
	char *tmpchar;
	char *filechar;

	int num_nodes;
	int num_edges;		
	bool directed = 0;
	
	const bool verbose = false;

	cl_int err = 0;

	if (argc == 3) 
	{
		tmpchar = argv[1];            //graph input_file
		filechar = argv[2];	          //kernel input_file		
	}
	else 
	{
		std::cerr << "Usage:\t" << "./app.exe <string>Dataset_path <string>kernel_path" << std::endl;
		std::cerr << "Example:\t" << "./app.exe c:/Development/pannotia/dataset/pagerank/coPapersDBLP.graph c:/Development/pannotia/graph_app/prk/kernel/kernel_spmv.cl" << std::endl;
		exit(1);
	}

	//allocate the csr structure
	csr_array *csr = (csr_array *)malloc(sizeof(csr_array));
	if (!csr) fprintf(stderr, "malloc failed csr\n");

	//parse the metis format file and store it in a csr format
	//when loading the file, swap the head and tail pointers
	csr = parseMetis(tmpchar, &num_nodes, &num_edges, directed, verbose);

	//allocate the page_rank array 1
	float *pagerank_array = (float *)malloc(num_nodes * sizeof(float));
	if (!pagerank_array) fprintf(stderr, "malloc failed page_rank_array\n");

	//allocate the page_rank array 2
	float *pagerank_array2 = (float *)malloc(num_nodes * sizeof(float));
	if (!pagerank_array2) fprintf(stderr, "malloc failed page_rank_array2\n");

	//load the OpenCL kernel source files
	int sourcesize = 1024 * 1024;
	char * source = (char *)calloc(sourcesize, sizeof(char));
	if (!source) { fprintf(stderr, "ERROR: calloc(%d) failed\n", sourcesize); return -1; }

	FILE * fp = fopen(filechar, "rb");
	if (!fp) { fprintf(stderr, "ERROR: unable to open '%s'\n", filechar); return -1; }
	fread(source + strlen(source), sourcesize, 1, fp);
	fclose(fp);
	
	coopcl::virtual_device device;
	
	//create OpenCL device-side buffers for the graph		
	auto row_d = device.alloc(num_nodes+1, csr->row_array);
	auto col_d = device.alloc(num_edges, csr->col_array);
	auto data_d = device.alloc(num_edges, 0.0f);
	
	//create OpenCL buffers for page_rank	
	auto pagerank1_d = device.alloc(num_nodes, 0.0f);
	auto pagerank2_d = device.alloc(num_nodes, 0.0f);
	auto col_cnt_d=device.alloc(num_nodes, csr->col_cnt);
	
	//set up OpenCL work dimensions	
	int block_size = 64;
	int global_size = (num_nodes%block_size == 0) ? num_nodes : (num_nodes / block_size + 1) * block_size;

    size_t local_work[3] = { static_cast<size_t>(block_size),  1, 1 };
    size_t global_work[3] = { static_cast<size_t>(global_size), 1, 1 };

	//create OpenCL kernels 	
	coopcl::clTask kernelprk1;
	device.build_task(kernelprk1,{ global_work[0],global_work[1],global_work[2] }, std::string(source), "inibuffer");
	coopcl::clTask kernelprk2;
	device.build_task(kernelprk2,{ global_work[0],global_work[1],global_work[2] }, std::string(source), "inicsr");
	coopcl::clTask kernelprk3;
	device.build_task(kernelprk3,{ global_work[0],global_work[1],global_work[2] }, std::string(source), "spmv_csr_scalar_kernel");
	coopcl::clTask kernelprk4;
	device.build_task(kernelprk4,{ global_work[0],global_work[1],global_work[2] }, std::string(source), "pagerank2");

	std::vector<float> offloads;
	auto step = 0.1f;
	generate_offload_range(offloads, step);

	header();

	for (const auto offload : offloads)
	{		
		err = device.execute(kernelprk1, offload,
		{ global_work[0],global_work[1],global_work[2] },
		{ local_work[0],local_work[1],local_work[2] },
			pagerank1_d, pagerank2_d, num_nodes);
		if (err != CL_SUCCESS) { fprintf(stderr, "ERROR: clEnqueueNDRangeKernel()=>%d failed\n", err); return -1; }

		err = device.execute(kernelprk2, offload,
		{ global_work[0],global_work[1],global_work[2] },
		{ local_work[0],local_work[1],local_work[2] },
			row_d, col_d, data_d, col_cnt_d, num_nodes, num_edges);
		if (err != CL_SUCCESS) { fprintf(stderr, "ERROR: clEnqueueNDRangeKernel()=>%d failed\n", err); return -1; }

		//run PageRank for some iter. TO: convergence determination
		for (int i = 0; i < ITER; i++)
		{
			const auto  timer1 = std::chrono::system_clock::now();

			//launch the simple spmv kernel
			err = device.execute(kernelprk3, offload,
			{ global_work[0],global_work[1],global_work[2] },
			{ local_work[0],local_work[1],local_work[2] },
				num_nodes, row_d, col_d, data_d, pagerank1_d, pagerank2_d);
			if (err != CL_SUCCESS) { fprintf(stderr, "ERROR: clEnqueueNDRangeKernel()=>%d failed\n", err); return -1; }

			//launch the page_rank update kernel
			err = device.execute(kernelprk4, offload,
			{ global_work[0],global_work[1],global_work[2] },
			{ local_work[0],local_work[1],local_work[2] },
				pagerank1_d, pagerank2_d, num_nodes);
			if (err != CL_SUCCESS) { fprintf(stderr, "ERROR: clEnqueueNDRangeKernel()=>%d failed\n", err); return -1; }
			
			const auto timer2 = std::chrono::system_clock::now();
			const auto et = std::chrono::duration_cast<std::chrono::milliseconds>(timer2 - timer1).count();
			std::cout << "Nodes_" << (num_nodes) << "," << et << "," << label_offload(offload) << "\n";
		}				
	}		

    return 0;
}

