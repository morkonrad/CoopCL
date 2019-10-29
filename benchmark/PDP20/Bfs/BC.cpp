
#include <algorithm>
#include <iostream>
#include <chrono>

#include <stdio.h>
#include <stdlib.h>

#include <string>
#include "clDriver.h"
#include "BC.h"

int main(int argc, char **argv) 
{
	char *tmpchar;
	char *filechar;

	int num_nodes;
	int num_edges;
	int use_gpu = 1;
	int file_format = 1;
	bool directed = 1;

	cl_int err = 0;

	if (argc == 4) {
		tmpchar = argv[1];			//graph input_file
		filechar = argv[2];			//kernel file
		file_format = atoi(argv[3]); //file format
	}
	else {		
		std::cerr << "Usage:\t" << "./app.exe <string>Dataset_path <string>kernel_path <int 0|1>data_format" << std::endl;
		std::cerr << "Example:\t" << "./app.exe c:/Development/pannotia/dataset/bc/2k_1M.gr c:/Development/pannotia/graph_app/bc/kernel/kernel.cl 0" << std::endl;
		exit(1);
	}

	//allocate the csr structure
	csr_array *csr = (csr_array *)malloc(sizeof(csr_array));
	if (!csr) fprintf(stderr, "malloc failed csr\n");

	//parse graph and store it in a CSR format
	csr = parseCOO(tmpchar, &num_nodes, &num_edges, directed);

	//allocate the bc host array
	float *bc_h = (float *)malloc(num_nodes * sizeof(float));
	if (!bc_h) fprintf(stderr, "malloc failed bc_h\n");

	//load kernel file
	int sourcesize = 1024 * 1024;
	char * source = (char *)calloc(sourcesize, sizeof(char));
	if (!source) { fprintf(stderr, "ERROR: calloc(%d) failed\n", sourcesize); return -1; }

	FILE * fp = fopen(filechar, "rb");
	if (!fp) { fprintf(stderr, "ERROR: unable to open '%s'\n", filechar); return -1; }
	fread(source + strlen(source), sourcesize, 1, fp);
	fclose(fp);

	clDriver::virtual_device device;
	
	// Create OpenCL programs		
	const auto kernelbc1 = device.build_task({ (size_t)num_nodes,1,1, }, std::string(source), "bfs_kernel");
	const auto kernelbc2 = device.build_task({ (size_t)num_nodes,1,1, }, std::string(source), "backtrack_kernel");
	const auto kernelbc3 = device.build_task({ (size_t)num_nodes,1,1, }, std::string(source), "clean_1d_array");
	const auto kernelbc4 = device.build_task({ (size_t)(num_nodes*num_nodes),1,1, }, std::string(source), "clean_2d_array");
	const auto kernelbc5 = device.build_task({ (size_t)num_nodes,1,1, }, std::string(source), "clean_bc");

	//create shared memory
	clDriver::clMemory bc_d(num_nodes, 0.0f);
	clDriver::clMemory dist_d(num_nodes, 0);
	clDriver::clMemory sigma_d(num_nodes, 0.0f);
	clDriver::clMemory rho_d(num_nodes, 0.0f);
	clDriver::clMemory p_d(num_nodes * num_nodes, 0);
	clDriver::clMemory stop_d(1, 0);
	clDriver::clMemory row_d(num_nodes + 1, csr->row_array);	
	clDriver::clMemory col_d(num_edges, csr->col_array);	
	clDriver::clMemory row_trans_d(num_nodes + 1, csr->row_array_t);
	clDriver::clMemory col_trans_d(num_edges, csr->col_array_t);
	
	const auto iter = 10;
	long acc_time = 0;

	const float offload = 0.0;

	/* 128k
	0		6056
	0.25	8246
	0.5		8432
	0.75	8510
	1		7053
	*/


	for(int it = 0;it<iter;it++)
	{
		const auto timer1 = std::chrono::system_clock::now();		
		
		//set up kernel dimensions
		int local_worksize = 128;
		size_t local_work[3] = { local_worksize,  1, 1 };
		size_t global_work[3] = { (num_nodes%local_worksize == 0) ? num_nodes : (num_nodes / local_worksize + 1) * local_worksize, 1,  1 };

		//initialization
		err = device.execute(kernelbc5, offload,
		{ global_work[0], global_work[1],global_work[2] },
		{ local_work[0], local_work[1],local_work[2] },
			bc_d, num_nodes);
		if (err != CL_SUCCESS) { fprintf(stderr, "ERROR: 1  clEnqueueNDRangeKernel()=>%d failed\n", err); return -1; }

		// main computation loop
		for (int i = 0; i < num_nodes; i++)
		{
			size_t local_work[3] = { local_worksize,  1, 1 };
			size_t global_work[3] = { (num_nodes%local_worksize == 0) ? num_nodes : (num_nodes / local_worksize + 1) * local_worksize, 1,  1 };
			
			err = device.execute(kernelbc3, offload,
			{ global_work[0], global_work[1],global_work[2] },
			{ local_work[0], local_work[1],local_work[2] },
				i, dist_d, sigma_d, rho_d, num_nodes);
			if (err != CL_SUCCESS) { fprintf(stderr, "ERROR: 1  clEnqueueNDRangeKernel()=>%d failed\n", err); return -1; }

			global_work[0] = ((num_nodes * num_nodes) % local_worksize == 0) ? num_nodes * num_nodes : ((num_nodes * num_nodes) / local_worksize + 1) * local_worksize;
			
			err = device.execute(kernelbc4, offload,
			{ global_work[0], global_work[1],global_work[2] },
			{ local_work[0], local_work[1],local_work[2] },
				p_d, num_nodes);
			if (err != CL_SUCCESS) { fprintf(stderr, "ERROR: 1  clEnqueueNDRangeKernel()=>%d failed\n", err); return -1; }

			// depth of the traversal
			int dist = 0;
			
			// termination variable
			int stop = 1;

			//traverse the graph from the source node i
			do 
			{
				stop = 0;				
				stop_d.write(stop, 0);

				global_work[0] = (num_nodes%local_worksize == 0) ? num_nodes : (num_nodes / local_worksize + 1) * local_worksize;
				
				err = device.execute(kernelbc1, offload,
				{ global_work[0], global_work[1],global_work[2] },
				{ local_work[0], local_work[1],local_work[2] },
					row_d, col_d, dist_d, rho_d, p_d, stop_d, num_nodes, num_edges, dist);
				if (err != CL_SUCCESS) { fprintf(stderr, "ERROR: kernel1 (%d)\n", err); return -1; }
				
				stop = stop_d.at<int>(0);
				//another level
				dist++;
			} 
			while (stop);

			//traverse back from the deepest part of the tree
			while (dist)
			{
				global_work[0] = (num_nodes%local_worksize == 0) ? num_nodes : (num_nodes / local_worksize + 1) * local_worksize;

				err = device.execute(kernelbc2, offload,
				{ global_work[0], global_work[1],global_work[2] },
				{ local_work[0], local_work[1],local_work[2] },
					row_trans_d, col_trans_d, dist_d, rho_d, sigma_d, p_d, num_nodes, num_edges, dist, i, bc_d);
				if (err != CL_SUCCESS) { fprintf(stderr, "ERROR: kernel2 (%d)\n", err); return -1; }

				//back one level
				dist--;
			}
		}			

		const auto timer2 = std::chrono::system_clock::now();
		const auto et = std::chrono::duration_cast<std::chrono::milliseconds>(timer2 - timer1).count();
		std::cout << "Trial:\t(" << it + 1 << "," << iter << ")\t"<<et<<" ms"<< std::endl;
		acc_time += et;
	}

	std::cout<<"Elapsed mean time: "<< acc_time/iter << " ms\n";		


    return 0;

}
