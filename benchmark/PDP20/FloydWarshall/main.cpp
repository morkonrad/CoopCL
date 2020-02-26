#include "common.h"
#include "clDriver.h"

namespace floyd_common
{
#define FALSE 0
#define TRUE 1
#define BIGNUM 9999

	//adjacency matrix to CSR
	void adjmatrix2CSR(int* adjmatrix, int *row_array, int *col_array, int *data_array, int num_nodes, int num_edges) {

		int col_cnt = 0;
		int row_cnt = 0;

		bool first;
		for (int i = 0; i < num_nodes; i++) {
			first = FALSE;
			for (int j = 0; j < num_nodes; j++) {
				if (adjmatrix[i * num_nodes + j] != -1) {
					col_array[col_cnt] = j;
					data_array[col_cnt] = adjmatrix[i * num_nodes + j];
					if (first == FALSE) {
						row_array[row_cnt++] = col_cnt + 1;
						first = TRUE;
					}
					col_cnt++;
				}
			}
		}
		row_array[row_cnt] = num_edges;

	}

	bool test_value(int* array, int dim, int i, int j) {

		//TODO: current does not support multiple edges between two vertices
		if (array[i * dim + j] != -1) {
			//fprintf(stderr, "Possibly duplicate records at (%d, %d)\n", i, j);
			return 0;
		}
		else
			return 1;
	}

	void set_value(int* array, int dim, int i, int j, int value) {

		array[i * dim + j] = value;
		//printf("(%d, %d) = %d\n", i, j, value);
	}

	//parse graph file
	int* parse_graph_file(int *num_nodes, int *num_edges, char* tmpchar,const bool verbose) {

		int *adjmatrix;
		int cnt = 0;
		unsigned int lineno = 0;
		std::string sp;
		char line[128], a, p;

		FILE *fptr;

		fptr = fopen(tmpchar, "r");

		if (!fptr) {
			fprintf(stderr, "Error when opennning file: %s\n", tmpchar);
			exit(1);
		}
		if(verbose)
		printf("Opening file: %s\n", tmpchar);

		while (fgets(line, 100, fptr))
		{
			int head, tail, weight, size;
			switch (line[0])
			{
			case 'c':
				break;
			case 'p':
                sscanf(line, "%c %s %d %d", &p, sp.data(), num_nodes, num_edges);
				
				if (verbose)
					printf("Read from file: num_nodes = %d, num_edges = %d\n", *num_nodes, *num_edges);

				size = (*num_nodes + 1) * (*num_nodes + 1);
				adjmatrix = (int *)malloc(size * sizeof(int));
				memset(adjmatrix, -1, size * sizeof(int));
				break;
			case 'a':
				sscanf(line, "%c %d %d %d", &a, &head, &tail, &weight);
				if (tail == head) printf("reporting self loop\n");
				if (test_value(adjmatrix, *num_nodes + 1, head, tail)) {
					set_value(adjmatrix, *num_nodes + 1, head, tail, weight);
					cnt++;
				}

#ifdef VERBOSE
				printf("Adding edge: %d ==> %d ( %d )\n", head, tail, weight);
#endif			
				break;
			default:
				fprintf(stderr, "exiting loop\n");
				break;
			}
			lineno++;
		}

		*num_edges = cnt;
		
		if (verbose)
			printf("Actual added edges: %d\n", cnt);

		fclose(fptr);

		return adjmatrix;

	}

	//dump an array to a file
	void dump2file(int *adjmatrix, int num_nodes) {

		FILE *dptr;

		printf("Dumping the adjacency matrix to adjmatrix_dump.txt\n");
		dptr = fopen("adjmatrix_dump.txt", "w");
		for (int i = 0; i < num_nodes; i++) {
			for (int j = 0; j < num_nodes; j++) {
				fprintf(dptr, "%d ", adjmatrix[i*num_nodes + j]);
			}
			fprintf(dptr, "\n");
		}
		fclose(dptr);
	}

	void print_vector(int *vector, int num) {
		for (int i = 0; i < num; i++) {
			printf("%d ", vector[i]);
		}
		printf("\n");
	}


}

static
int call_floyd_warshall_naive(int argc, char** argv)
{
	const bool verbose = false;
	char *tmpchar;
	char *filechar;

	int num_nodes;
	int num_edges;
	cl_int err = 0;

	//get program input
	if (argc == 3) 
	{
		tmpchar = argv[1];  //graph input file
		filechar = argv[2]; //kernel file
	}
	else 
	{
		std::cerr << "Usage:\t" << "./app.exe <string>Dataset_path <string>kernel_path" << std::endl;
		std::cerr << "Example:\t" << "./app.exe c:/Development/pannotia/dataset/floydwarshall/256_16384.gr c:/Development/pannotia/graph_app/fw/kernel/kernel.cl" << std::endl;
		exit(1);
	}

	//parse the adjacency matrix
	int *adjmatrix = floyd_common::parse_graph_file(&num_nodes, &num_edges, tmpchar,verbose);
	int dim = num_nodes;

	//initialize the distance matrix
	int *distmatrix = (int *)malloc(dim * dim * sizeof(int));
	if (!distmatrix) fprintf(stderr, "malloc failed - distmatrix\n");

	//initialize the result matrix 
	int *result = (int *)malloc(dim * dim * sizeof(int));
	if (!result) fprintf(stderr, "malloc failed - result\n");

	//initialize the result matrix on the CPU 
	int *result_cpu = (int *)malloc(dim * dim * sizeof(int));
	if (!result_cpu) fprintf(stderr, "malloc failed - result_cpu\n");

	//TODO: now only supports integer weights
	//setup the input matrix
	std::vector<int> val;
	for (int i = 0; i < dim; i++) {
		for (int j = 0; j < dim; j++) {
			//diagonal 
			if (i == j)
				distmatrix[i * dim + j] = 0;
			//without edge
			else if (adjmatrix[i * dim + j] == -1)
				distmatrix[i * dim + j] = BIGNUM;
			//with edge
			else
				distmatrix[i * dim + j] = adjmatrix[i * dim + j];

			val.push_back(distmatrix[i * dim + j]);
		}
	}

	//load kernel file
	int sourcesize = 1024 * 1024;
	char * source = (char *)calloc(sourcesize, sizeof(char));
	if (!source) { fprintf(stderr, "ERROR: calloc(%d) failed\n", sourcesize); return -1; }

	FILE * fp = fopen(filechar, "rb");
	if (!fp) { fprintf(stderr, "ERROR: unable to open '%s'\n", filechar); return -1; }
	
	fread(source + strlen(source), sourcesize, 1, fp);
	fclose(fp);
	if (err != CL_SUCCESS) { fprintf(stderr, "ERROR: clBuildProgram() => %d\n", err); return -1; }

	coopcl::virtual_device device;
	auto dist_d = device.alloc(dim * dim, distmatrix);
	auto next_d = device.alloc(dim * dim, 0);

	//OpenCL work dimension

	size_t local_work[3] = { 16, 1, 1 };
    size_t global_work[3] = { static_cast<size_t>(num_nodes),
                              static_cast<size_t>(num_nodes),
                              static_cast<size_t>(dim) };
	
	coopcl::clTask kernelfw3d;
		device.build_task(kernelfw3d, std::string(source), "fw");

	const int iter = 10;	
	long acc_time = 0;	

	std::vector<float> offloads;
	auto step = 0.1f;
	generate_offload_range(offloads, step);

	header();

	for (const auto offload : offloads)
	{
		int it;
		for (it = 0; it < iter; it++)
		{
			const auto timer1 = std::chrono::system_clock::now();

			err = device.execute(kernelfw3d, offload,
			{ global_work[0],global_work[1],global_work[2] },
			{ local_work[0],local_work[1],local_work[2] },
				dist_d, next_d, dim);
			if (err != CL_SUCCESS) { fprintf(stderr, "ERROR: coopcl:execute() => %d\n", err); return -1; }

			const auto timer2 = std::chrono::system_clock::now();

			const auto et = std::chrono::duration_cast<std::chrono::milliseconds>(timer2 - timer1).count();
			std::cout << "Nodes_" << num_nodes*num_nodes<< "," << et << "," << label_offload(offload) << "\n";
			acc_time += et;
		}		

		if (verbose) {
			std::cout << std::endl;
			std::cout << "Elapsed mean time:\t" << acc_time / iter << " ms\n";
		}

		//below is the verification part
		//calculate the result on the CPU
		int *dist = distmatrix;
		for (int k = 0; k < dim; k++) {
			for (int i = 0; i < dim; i++) {
				for (int j = 0; j < dim; j++) {
					if (dist[i * dim + k] + dist[k * dim + j] < dist[i * dim + j]) {
						dist[i * dim + j] = dist[i * dim + k] + dist[k * dim + j];
					}
				}
			}
		}

		//check result
		bool check_flag = 0;
		for (int i = 0; i < dim; i++) {
			for (int j = 0; j < dim; j++) {
				if (dist[i * dim + j] != dist_d->at<int>(i * dim + j)) {
					printf("mismatch at (%d, %d) - (%d, %d) \n", i, j, dist[i * dim + j], result[i * dim + j]);
					check_flag = 1;
				}
			}
		}

		if (verbose) {
			//if there is result mismatch, report
			if (check_flag) printf("produce wrong results\n");
			printf("Finishing Floyd-Warshall\n");
		}
	}


	return 0;
}

static 
int call_floyd_warshall_block(int argc, char** argv)
{
	char *tmpchar;
	char *filechar;

	int num_nodes;
	int num_edges;
	
	const bool verbose = false;

	cl_int err = 0;

	if (argc == 3) {
		tmpchar = argv[1];  //graph input file
		filechar = argv[2]; //kernel file
	}
	else {
		std::cerr << "Usage:\t" << "./app.exe <string>Dataset_path <string>kernel_path " << std::endl;
		std::cerr << "Example:\t" << "./app.exe c:/Development/pannotia/dataset/floydwarshall/256_16384.gr c:/Development/pannotia/graph_app/fw/kernel/kernel_block.cl" << std::endl;
		exit(1);
	}

	//parse the adjacency matrix
	int *adjmatrix = floyd_common::parse_graph_file(&num_nodes, &num_edges, tmpchar,verbose);

	int dim = num_nodes;

	int *distmatrix = (int *)malloc(dim * dim * sizeof(int));
	if (!distmatrix) fprintf(stderr, "malloc failed - distmatrix\n");

	int *result = (int *)malloc(dim * dim * sizeof(int));
	if (!result) fprintf(stderr, "malloc failed - result\n");

	//TODO: now only supports integer weights
	//initialize the dist matrix
	for (int i = 0; i < dim; i++) {
		for (int j = 0; j < dim; j++) {
			if (i == j)
				distmatrix[i * dim + j] = 0;
			else if (adjmatrix[i * dim + j] == -1)
				distmatrix[i * dim + j] = BIGNUM;
			else
				distmatrix[i * dim + j] = adjmatrix[i * dim + j];
		}
	}

	//load the OpenCL kernel source
	int sourcesize = 1024 * 1024;
	char * source = (char *)calloc(sourcesize, sizeof(char));
	if (!source) { printf("ERROR: calloc(%d) failed\n", sourcesize); return -1; }

	FILE * fp = fopen(filechar, "rb");
	if (!fp) { printf("ERROR: unable to open '%s'\n", filechar); return -1; }
	fread(source + strlen(source), sourcesize, 1, fp);
	fclose(fp);
	
	coopcl::virtual_device device;
	const int block_size = 16;
	
	//create device-side buffers for floyd-warshall
	auto  dist_d=device.alloc(dim * dim, distmatrix);	
	
	//main computation loop
	int num_blk_per_dim = num_nodes / block_size;

	//create OpenCL kernels	
	coopcl::clTask kernel1;
	device.build_task(kernel1, std::string(source), "floydwarshall_dia_block");
	coopcl::clTask kernel2;
	device.build_task(kernel2, std::string(source), "floydwarshall_strip_blocks_x");
	coopcl::clTask kernel3;
	device.build_task(kernel3, std::string(source), "floydwarshall_strip_blocks_y");
	coopcl::clTask kernel4;
	device.build_task(kernel4, std::string(source), "floydwarshall_remaining_blocks");
	
	long acc_time = 0;
	const auto iter = 10;
	const float offload = 0.0f;

	header();
	
	for (int it = 0; it < iter; it++)
	{
		const auto timer1 = std::chrono::system_clock::now();

		for (int blk = 0; blk < num_blk_per_dim; blk++)
		{
			//phase 1
			err = device.execute(kernel1, offload,
			{ (size_t)block_size, (size_t)block_size, 1 },
			{ (size_t)block_size, (size_t)block_size, 1 },
				dist_d, blk, dim);
			if (err != CL_SUCCESS) { fprintf(stderr, "ERROR: 1  clEnqueueNDRangeKernel()=>%d failed\n", err); return -1; }

			//phase 2 A
			err = device.execute(kernel2, offload,
			{ (size_t)block_size*num_blk_per_dim, (size_t)block_size, 1 },
			{ (size_t)block_size, (size_t)block_size, 1 },
				dist_d, blk, dim);
			if (err != CL_SUCCESS) { fprintf(stderr, "ERROR: 2  clEnqueueNDRangeKernel()=>%d failed\n", err); return -1; }

			//phase 2 B
			err = device.execute(kernel3, offload,
			{ (size_t)block_size, (size_t)block_size*num_blk_per_dim, 1 },
			{ (size_t)block_size, (size_t)block_size, 1 },
				dist_d, blk, dim);
			if (err != CL_SUCCESS) { fprintf(stderr, "ERROR: 3  clEnqueueNDRangeKernel()=>%d failed\n", err); return -1; }

			//phase 3
			err = device.execute(kernel4, offload,
			{ (size_t)block_size*num_blk_per_dim, (size_t)block_size*num_blk_per_dim, 1 },
			{ (size_t)block_size, (size_t)block_size, 1 },
				dist_d, blk, dim);
			if (err != CL_SUCCESS) { fprintf(stderr, "ERROR: 4  clEnqueueNDRangeKernel()=>%d failed\n", err); return -1; }
		}
		const auto timer2 = std::chrono::system_clock::now();
		const auto et = std::chrono::duration_cast<std::chrono::milliseconds>(timer2 - timer1).count();
		acc_time += et;
	}
	std::cout << "Elapsed mean time:\t" << acc_time/iter << " ms\n";

	//below is the verification part
	//calculate the result on the CPU
	int *dist = distmatrix;
	for (int k = 0; k < dim; k++) {
		for (int i = 0; i < dim; i++) {
			for (int j = 0; j < dim; j++) {
				if (dist[i * dim + k] + dist[k * dim + j] < dist[i * dim + j]) {
					dist[i * dim + j] = dist[i * dim + k] + dist[k * dim + j];
				}
			}
		}
	}

	const int* results = (const int*)dist_d->data();

	//check result
	bool check_flag = 0;
	for (int i = 0; i < dim; i++) {
		for (int j = 0; j < dim; j++) {
			if (dist[i * dim + j] != results[i * dim + j]) {
				printf("mismatch at (%d, %d) - (%d, %d) \n", i, j, dist[i * dim + j], result[i * dim + j]);
				check_flag = 1;
			}
		}
	}
	//if there is result mismatch, report
	if (check_flag) printf("produce wrong results\n");
	printf("Finishing Floyd-Warshall\n");
	return 0;
}

int main(int argc, char** argv)
{
	return call_floyd_warshall_naive(argc, argv);
	//return call_floyd_warshall_block(argc, argv);

}

