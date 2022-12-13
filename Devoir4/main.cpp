#include <stdio.h>
#include <sys/stat.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif
#define INF 99999

void print_matrix(int* mat, int n) {
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++)
			printf("%d ", mat[i * n + j]);
		printf("\n");
	}
	printf("\n");
}

void init_graphe(int* graphe, int n) {
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			if (i == j)
				graphe[i * n + j] = 0;
			else if (j == i + 1 || (i == n - 1 && j == 0)) {
				graphe[i * n + j] = 1;
			}
			else
				graphe[i * n + j] = n + 1;
		}
	}
}

void copy(int* mat1, int* mat2, int n) {
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++)
			mat2[i * n + j] = mat1[i * n + j];
	}
}

void init2(int* graphe, int n) {
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			if (i == j)
				graphe[i * n + j] = 0;
			else
				graphe[i * n + j] = INF;
		}
	}
	graphe[1] = 5;
	graphe[3] = 10;
	graphe[6] = 3;
	graphe[11] = 1;
}

void floyd_seq(int* graphe, int n) {
	int* D = (int*)malloc(sizeof(int) * n * n);

	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++)
			D[i * n + j] = graphe[i * n + j];
	}

	for (int k = 0; k < n; k++) {
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < n; j++) {
				if (D[i * n + k] + D[k * n + j] < D[i * n + j])
					D[i * n + j] = D[i * n + k] + D[k * n + j];
			}
		}
	}
	print_matrix(D, n);
}

char* load_kernel(const char* filename) {
	FILE* fp;
	char* source;
	int sz = 0;
	struct stat status;

	fp = fopen(filename, "rb");
	if (fp == 0) {
		printf("Echec\n");
		return 0;
	}

	if (stat(filename, &status) == 0)
		sz = (int)status.st_size;

	source = (char*)malloc(sz + 1);
	fread(source, sz, 1, fp);
	source[sz] = '\0';

	return source;
}

void getError(cl_int status, cl_program program, cl_device_id* devices, int line) {
	if (status) {
		printf("ligne %d, ERREUR A LA COMPILATION: %d\n", line, status);
		size_t log_size;
		switch (status)
		{
		
		default:
			break;
		}
		printf("\n");
	}
}

int main() {
	int n;
	printf("entrez la valeur de n: ");
	scanf("%d", &n);
	printf("\n");
	int n2 = n * n;
	int matSize = n2 * sizeof(int*);
	int* graphe = (int*)calloc(n * n, sizeof(int*));
	int* distances = (int*)calloc(n * n, sizeof(int*));
	if (graphe == NULL)
		printf("malloc failed\n");
	//mat[i,j] => mat[i*n + j]
	init_graphe(graphe, n);
	//floyd_seq(graphe, n);
	copy(graphe, distances, n);

	char* programSource = load_kernel("kernel.cl");

	cl_int status;
	// STEP 1: Discover and initialize the platforms

	cl_uint numPlatforms = 0;

	cl_platform_id* platforms = NULL;

	// Calcul du nombre de plateformes
	status = clGetPlatformIDs(0, NULL, &numPlatforms);
	printf("Number of platforms = %d\n", numPlatforms);

	// Allocation de l'espace
	platforms = (cl_platform_id*)malloc(numPlatforms * sizeof(cl_platform_id));

	// Trouver les plateformes
	status = clGetPlatformIDs(numPlatforms, platforms, NULL);

	char Name[1000];
	clGetPlatformInfo(platforms[0], CL_PLATFORM_NAME, sizeof(Name), Name, NULL);
	printf("Name of platform : %s\n", Name);
	fflush(stdout);

	// STEP 2: Discover and initialize the devices

	cl_uint numDevices = 0;
	cl_device_id* devices = NULL;

	// calcul du nombre de périphériques
	status = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_ALL, 0, NULL, &numDevices);


	printf("Number of devices = %d\n", (int)numDevices);

	// Allocation de l'espace
	devices = (cl_device_id*)malloc(numDevices * sizeof(cl_device_id));

	// Trouver les périphériques
	status = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_ALL, numDevices, devices, NULL);


	for (int i = 0; i < numDevices; i++) {
		clGetDeviceInfo(devices[i], CL_DEVICE_NAME, sizeof(Name), Name, NULL);
		printf("Name of device %d: %s\n\n", i, Name);
	}

	// STEP 3: Create a context
	printf("Création du contexte\n");
	fflush(stdout);

	cl_context context = NULL;

	context = clCreateContext(NULL, numDevices, devices, NULL, NULL, &status);

	// STEP 4: Create a command queue

	printf("Création de la file d'attente\n");
	fflush(stdout);
	cl_command_queue cmdQueue;

	cmdQueue = clCreateCommandQueue(context, devices[0], 0, &status);    //tester aussi avec devices[0]

	// STEP 5: Create device buffers

	printf("Création des buffers\n");
	fflush(stdout);

	cl_mem buffer_graphe;
	cl_mem buffer_n;
	cl_mem buffer_distances;
	cl_mem buffer_k;

	buffer_graphe = clCreateBuffer(context, CL_MEM_READ_WRITE, matSize, NULL, &status);
	buffer_n = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int), NULL, &status);
	buffer_distances = clCreateBuffer(context, CL_MEM_READ_WRITE, matSize, NULL, &status);
	buffer_k = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int), NULL, &status);

	// STEP 6: Write host data to device buffers

	printf("Ecriture dans les buffers\n");
	fflush(stdout);


	// STEP 7: Create and compile the program

	printf("CreateProgramWithSource\n");
	fflush(stdout);

	cl_program program = clCreateProgramWithSource(context, 1, (const char**)&programSource, NULL, &status);
	printf("Compilation\n");
	fflush(stdout);
	status = clBuildProgram(program, numDevices, devices, NULL, NULL, NULL);

	getError(status, program, devices, 217);

	// STEP 8: Create the kernel

	cl_kernel kernel = NULL;

	printf("Création du kernel\n");
	fflush(stdout);
	kernel = clCreateKernel(program, "floyd", &status);

	size_t globalWorkSize[2] = { n, n };
	size_t localWorkSize[3] = { 64,64 };

	clock_t start = clock();
	int k;
	for (k = 0; k < n; k++) {
		status = clEnqueueWriteBuffer(cmdQueue, buffer_graphe, CL_TRUE, 0, matSize, graphe, 0, NULL, NULL);
		getError(status, program, devices, 235);
		status = clEnqueueWriteBuffer(cmdQueue, buffer_n, CL_TRUE, 0, sizeof(int), &n, 0, NULL, NULL);
		getError(status, program, devices, 237);
		status = clEnqueueWriteBuffer(cmdQueue, buffer_distances, CL_TRUE, 0, matSize, distances, 0, NULL, NULL);
		getError(status, program, devices, 239);
		status = clEnqueueWriteBuffer(cmdQueue, buffer_k, CL_TRUE, 0, sizeof(int), &k, 0, NULL, NULL);
		getError(status, program, devices, 241);

		status = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&buffer_graphe);
		getError(status, program, devices, 244);
		status = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&buffer_n);
		getError(status, program, devices, 246);
		status = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&buffer_distances);
		getError(status, program, devices, 248);
		status = clSetKernelArg(kernel, 3, sizeof(cl_mem), (void*)&buffer_k);
		getError(status, program, devices, 250);

		//printf("Debut des appels\n");
		status = clEnqueueNDRangeKernel(cmdQueue, kernel, 2, NULL, globalWorkSize, NULL, 0, NULL, NULL);

		//printf("Fin premier appel: status=%d\n", status);
		clFinish(cmdQueue);  // Pas nécessaire car la pile a été créée "In-order"

		status = clEnqueueReadBuffer(cmdQueue, buffer_distances, CL_TRUE, 0, matSize, distances, 0, NULL, NULL);
		getError(status, program, devices, 260);
		//status = clEnqueueNDRangeKernel(cmdQueue, kernel, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
		//printf("Fin second appel: status=%d\n", status);
		clFinish(cmdQueue);
	}
	clock_t end = clock();
	double time_spent = (double)(end - start) / CLOCKS_PER_SEC;


	//print_matrix(distances, n);
	printf("temps d'execution : %f sec", time_spent);
	// Free OpenCL resources
	clReleaseKernel(kernel);
	clReleaseProgram(program);
	clReleaseCommandQueue(cmdQueue);
	clReleaseMemObject(buffer_graphe);
	clReleaseMemObject(buffer_n);
	clReleaseMemObject(buffer_distances);
	clReleaseMemObject(buffer_k);
	clReleaseContext(context);

	// Free host resources
	free(platforms);
	free(devices);
	free(graphe);
	free(distances);



	return 0;
}

