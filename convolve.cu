#include <stdio.h>
#include <helper_functions.h>
#include <helper_cuda.h>


#define N (3*3)
#define W 3
#define H 3

#define KERNEL_WIDTH 3
#define KERNEL_RADIUS 1
#define kernel_N (KERNEL_WIDTH*KERNEL_WIDTH)
#define Padding_Entry 0
#define Bias -1

#define BLOCK_DIM_X 3
#define BLOCK_DIM_Y 3

#define Shared_Width BLOCK_DIM_X+2*KERNEL_RADIUS
#define Shared_Height BLOCK_DIM_Y+2*KERNEL_RADIUS

__constant__ int d_kernel[KERNEL_WIDTH * KERNEL_WIDTH];

__global__ void convolve2D(int* input, int* output, int width, int height, int kernel_width, int padding_entry)
{
	unsigned int shared_width = Shared_Width;
	unsigned int shared_height = Shared_Height;

	unsigned int ix, iy, tx, ty;
	__shared__ int shared_mem[Shared_Width * Shared_Height];

	int kernel_radius = kernel_width / 2;

	tx = threadIdx.x;
	ty = threadIdx.y;
	ix = blockIdx.x * blockDim.x + threadIdx.x;
	iy = blockIdx.y * blockDim.y + threadIdx.y;

	int shared_x = tx + kernel_radius;
	int shared_y = ty + kernel_radius;

	if (ix >= width || iy >= height) {
		shared_mem[shared_y * shared_width + shared_x] = padding_entry;
		return;
	}
	else {
		shared_mem[shared_y * shared_width + shared_x] = input[iy * width + ix];
	}

	// left, right, upper, bottom halo
	if (tx < kernel_radius) {
		int halo_x = ix - kernel_radius;
		int halo_y = iy;
		shared_mem[shared_y * shared_width + tx]
			= (halo_x >= 0 && halo_x < width&& halo_y >= 0 && halo_y < height) ? input[halo_y * width + halo_x] : Padding_Entry;
	}

	if (tx >= blockDim.x - kernel_radius) {
		int halo_x = ix + kernel_radius;
		int halo_y = iy;
		shared_mem[shared_y * shared_width + (shared_x + kernel_radius)]
			= (halo_x >= 0 && halo_x < width&& halo_y >= 0 && halo_y < height) ? input[halo_y * width + halo_x] : Padding_Entry;
	}


	if (ty < kernel_radius) {
		int halo_x = ix;
		int halo_y = iy - kernel_radius;
		shared_mem[ty * shared_width + shared_x]
			= (halo_x >= 0 && halo_x < width&& halo_y >= 0 && halo_y < height) ? input[halo_y * width + halo_x] : Padding_Entry;
	}

	if (ty >= blockDim.y - kernel_radius) {
		int halo_x = ix;
		int halo_y = iy + kernel_radius;
		shared_mem[(shared_y + kernel_radius) * shared_width + shared_x]
			= (halo_x >= 0 && halo_x < width&& halo_y >= 0 && halo_y < height) ? input[halo_y * width + halo_x] : Padding_Entry;
	}

	// Coner Halo

	// left up
	if (tx < kernel_radius || ty < kernel_radius) {
		int halo_x = ix - kernel_radius;
		int halo_y = iy - kernel_radius;
		shared_mem[ty * shared_width + tx]
			= (halo_x >= 0 && halo_x < width&& halo_y >= 0 && halo_y < height) ? input[halo_y * width + halo_x] : Padding_Entry;
	}
	// left bottom
	if (tx < kernel_radius || ty >= blockDim.y - kernel_radius) {
		int halo_x = ix - kernel_radius;
		int halo_y = iy + kernel_radius;
		shared_mem[(shared_y + kernel_radius) * shared_width + tx]
			= (halo_x >= 0 && halo_x < width&& halo_y >= 0 && halo_y < height) ? input[halo_y * width + halo_x] : Padding_Entry;
	}
	// right up
	if (tx >= blockDim.x - kernel_radius || ty < kernel_radius) {
		int halo_x = ix + kernel_radius;
		int halo_y = iy - kernel_radius;
		shared_mem[ty * shared_width + (shared_x + kernel_radius)]
			= (halo_x >= 0 && halo_x < width&& halo_y >= 0 && halo_y < height) ? input[halo_y * width + halo_x] : Padding_Entry;
	}
	// right bottom
	if (tx >= blockDim.x - kernel_radius || ty >= blockDim.y - kernel_radius) {
		int halo_x = ix + kernel_radius;
		int halo_y = iy + kernel_radius;
		shared_mem[(shared_y + kernel_radius) * shared_width + (shared_x + kernel_radius)]
			= (halo_x >= 0 && halo_x < width&& halo_y >= 0 && halo_y < height) ? input[halo_y * width + halo_x] : Padding_Entry;
	}

	//synchronize
	__syncthreads();

	/*
	if (blockIdx.x == 1 && blockIdx.y == 0 && threadIdx.x == 1 && threadIdx.y == 0) {
		for (int i = 0; i < shared_height; i++)
		{
			for (int j = 0; j < shared_width; j++)
			{
				int index = i * shared_width + j;
				printf("%d %d :%d \n", i, j, shared_mem[index]);
			}
			printf("\n");
		}

		if (ix < width && iy < height)
		{
			printf("%d %d %d %d\n", shared_y, shared_x, iy, ix);
			unsigned int sum = 0;
			for (int ky = -kernel_radius; ky <= kernel_radius; ky++)
				for (int kx = -kernel_radius; kx <= kernel_radius; kx++)
				{
					int tmp_sy= shared_y + ky;
					int tmp_sx = shared_x + kx;
					int shared_index = tmp_sy * shared_width + tmp_sx;
					printf("%d %d :%d -- %d, %d, %d \n", tmp_sy, tmp_sx, shared_mem[shared_index], ky + kernel_radius,
						kx + kernel_radius, d_kernel[(ky + kernel_radius) * kernel_width + (kx + kernel_radius)]);

				}
		}
	}
	*/

	//convolve
	if (ix < width && iy < height)
	{
		int sum = Bias;
		for (int ky = -kernel_radius; ky <= kernel_radius; ky++)
			for (int kx = -kernel_radius; kx <= kernel_radius; kx++)
			{
				int tmp_sy = shared_y + ky;
				int tmp_sx = shared_x + kx;
				int shared_index = tmp_sy * shared_width + tmp_sx;

				int kernel_index = (ky + kernel_radius) * kernel_width + (kx + kernel_radius);
				sum += shared_mem[shared_index] * d_kernel[kernel_index];
			}


		unsigned int out_index = iy * width + ix;
		output[out_index] = sum;
	}
}


int main()
{
	int* input, * output, * kernel, * golden;
	int* d_input, * d_output;
	int size = N * sizeof(int);
	int kernel_size = kernel_N * sizeof(int);

	cudaMalloc((void**)&d_input, size);
	cudaMalloc((void**)&d_output, size);
	cudaMalloc((void**)&d_kernel, kernel_size);


	input = (int*)malloc(size);
	output = (int*)malloc(size);
	kernel = (int*)malloc(kernel_size);

	golden = (int*)malloc(size);

	for (int i = 0; i < N; i++)
	{
		input[i] = i + 1;
		output[i] = 0;
		golden[i] = 9;
	}

	for (int i = 0; i < kernel_N; i++)
	{
		if (i % 2 == 0) {
			kernel[i] = 1;
		}
		else {
			kernel[i] = 0;
		}
	}

	printf("Golden Matrix: \n");
	int kernel_radius = KERNEL_WIDTH / 2;
	for (int i = 0; i < H; i++)
	{
		for (int j = 0; j < W; j++)
		{
			int ix = j;
			int iy = i;
			int sum = Bias;

			for (int ky = -kernel_radius; ky <= kernel_radius; ky++)
			{
				for (int kx = -kernel_radius; kx <= kernel_radius; kx++)
				{
					int tmp_ix = ix + kx;
					int tmp_iy = iy + ky;

					if (tmp_ix >= 0 && tmp_ix < W && tmp_iy >= 0 && tmp_iy < H)
					{
						int input_index = tmp_iy * W + tmp_ix;
						int kernel_index = (ky + kernel_radius) * KERNEL_WIDTH + (kx + kernel_radius);
						sum += input[input_index] * kernel[kernel_index];
					}
					else {
						int kernel_index = (ky + kernel_radius) * KERNEL_WIDTH + (kx + kernel_radius);
						sum += Padding_Entry * kernel[kernel_index];
					}
				}
			}
			golden[iy * W + ix] = sum;
			printf("%d ", golden[iy * W + ix]);
		}
		printf("\n");
	}

	printf("\n");
	printf("\n");

	cudaMemcpy(d_input, input, size, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(d_kernel, kernel, kernel_size);

	dim3 blockDim(BLOCK_DIM_X, BLOCK_DIM_Y);
	dim3 gridDim((W + blockDim.x - 1) / blockDim.x,
		(H + blockDim.y - 1) / blockDim.y);


	convolve2D << <gridDim, blockDim >> > (d_input, d_output, W, H, KERNEL_WIDTH, Padding_Entry);


	cudaMemcpy(output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);

	printf("Convolved Matrix: \n");
	for (int i = 0; i < H; i++)
	{
		for (int j = 0; j < W; j++)
		{
			int index = i * W + j;
			printf("%d ", output[index]);
		}
		printf("\n");
	}

	bool pass = true;
	for (int i = 0; i < N; i++) {
		if (golden[i] != output[i])
			pass = false;
	}

	if (pass)
		printf("PASS\n");
	else
		printf("FAIL\n");


	free(input);
	free(output);
	free(kernel);
	free(golden);
	cudaFree(d_input);
	cudaFree(d_output);
	cudaFree(d_kernel);

	return 0;
}
