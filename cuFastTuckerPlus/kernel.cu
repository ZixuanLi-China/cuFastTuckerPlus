#include <cublas_v2.h>
#include <mma.h>
#include "parameter.h"

using namespace nvcuda;

__global__ void Update_Parameter_A_SGD(const int order, const int core_kernel,
		const int core_dimen, type_of_data **parameter_a,
		type_of_data **parameter_b, const int nnz, const type_of_data *value,
		const int *index, const type_of_data learn_rate_a,
		const type_of_data lambda_a) {

	int worker = block_size / warp_size;
	int lane_id = threadIdx.x % warp_size;
	int row_id = lane_id / 16;
	int col_id = lane_id % 16;
	int warp_id = threadIdx.x / warp_size;
	int worker_id = worker * blockIdx.x + warp_id;
	int workers = worker * gridDim.x;

	__shared__ half shared_a[block_size / warp_size][register_size][register_size];
	__shared__ type_of_data shared_c_temp[block_size / warp_size][register_size][register_size];
	__shared__ type_of_data shared_c[block_size / warp_size][order_size][register_size][register_size];

	wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a;
	wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b[order_size];
	wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_T[order_size];
	wmma::fragment<wmma::accumulator, 16, 16, 16, float> c;

#pragma unroll
	for (int order_index = 0; order_index < order_size; order_index++) {
#pragma unroll
		for (int count_index = 0; count_index < 8; count_index++) {
			shared_a[warp_id][count_index * 2 + row_id][col_id] =
					__float2half(
							parameter_b[order_index][count_index * warp_size
									+ lane_id]);
		}
		__syncthreads();
		const half *b_ptr = &shared_a[warp_id][0][0];
		wmma::load_matrix_sync(b[order_index], b_ptr, register_size);
		wmma::load_matrix_sync(b_T[order_index], b_ptr, register_size);
	}

	for (int nnz_index = worker_id * 16; nnz_index < nnz;
			nnz_index += workers * 16) {

#pragma unroll
		for (int order_index = 0; order_index < order_size; order_index++) {
#pragma unroll
			for (int count_index = 0; count_index < 8; count_index++) {
				shared_c[warp_id][order_index][count_index * 2 + row_id][col_id] =
						1.0f;
			}
		}
		__syncthreads();

#pragma unroll
		for (int order_index = 0; order_index < order_size; order_index++) {

#pragma unroll
			for (int count_index = 0; count_index < 8; count_index++) {
				int ele_index = (nnz_index + 2 * count_index + row_id) % nnz;
				shared_a[warp_id][count_index * 2 + row_id][col_id] =
						__float2half(
								__ldg(
										&parameter_a[order_index][index[ele_index
												* order + order_index]
												* core_dimen + col_id]));
			}
			__syncthreads();

			const half *a_ptr = &shared_a[warp_id][0][0];
			wmma::load_matrix_sync(a, a_ptr, register_size);

			wmma::fill_fragment(c, 0.0f);
			wmma::mma_sync(c, a, b[order_index], c);

			float *c_ptr = &shared_c_temp[warp_id][0][0];
			wmma::store_matrix_sync(c_ptr, c, register_size,
					wmma::mem_row_major);

			__syncthreads();

#pragma unroll
			for (int inner_order_index = 0; inner_order_index < order_size;
					inner_order_index++) {
				if (inner_order_index != order_index) {
#pragma unroll
					for (int count_index = 0; count_index < 8; count_index++) {
						shared_c[warp_id][inner_order_index][count_index * 2
								+ row_id][col_id] *=
								shared_c_temp[warp_id][count_index * 2 + row_id][col_id];
					}
				}
			}
			__syncthreads();
		}

#pragma unroll
		for (int count_index = 0; count_index < 8; count_index++) {
			shared_c_temp[warp_id][count_index * 2 + row_id][col_id] *=
					shared_c[warp_id][order_size - 1][count_index * 2 + row_id][col_id];
		}
		__syncthreads();

#pragma unroll
		for (int order_index = 0; order_index < order_size; order_index++) {
#pragma unroll
			for (int count_index = 0; count_index < 8; count_index++) {
				shared_a[warp_id][count_index * 2 + row_id][col_id] =
						__float2half(
								shared_c[warp_id][order_index][count_index * 2
										+ row_id][col_id]);
			}
			__syncthreads();

			const half *a_ptr = &shared_a[warp_id][0][0];
			wmma::load_matrix_sync(a, a_ptr, register_size);

			wmma::fill_fragment(c, 0.0f);
			wmma::mma_sync(c, a, b_T[order_index], c);

			float *c_ptr = &shared_c[warp_id][order_index][0][0];
			wmma::store_matrix_sync(c_ptr, c, register_size,
					wmma::mem_row_major);

		}
		__syncthreads();
#pragma unroll
		for (int count_index = 0; count_index < 8; count_index++) {
			int ele_index = (nnz_index + 2 * count_index + row_id) % nnz;
			type_of_data x_r =
					shared_c_temp[warp_id][count_index * 2 + row_id][col_id];

			x_r += __shfl_down_sync(mask, x_r, 8);
			x_r += __shfl_down_sync(mask, x_r, 4);
			x_r += __shfl_down_sync(mask, x_r, 2);
			x_r += __shfl_down_sync(mask, x_r, 1);
			x_r = __shfl_sync(mask, x_r, 0, 16);

			x_r -= value[ele_index];

#pragma unroll
			for (int order_index = 0; order_index < order_size; order_index++) {

				parameter_a[order_index][index[ele_index * order + order_index]
						* core_dimen + col_id] -=
						learn_rate_a
								* (x_r
										* shared_c[warp_id][order_index][count_index
												* 2 + row_id][col_id]
										+ lambda_a
												* parameter_a[order_index][index[ele_index
														* order + order_index]
														* core_dimen + col_id]);
			}
		}

	}
}

__global__ void Update_Parameter_A_SGD_1(const int order, const int core_kernel,
		const int core_dimen, type_of_data **parameter_a,
		type_of_data **parameter_b, const int nnz, const type_of_data *value,
		const int *index, const type_of_data learn_rate_a,
		const type_of_data lambda_a) {

	int worker = block_size / warp_size;
	int lane_id = threadIdx.x % warp_size;
	int row_id = lane_id / 16;
	int col_id = lane_id % 16;
	int warp_id = threadIdx.x / warp_size;
	int worker_id = worker * blockIdx.x + warp_id;
	int workers = worker * gridDim.x;

	__shared__ half shared_a[block_size / warp_size][register_size][register_size];
	__shared__ type_of_data shared_c_temp[block_size / warp_size][register_size][register_size];
	__shared__ type_of_data shared_c[block_size / warp_size][order_size][register_size][register_size];

	wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a;
	wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b[order_size];
	wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_T[order_size];
	wmma::fragment<wmma::accumulator, 16, 16, 16, float> c;

#pragma unroll
	for (int order_index = 0; order_index < order_size; order_index++) {
#pragma unroll
		for (int count_index = 0; count_index < 8; count_index++) {
			shared_a[warp_id][count_index * 2 + row_id][col_id] =
					__float2half(
							parameter_b[order_index][count_index * warp_size
									+ lane_id]);
		}
		__syncthreads();
		const half *b_ptr = &shared_a[warp_id][0][0];
		wmma::load_matrix_sync(b[order_index], b_ptr, register_size);
		wmma::load_matrix_sync(b_T[order_index], b_ptr, register_size);
	}

	for (int nnz_index = worker_id * 16; nnz_index < nnz;
			nnz_index += workers * 16) {

#pragma unroll
		for (int order_index = 0; order_index < order_size; order_index++) {
#pragma unroll
			for (int count_index = 0; count_index < 8; count_index++) {
				shared_c[warp_id][order_index][count_index * 2 + row_id][col_id] =
						1.0f;
			}
		}
		__syncthreads();

#pragma unroll
		for (int order_index = 0; order_index < order_size; order_index++) {

#pragma unroll
			for (int count_index = 0; count_index < 8; count_index++) {
				int ele_index = (nnz_index + 2 * count_index + row_id) % nnz;
				shared_a[warp_id][count_index * 2 + row_id][col_id] =
						__float2half(
								__ldg(
										&parameter_a[order_index][index[ele_index
												* order + order_index]
												* core_dimen + col_id]));
			}
			__syncthreads();

			const half *a_ptr = &shared_a[warp_id][0][0];
			wmma::load_matrix_sync(a, a_ptr, register_size);

			wmma::fill_fragment(c, 0.0f);
			wmma::mma_sync(c, a, b[order_index], c);

			float *c_ptr = &shared_c_temp[warp_id][0][0];
			wmma::store_matrix_sync(c_ptr, c, register_size,
					wmma::mem_row_major);

			__syncthreads();

#pragma unroll
			for (int inner_order_index = 0; inner_order_index < order_size;
					inner_order_index++) {
				if (inner_order_index != order_index) {
#pragma unroll
					for (int count_index = 0; count_index < 8; count_index++) {
						shared_c[warp_id][inner_order_index][count_index * 2
								+ row_id][col_id] *=
								shared_c_temp[warp_id][count_index * 2 + row_id][col_id];
					}
				}
			}
			__syncthreads();
		}

#pragma unroll
		for (int count_index = 0; count_index < 8; count_index++) {
			shared_c_temp[warp_id][count_index * 2 + row_id][col_id] *=
					shared_c[warp_id][order_size - 1][count_index * 2 + row_id][col_id];
		}
		__syncthreads();

#pragma unroll
		for (int order_index = 0; order_index < order_size; order_index++) {
#pragma unroll
			for (int count_index = 0; count_index < 8; count_index++) {
				shared_a[warp_id][count_index * 2 + row_id][col_id] =
						__float2half(
								shared_c[warp_id][order_index][count_index * 2
										+ row_id][col_id]);
			}
			__syncthreads();

			const half *a_ptr = &shared_a[warp_id][0][0];
			wmma::load_matrix_sync(a, a_ptr, register_size);

			wmma::fill_fragment(c, 0.0f);
			wmma::mma_sync(c, a, b_T[order_index], c);

			float *c_ptr = &shared_c[warp_id][order_index][0][0];
			wmma::store_matrix_sync(c_ptr, c, register_size,
					wmma::mem_row_major);

		}
		__syncthreads();
#pragma unroll
		for (int count_index = 0; count_index < 8; count_index++) {
			int ele_index = (nnz_index + 2 * count_index + row_id) % nnz;
			type_of_data x_r =
					shared_c_temp[warp_id][count_index * 2 + row_id][col_id];

			x_r += __shfl_down_sync(mask, x_r, 8);
			x_r += __shfl_down_sync(mask, x_r, 4);
			x_r += __shfl_down_sync(mask, x_r, 2);
			x_r += __shfl_down_sync(mask, x_r, 1);
			x_r = __shfl_sync(mask, x_r, 0, 16);

			x_r -= value[ele_index];

#pragma unroll
			for (int order_index = 0; order_index < order_size; order_index++) {

				parameter_a[order_index][index[ele_index * order + order_index]
						* core_dimen + col_id] -=
						learn_rate_a
								* (x_r
										* shared_c[warp_id][order_index][count_index
												* 2 + row_id][col_id]
										+ lambda_a
												* parameter_a[order_index][index[ele_index
														* order + order_index]
														* core_dimen + col_id]);
			}
		}

	}
}

void Update_Parameter_A(const int order, const int core_kernel,
		const int core_dimen, type_of_data **parameter_a_device,
		type_of_data **parameter_b_device, const int nnz_train,
		type_of_data **value_train_device, int **index_train_device,
		type_of_data learn_rate_a, type_of_data lambda_a) {

	int data_per_part = nnz_train / data_part + 1;

	for (int i = 0; i < data_part - 1; i++) {
		Update_Parameter_A_SGD <<<grid_size, block_size>>>(order, core_kernel,
				core_dimen, parameter_a_device, parameter_b_device,
				data_per_part, value_train_device[i], index_train_device[i],
				learn_rate_a, lambda_a);
		cudaDeviceSynchronize();
	}
	Update_Parameter_A_SGD <<<grid_size,
	block_size>>>(order, core_kernel, core_dimen, parameter_a_device,
			parameter_b_device, nnz_train - (data_part - 1) * data_per_part,
			value_train_device[data_part - 1],
			index_train_device[data_part - 1], learn_rate_a, lambda_a);
	cudaDeviceSynchronize();

}

__global__ void Update_Parameter_B_SGD_Gradient(const int order,
		const int core_kernel, const int core_dimen, type_of_data **parameter_a,
		type_of_data **parameter_b, const int nnz, const type_of_data *value,
		const int *index, type_of_data *b_sum) {

	int worker = block_size / warp_size;
	int lane_id = threadIdx.x % warp_size;
	int row_id = lane_id / 16;
	int col_id = lane_id % 16;
	int warp_id = threadIdx.x / warp_size;
	int worker_id = worker * blockIdx.x + warp_id;
	int workers = worker * gridDim.x;

	__shared__ half shared_a[block_size / warp_size][order_size][register_size][register_size];
	__shared__ type_of_data shared_c_temp[block_size / warp_size][register_size][register_size];
	__shared__ type_of_data shared_c[block_size / warp_size][order_size][register_size][register_size];

	type_of_data b_sum_temp[order_size][register_size];

	type_of_data p_a_gs[8];

	wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_1;
	wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_1[order_size];
	wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::col_major> a;
	wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b;
	wmma::fragment<wmma::accumulator, 16, 16, 16, float> c;

#pragma unroll
	for (int order_index = 0; order_index < order_size; order_index++) {
#pragma unroll
		for (int count_index = 0; count_index < 8; count_index++) {
			shared_a[warp_id][order_index][count_index * 2 + row_id][col_id] =
					__float2half(
							parameter_b[order_index][count_index * warp_size
									+ lane_id]);
			b_sum_temp[order_index][count_index * 2 + row_id] = 0.0f;
		}
		__syncthreads();
		const half *b_ptr = &shared_a[warp_id][order_index][0][0];
		wmma::load_matrix_sync(b_1[order_index], b_ptr, register_size);
	}

	for (int nnz_index = worker_id * 16; nnz_index < nnz;
			nnz_index += workers * 16) {

#pragma unroll
		for (int order_index = 0; order_index < order_size; order_index++) {
#pragma unroll
			for (int count_index = 0; count_index < 8; count_index++) {
				shared_c[warp_id][order_index][count_index * 2 + row_id][col_id] =
						1.0f;
			}
		}
		__syncthreads();

#pragma unroll
		for (int order_index = 0; order_index < order_size; order_index++) {

#pragma unroll
			for (int count_index = 0; count_index < 8; count_index++) {
				int ele_index = (nnz_index + 2 * count_index + row_id) % nnz;
				shared_a[warp_id][order_index][count_index * 2 + row_id][col_id] =
						__float2half(
								__ldg(
										&parameter_a[order_index][index[ele_index
												* order + order_index]
												* core_dimen + col_id]));
			}
			__syncthreads();

			const half *a_ptr = &shared_a[warp_id][order_index][0][0];
			wmma::load_matrix_sync(a_1, a_ptr, register_size);

			wmma::fill_fragment(c, 0.0f);
			wmma::mma_sync(c, a_1, b_1[order_index], c);

			float *c_ptr = &shared_c_temp[warp_id][0][0];
			wmma::store_matrix_sync(c_ptr, c, register_size,
					wmma::mem_row_major);

			__syncthreads();

#pragma unroll
			for (int inner_order_index = 0; inner_order_index < order_size;
					inner_order_index++) {
				if (inner_order_index != order_index) {
#pragma unroll
					for (int count_index = 0; count_index < 8; count_index++) {
						shared_c[warp_id][inner_order_index][count_index * 2
								+ row_id][col_id] *=
								shared_c_temp[warp_id][count_index * 2 + row_id][col_id];
					}
				}
			}
			__syncthreads();
		}

#pragma unroll
		for (int count_index = 0; count_index < 8; count_index++) {
			int ele_index = (nnz_index + 2 * count_index + row_id) % nnz;
			p_a_gs[count_index] = shared_c[warp_id][order_size - 1][count_index
					* 2 + row_id][col_id]
					* shared_c_temp[warp_id][count_index * 2 + row_id][col_id];

			p_a_gs[count_index] += __shfl_down_sync(mask, p_a_gs[count_index],
					8);
			p_a_gs[count_index] += __shfl_down_sync(mask, p_a_gs[count_index],
					4);
			p_a_gs[count_index] += __shfl_down_sync(mask, p_a_gs[count_index],
					2);
			p_a_gs[count_index] += __shfl_down_sync(mask, p_a_gs[count_index],
					1);
			p_a_gs[count_index] = __shfl_sync(mask, p_a_gs[count_index], 0, 16);

			p_a_gs[count_index] -= value[ele_index];

		}
		__syncthreads();

#pragma unroll
		for (int order_index = 0; order_index < order_size; order_index++) {

#pragma unroll
			for (int count_index = 0; count_index < 8; count_index++) {
				shared_a[warp_id][order_index][count_index * 2 + row_id][col_id] *=
						__float2half(p_a_gs[count_index]);
			}
			__syncthreads();

			const half *b_ptr = &shared_a[warp_id][order_index][0][0];
			wmma::load_matrix_sync(b, b_ptr, register_size);

#pragma unroll
			for (int count_index = 0; count_index < 8; count_index++) {
				shared_a[warp_id][order_index][count_index * 2 + row_id][col_id] =
						__float2half(
								shared_c[warp_id][order_index][count_index * 2
										+ row_id][col_id]);
			}
			__syncthreads();

			const half *a_ptr = &shared_a[warp_id][order_index][0][0];
			wmma::load_matrix_sync(a, a_ptr, register_size);

			wmma::fill_fragment(c, 0.0f);
			wmma::mma_sync(c, a, b, c);

			float *c_ptr = &shared_c[warp_id][order_index][0][0];
			wmma::store_matrix_sync(c_ptr, c, register_size,
					wmma::mem_row_major);

			__syncthreads();

#pragma unroll
			for (int count_index = 0; count_index < 8; count_index++) {
				b_sum_temp[order_index][count_index * 2 + row_id] +=
						shared_c[warp_id][order_index][count_index * 2 + row_id][col_id];
			}
			__syncthreads();
		}
	}

#pragma unroll
	for (int order_index = 0; order_index < order_size; order_index++) {
#pragma unroll
		for (int count_index = 0; count_index < 8; count_index++) {
			atomicAdd(
					&b_sum[(worker_id % sum_size) * order * core_kernel
							* core_dimen
							+ order_index * core_kernel * core_dimen
							+ (count_index * 2 + row_id) * core_dimen + col_id],
					b_sum_temp[order_index][count_index * 2 + row_id]);
		}
	}
	__syncthreads();
}

__global__ void Parameter_B_Gradient_Sum(const int order, const int core_kernel,
		const int core_dimen, const int nnz,
		type_of_data *b_sum, type_of_data *b_grad) {

	int core = core_dimen;
	int worker = block_size / core;
	int lane_id = threadIdx.x % core;
	int local_id = threadIdx.x / core;
	int worker_id = worker * blockIdx.x + local_id;
	int workers = worker * gridDim.x;

	for (int b_index = worker_id; b_index < order * core_kernel; b_index +=
			workers) {
		for (int sum_size_index = 0; sum_size_index < sum_size;
				sum_size_index++) {
			b_grad[b_index * core_dimen + lane_id] += b_sum[b_index * core_dimen
					+ lane_id];
		}
		b_grad[b_index * core_dimen + lane_id] /= nnz;
	}
}

__global__ void Update_Parameter_B(const int order, const int core_kernel,
		const int core_dimen,
		type_of_data **parameter_b, type_of_data *b_grad,
		const type_of_data learn_rate_b, const type_of_data lambda_b) {

	int core = core_dimen;
	int worker = block_size / core;
	int lane_id = threadIdx.x % core;
	int local_id = threadIdx.x / core;
	int worker_id = worker * blockIdx.x + local_id;
	int workers = worker * gridDim.x;

	for (int b_index = worker_id; b_index < order * core_kernel; b_index +=
			workers) {

		int order_index = b_index / core_kernel;
		int core_kernel_index = b_index % core_kernel;
		parameter_b[order_index][core_kernel_index * core_kernel + lane_id] -=
				learn_rate_b
						* (b_grad[b_index * core_dimen + lane_id]
								+ lambda_b
										* parameter_b[order_index][core_kernel_index
												* core_kernel + lane_id]);
	}
}

void Update_Parameter_B_Batch(const int order, int *dimen,
		const int core_kernel, const int core_dimen,
		type_of_data **parameter_a,
		type_of_data **parameter_b, const int nnz,
		type_of_data **value, int **index, const type_of_data learn_rate_b,
		const type_of_data lambda_b) {

	type_of_data *b_sum;
	type_of_data *b_grad;

	cudaMalloc((void**) &b_sum,
	sum_size * order * core_kernel * core_dimen * sizeof(type_of_data));
	cudaMalloc((void**) &b_grad,
			order * core_kernel * core_dimen * sizeof(type_of_data));
	cudaMemset(b_sum, 0,
	sum_size * order * core_kernel * core_dimen * sizeof(type_of_data));
	cudaMemset(b_grad, 0,
			order * core_kernel * core_dimen * sizeof(type_of_data));

	int data_per_part = nnz / data_part + 1;

	for (int i = 0; i < data_part - 1; i++) {
		Update_Parameter_B_SGD_Gradient<<<grid_size,
		block_size>>>( order, core_kernel, core_dimen, parameter_a, parameter_b,
				data_per_part, value[i], index[i], b_sum);
		cudaDeviceSynchronize();
	}
	Update_Parameter_B_SGD_Gradient<<<grid_size,
	block_size>>>( order, core_kernel, core_dimen, parameter_a, parameter_b,
			nnz - (data_part - 1) * data_per_part, value[data_part - 1],
			index[data_part - 1], b_sum);
	cudaDeviceSynchronize();

	Parameter_B_Gradient_Sum <<<
	order * core_kernel / (block_size / core_dimen) + 1, block_size>>>(
			order, core_kernel, core_dimen, nnz, b_sum, b_grad);
	cudaDeviceSynchronize();

	Update_Parameter_B <<<order * core_kernel / (block_size / core_dimen) + 1,
	block_size>>>(order, core_kernel, core_dimen, parameter_b, b_grad,
			learn_rate_b, lambda_b);
	cudaDeviceSynchronize();

	cudaFree(b_sum);
	cudaFree(b_grad);

}

__global__ void RMSE_AND_MAE(const int order, const int core_kernel,
		const int core_dimen,
		type_of_data **parameter_a, type_of_data **parameter_b, const int nnz,
		const type_of_data *value, const int *index, type_of_data *rmse,
		type_of_data *mae) {

	int core = core_dimen;
	int worker = block_size / core;
	int lane_id = threadIdx.x % core;
	int local_id = threadIdx.x / core;
	int worker_id = worker * blockIdx.x + local_id;
	int workers = worker * gridDim.x;

	for (int nnz_index = worker_id; nnz_index < nnz; nnz_index += workers) {
		type_of_data p_a_gs = 0.0;
		type_of_data gs = 0.0;

		for (int core_kernel_index = 0; core_kernel_index < core_kernel;
				core_kernel_index++) {
			type_of_data gs_temp = parameter_b[0][core_kernel_index * core_dimen
					+ lane_id];

			for (int inner_order_index = 0; inner_order_index < order;
					inner_order_index++) {
				if (inner_order_index != 0) {
					type_of_data temp =
							parameter_a[inner_order_index][index[nnz_index
									* order + inner_order_index] * core_dimen
									+ lane_id]
									* parameter_b[inner_order_index][core_kernel_index
											* core_dimen + lane_id];

					temp += __shfl_down_sync(mask, temp, 8);
					temp += __shfl_down_sync(mask, temp, 4);
					temp += __shfl_down_sync(mask, temp, 2);
					temp += __shfl_down_sync(mask, temp, 1);
					temp = __shfl_sync(mask, temp, 0, 16);

					gs_temp *= temp;

				}
			}
			gs += gs_temp;
		}

		p_a_gs = parameter_a[0][index[nnz_index * order] * core_dimen + lane_id]
				* gs;

		p_a_gs += __shfl_down_sync(mask, p_a_gs, 8);
		p_a_gs += __shfl_down_sync(mask, p_a_gs, 4);
		p_a_gs += __shfl_down_sync(mask, p_a_gs, 2);
		p_a_gs += __shfl_down_sync(mask, p_a_gs, 1);
		p_a_gs = __shfl_sync(mask, p_a_gs, 0, 16);

		p_a_gs -= value[nnz_index];

		if (lane_id == 0) {
			atomicAdd(&rmse[nnz_index % error_size], p_a_gs * p_a_gs);
			atomicAdd(&mae[nnz_index % error_size], abs(p_a_gs));
		}

	}

}

void GET_RMSE_AND_MAE(const int order, const int core_kernel,
		const int core_dimen,
		type_of_data **parameter_a, type_of_data **parameter_b, const int nnz,
		type_of_data **value, int **index, type_of_data *rmse,
		type_of_data *mae) {

	type_of_data *errors_rmse;
	type_of_data *errors_mae;
	cublasHandle_t handle_rmse;
	cublasCreate(&handle_rmse);
	cublasHandle_t handle_mae;
	cublasCreate(&handle_mae);
	cudaMalloc((void**) &errors_rmse,
	error_size * sizeof(type_of_data));
	cudaMalloc((void**) &errors_mae,
	error_size * sizeof(type_of_data));
	cudaMemset(errors_rmse, 0,
	error_size * sizeof(type_of_data));
	cudaMemset(errors_mae, 0,
	error_size * sizeof(type_of_data));

	int data_per_part = nnz / data_part + 1;
	for (int i = 0; i < data_part - 1; i++) {
		RMSE_AND_MAE <<<data_per_part / block_size + 1, block_size>>>(order,
				core_kernel, core_dimen, parameter_a, parameter_b,
				data_per_part, value[i], index[i], errors_rmse, errors_mae);
		cudaDeviceSynchronize();
	}
	RMSE_AND_MAE <<<data_per_part / block_size + 1, block_size>>>(order,
			core_kernel, core_dimen, parameter_a, parameter_b,
			nnz - (data_part - 1) * data_per_part, value[data_part - 1],
			index[data_part - 1], errors_rmse, errors_mae);
	cudaDeviceSynchronize();

	type_of_data *rmse_sum = (type_of_data*) malloc(sizeof(type_of_data));
	type_of_data *mae_sum = (type_of_data*) malloc(sizeof(type_of_data));

	cublasSasum(handle_rmse, error_size, errors_rmse, 1, rmse_sum);
	cudaDeviceSynchronize();
	cublasSasum(handle_mae, error_size, errors_mae, 1, mae_sum);
	cudaDeviceSynchronize();

	*rmse = sqrt((*rmse_sum) / nnz);
	*mae = (*mae_sum) / nnz;
	cudaFree(errors_rmse);
	cudaFree(errors_mae);
	cublasDestroy(handle_rmse);
	cublasDestroy(handle_mae);
	free(rmse_sum);
	free(mae_sum);

}

void GET_RMSE_AND_MAE(const int order, const int core_kernel,
		const int core_dimen,
		type_of_data **parameter_a, type_of_data **parameter_b, const int nnz,
		type_of_data *value, int *index,
		type_of_data *rmse,
		type_of_data *mae) {

	type_of_data *errors_rmse;
	type_of_data *errors_mae;
	cublasHandle_t handle_rmse;
	cublasCreate(&handle_rmse);
	cublasHandle_t handle_mae;
	cublasCreate(&handle_mae);
	cudaMalloc((void**) &errors_rmse,
	error_size * sizeof(type_of_data));
	cudaMalloc((void**) &errors_mae,
	error_size * sizeof(type_of_data));
	cudaMemset(errors_rmse, 0,
	error_size * sizeof(type_of_data));
	cudaMemset(errors_mae, 0,
	error_size * sizeof(type_of_data));

	RMSE_AND_MAE <<<nnz / block_size + 1, block_size>>>(order, core_kernel,
			core_dimen, parameter_a, parameter_b, nnz, value, index,
			errors_rmse, errors_mae);
	cudaDeviceSynchronize();

	type_of_data *rmse_sum = (type_of_data*) malloc(sizeof(type_of_data));
	type_of_data *mae_sum = (type_of_data*) malloc(sizeof(type_of_data));

	cublasSasum(handle_rmse, error_size, errors_rmse, 1, rmse_sum);
	cudaDeviceSynchronize();
	cublasSasum(handle_mae, error_size, errors_mae, 1, mae_sum);
	cudaDeviceSynchronize();

	*rmse = sqrt((*rmse_sum) / nnz);
	*mae = (*mae_sum) / nnz;
	cudaFree(errors_rmse);
	cudaFree(errors_mae);
	cublasDestroy(handle_rmse);
	cublasDestroy(handle_mae);
	free(rmse_sum);
	free(mae_sum);

}
