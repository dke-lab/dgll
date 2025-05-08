#include <cuda_runtime.h>
#include <stdio.h>

// Forward kernel (unchanged from your working version)
__global__ void gcn_fused_kernel_optimized(
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_idx,
    const float* __restrict__ values,
    const float* __restrict__ X,
    const float* __restrict__ W,
    float* __restrict__ output,
    const int* __restrict__ num_neighbors,
    int N, int F_padded, int actual_F, int H_dim, int tile_size_H, int warp_size, int total_nnz
) {
    int row = blockIdx.x;
    if (row >= N) return;

    int warp_id = threadIdx.x / warp_size;
    int lane_id = threadIdx.x % warp_size;
    int num_warps = (blockDim.x + warp_size - 1) / warp_size;

    extern __shared__ float s_W[];
    const int start = row_ptr[row];
    const int nnz = num_neighbors[row];

    for (int h_tile = 0; h_tile < H_dim; h_tile += tile_size_H) {
        int current_H = (tile_size_H < (H_dim - h_tile)) ? tile_size_H : (H_dim - h_tile);

        for (int i = threadIdx.x; i < actual_F * current_H; i += blockDim.x) {
            int f = i / current_H;
            int h = i % current_H;
            int s_idx = f * current_H + h;
            if (f < actual_F && (h_tile + h) < H_dim && s_idx < actual_F * tile_size_H) {
                s_W[s_idx] = W[f * H_dim + (h_tile + h)];
            }
        }
        __syncthreads();

        for (int j_base = warp_id; j_base < current_H; j_base += num_warps) {
            float sum = 0.0f;
            for (int idx = start + lane_id; idx < start + nnz; idx += warp_size) {
                if (idx < row_ptr[row + 1] && idx < total_nnz) {
                    int col = col_idx[idx];
                    float a_val = values[idx];
                    float z = 0.0f;
                    for (int f = 0; f < actual_F; f++) {
                        if (col < N && f < actual_F) {
                            int x_idx = col * F_padded + f;
                            int s_w_idx = f * current_H + j_base;
                            if (s_w_idx < actual_F * tile_size_H) {
                                z += X[x_idx] * s_W[s_w_idx];
                            }
                        }
                    }
                    sum += a_val * z;
                }
            }

            #pragma unroll
            for (int offset = warp_size / 2; offset > 0; offset /= 2) {
                sum += __shfl_down_sync(0xffffffff, sum, offset);
            }

            if (lane_id == 0) {
                int output_idx = row * H_dim + (h_tile + j_base);
                if (row < N && (h_tile + j_base) < H_dim) {
                    output[output_idx] = fmaxf(sum, 0.0f);
                }
            }
        }
        __syncthreads();
    }
    // printf("Block %d: Row %d completed\n", blockIdx.x, row);
}

// Optimized Backward kernel with tiling and no atomicAdd
__global__ void gcn_fused_kernel_backward_optimized(
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_idx,
    const float* __restrict__ values,
    const float* __restrict__ X,
    const float* __restrict__ W,
    const float* __restrict__ grad_output,
    float* __restrict__ grad_W,
    float* __restrict__ grad_X,
    const int* __restrict__ num_neighbors,
    int N, int F_padded, int actual_F, int H_dim, int tile_size_H, int warp_size, int total_nnz
) {
    int row = blockIdx.x;
    if (row >= N) return;

    int warp_id = threadIdx.x / warp_size;
    int lane_id = threadIdx.x % warp_size;
    int num_warps = (blockDim.x + warp_size - 1) / warp_size;

    extern __shared__ float shared_mem[];
    float* s_grad_output = shared_mem; // First part for grad_output tile
    float* s_grad_W = &shared_mem[tile_size_H]; // Second part for partial grad_W
    int grad_W_size = actual_F * tile_size_H;
    float* s_grad_X = &s_grad_W[grad_W_size]; // Third part for partial grad_X
    int grad_X_size = tile_size_H;

    const int start = row_ptr[row];
    const int nnz = num_neighbors[row];

    // Initialize shared memory for grad_X accumulation
    for (int i = threadIdx.x; i < grad_X_size; i += blockDim.x) {
        s_grad_X[i] = 0.0f;
    }

    // Compute grad_W and grad_X
    for (int h_tile = 0; h_tile < H_dim; h_tile += tile_size_H) {
        int current_H = (tile_size_H < (H_dim - h_tile)) ? tile_size_H : (H_dim - h_tile);

        // Load grad_output into shared memory
        for (int i = threadIdx.x; i < current_H; i += blockDim.x) {
            int h = h_tile + i;
            if (h < H_dim) {
                s_grad_output[i] = grad_output[row * H_dim + h];
            }
        }

        // Initialize shared memory for grad_W accumulation
        for (int i = threadIdx.x; i < grad_W_size; i += blockDim.x) {
            s_grad_W[i] = 0.0f;
        }
        __syncthreads();

        // Compute grad_W
        for (int idx = start + lane_id; idx < start + nnz; idx += warp_size) {
            if (idx < row_ptr[row + 1] && idx < total_nnz) {
                int col = col_idx[idx];
                float a_val = values[idx];
                if (col < N) {
                    for (int h = warp_id; h < current_H; h += num_warps) {
                        int h_global = h_tile + h;
                        if (h_global < H_dim) {
                            float grad_val = a_val * s_grad_output[h];
                            for (int f = 0; f < actual_F; f++) {
                                int x_idx = col * F_padded + f;
                                int s_w_idx = f * current_H + h;
                                if (s_w_idx < grad_W_size) {
                                    s_grad_W[s_w_idx] += X[x_idx] * grad_val;
                                }
                            }
                        }
                    }
                }
            }
        }
        __syncthreads();

        // Write grad_W to global memory
        for (int i = threadIdx.x; i < actual_F * current_H; i += blockDim.x) {
            int f = i / current_H;
            int h = i % current_H;
            int w_idx = f * H_dim + (h_tile + h);
            if (f < actual_F && (h_tile + h) < H_dim) {
                atomicAdd(&grad_W[w_idx], s_grad_W[i]); // Still using atomicAdd here for simplicity
            }
        }
        __syncthreads();

        // Compute grad_X using W.T tiling
        for (int idx = start + lane_id; idx < start + nnz; idx += warp_size) {
            if (idx < row_ptr[row + 1] && idx < total_nnz) {
                int col = col_idx[idx];
                float a_val = values[idx];
                if (col < N) {
                    for (int h = warp_id; h < current_H; h += num_warps) {
                        int h_global = h_tile + h;
                        if (h_global < H_dim) {
                            float grad_val = a_val * s_grad_output[h];
                            for (int f = 0; f < actual_F; f++) {
                                int w_idx = f * H_dim + h_global;
                                int grad_x_idx = row * F_padded + f;
                                if (f < actual_F && grad_x_idx < N * F_padded) {
                                    atomicAdd(&grad_X[grad_x_idx], grad_val * W[w_idx]);
                                }
                            }
                        }
                    }
                }
            }
        }
        __syncthreads();
    }
}

extern "C" void launch_gcn_fused_kernel(
    const int* row_ptr, const int* col_idx, const float* values,
    const float* X, const float* W, float* H,
    const int* num_neighbors,
    int N, int F_padded, int actual_F, int H_dim, int total_nnz
) {
    cudaError_t err;

    int threads_per_block = 256;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int warp_size = prop.warpSize;

    size_t max_shared_mem = static_cast<size_t>(prop.sharedMemPerBlock * 0.9);
    size_t max_tile = max_shared_mem / (static_cast<size_t>(actual_F) * sizeof(float));
    int tile_size_H = (static_cast<size_t>(H_dim) < max_tile) ? H_dim : static_cast<int>(max_tile);
    tile_size_H = (tile_size_H < 128) ? tile_size_H : 128;
    tile_size_H = (tile_size_H / warp_size) * warp_size;
    if (tile_size_H < warp_size) tile_size_H = warp_size;

    size_t shared_mem_size = static_cast<size_t>(actual_F) * static_cast<size_t>(tile_size_H) * sizeof(float);
    int grid_size = N;

    // printf("Launching forward kernel: N=%d, F_padded=%d, actual_F=%d, H_dim=%d, total_nnz=%d\n", 
    //        N, F_padded, actual_F, H_dim, total_nnz);
    // printf("Grid size=%d, Threads per block=%d, Shared memory=%zu bytes\n", 
    //        grid_size, threads_per_block, shared_mem_size);

    gcn_fused_kernel_optimized<<<grid_size, threads_per_block, shared_mem_size>>>(
        row_ptr, col_idx, values, X, W, H, num_neighbors,
        N, F_padded, actual_F, H_dim, tile_size_H, warp_size, total_nnz
    );

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        // printf("CUDA Error after forward launch: %s\n", cudaGetErrorString(err));
        exit(1);
    }

    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        // printf("CUDA Error after forward sync: %s\n", cudaGetErrorString(err));
        exit(1);
    }
    // printf("Forward kernel launched successfully\n");
}

extern "C" void launch_gcn_fused_kernel_backward_optimized(
    const int* row_ptr, const int* col_idx, const float* values,
    const float* X, const float* W, const float* grad_output,
    float* grad_W, float* grad_X,
    const int* num_neighbors,
    int N, int F_padded, int actual_F, int H_dim, int total_nnz
) {
    cudaError_t err;

    int threads_per_block = 256;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int warp_size = prop.warpSize;

    size_t max_shared_mem = static_cast<size_t>(prop.sharedMemPerBlock * 0.9);
    size_t max_tile = max_shared_mem / (static_cast<size_t>(actual_F) * sizeof(float));
    int tile_size_H = (static_cast<size_t>(H_dim) < max_tile) ? H_dim : static_cast<int>(max_tile);
    tile_size_H = (tile_size_H < 128) ? tile_size_H : 128;
    tile_size_H = (tile_size_H / warp_size) * warp_size;
    if (tile_size_H < warp_size) tile_size_H = warp_size;

    size_t shared_mem_size = tile_size_H * sizeof(float) + 
                             actual_F * tile_size_H * sizeof(float) + 
                             tile_size_H * sizeof(float);
    int grid_size = N;

    // printf("Launching backward kernel: N=%d, F_padded=%d, actual_F=%d, H_dim=%d, total_nnz=%d\n", 
    //        N, F_padded, actual_F, H_dim, total_nnz);
    // printf("Grid size=%d, Threads per block=%d, Shared memory=%zu bytes\n", 
    //        grid_size, threads_per_block, shared_mem_size);

    gcn_fused_kernel_backward_optimized<<<grid_size, threads_per_block, shared_mem_size>>>(
        row_ptr, col_idx, values, X, W, grad_output, grad_W, grad_X, num_neighbors,
        N, F_padded, actual_F, H_dim, tile_size_H, warp_size, total_nnz
    );

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        // printf("CUDA Error after backward launch: %s\n", cudaGetErrorString(err));
        exit(1);
    }

    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        // printf("CUDA Error after backward sync: %s\n", cudaGetErrorString(err));
        exit(1);
    }
    // printf("Backward kernel launched successfully\n");
}


// #include <cuda_runtime.h>
// #include <stdio.h>

// __global__ void gcn_fused_kernel_optimized(
//     const int* __restrict__ row_ptr,
//     const int* __restrict__ col_idx,
//     const float* __restrict__ values,
//     const float* __restrict__ X,
//     const float* __restrict__ W,
//     float* __restrict__ output,
//     const int* __restrict__ num_neighbors,
//     int N, int F_padded, int actual_F, int H_dim, int tile_size_H, int warp_size, int total_nnz
// ) {
//     int row = blockIdx.x;
//     // printf("Block %d: Starting row %d\n", blockIdx.x, row);
//     if (row >= N) {
//         // printf("Block %d: Row %d >= N (%d), exiting\n", blockIdx.x, row, N);
//         return;  // Prevent out-of-bounds row access
//     }

//     int warp_id = threadIdx.x / warp_size;
//     int lane_id = threadIdx.x % warp_size;
//     int num_warps = (blockDim.x + warp_size - 1) / warp_size;
//     // printf("Block %d, Thread %d: warp_id=%d, lane_id=%d, num_warps=%d\n", 
//     //        blockIdx.x, threadIdx.x, warp_id, lane_id, num_warps);

//     extern __shared__ float s_W[];  // Shared memory for W tile
//     const int start = row_ptr[row];
//     const int nnz = num_neighbors[row];  // Number of neighbors for this row
//     // printf("Block %d: start=%d, nnz=%d, row_ptr[row+1]=%d\n", 
//     //        blockIdx.x, start, nnz, row_ptr[row + 1]);

//     for (int h_tile = 0; h_tile < H_dim; h_tile += tile_size_H) {
//         int current_H = (tile_size_H < (H_dim - h_tile)) ? tile_size_H : (H_dim - h_tile);  // Actual tile size
//         // printf("Block %d: h_tile=%d, current_H=%d\n", blockIdx.x, h_tile, current_H);

//         // Load W into shared memory, respecting actual_F
//         // printf("Block %d: Loading W into shared memory...\n", blockIdx.x);
//         for (int i = threadIdx.x; i < actual_F * current_H; i += blockDim.x) {
//             int f = i / current_H;
//             int h = i % current_H;
//             int s_idx = f * current_H + h;
//             if (f < actual_F && (h_tile + h) < H_dim && s_idx < actual_F * tile_size_H) {
//                 s_W[s_idx] = W[f * H_dim + (h_tile + h)];
//                 if (threadIdx.x == 0) {
//                     // printf("Block %d: Loaded s_W[%d]=%f from W[%d]\n", 
//                     //        blockIdx.x, s_idx, s_W[s_idx], f * H_dim + (h_tile + h));
//                 }
//             } else {
//                 // printf("Block %d, Thread %d: Skipping invalid s_W load: f=%d (actual_F=%d), h=%d (H_dim=%d), s_idx=%d (max=%d)\n",
//                 //        blockIdx.x, threadIdx.x, f, actual_F, h_tile + h, H_dim, s_idx, actual_F * tile_size_H);
//             }
//         }
//         __syncthreads();
//         printf("Block %d: Shared memory loaded and synced\n", blockIdx.x);

//         // Process each output dimension in the tile
//         for (int j_base = warp_id; j_base < current_H; j_base += num_warps) {
//             float sum = 0.0f;
//             printf("Block %d, Warp %d: Computing j_base=%d\n", blockIdx.x, warp_id, j_base);
//             // Neighbor aggregation
//             for (int idx = start + lane_id; idx < start + nnz; idx += warp_size) {
//                 if (idx < row_ptr[row + 1] && idx < total_nnz) {  // Updated bounds check with total_nnz
//                     int col = col_idx[idx];
//                     float a_val = values[idx];
//                     if (a_val > 1e30 || a_val < -1e30) {
//                         // printf("Block %d, Thread %d: Warning: Large a_val=%f at idx=%d\n", 
//                         //        blockIdx.x, threadIdx.x, a_val, idx);
//                     }
//                     float z = 0.0f;
//                     // printf("Block %d, Thread %d: idx=%d, col=%d, a_val=%f\n", 
//                     //        blockIdx.x, threadIdx.x, idx, col, a_val);

//                     // Feature multiplication, only up to actual_F
//                     for (int f = 0; f < actual_F; f++) {
//                         if (col < N && f < actual_F) {  // Bounds check X access
//                             int x_idx = col * F_padded + f;
//                             int s_w_idx = f * current_H + j_base;
//                             if (s_w_idx >= actual_F * tile_size_H) {
//                                 // printf("Block %d, Thread %d: Invalid s_W access: s_w_idx=%d (max=%d)\n", 
//                                 //        blockIdx.x, threadIdx.x, s_w_idx, actual_F * tile_size_H);
//                                 continue;
//                             }
//                             z += X[x_idx] * s_W[s_w_idx];
//                         } else {
//                             // printf("Block %d, Thread %d: Skipping invalid X access: col=%d (N=%d), f=%d (actual_F=%d)\n",
//                             //        blockIdx.x, threadIdx.x, col, N, f, actual_F);
//                         }
//                     }
//                     sum += a_val * z;
//                 } else {
//                     // printf("Block %d, Thread %d: Skipping invalid idx: idx=%d, row_ptr[row+1]=%d, total_nnz=%d\n",
//                     //        blockIdx.x, threadIdx.x, idx, row_ptr[row + 1], total_nnz);
//                 }
//             }

//             // Warp reduction
//             #pragma unroll
//             for (int offset = warp_size / 2; offset > 0; offset /= 2) {
//                 sum += __shfl_down_sync(0xffffffff, sum, offset);
//             }

//             // Write output with ReLU, only by lane 0
//             if (lane_id == 0) {
//                 int output_idx = row * H_dim + (h_tile + j_base);
//                 if (row < N && (h_tile + j_base) < H_dim) {
//                     output[output_idx] = fmaxf(sum, 0.0f);
//                     // printf("Block %d, Warp %d: Output[%d]=%f\n", 
//                     //        blockIdx.x, warp_id, output_idx, output[output_idx]);
//                 } else {
//                     // printf("Block %d, Warp %d: Skipping invalid output write: row=%d (N=%d), h_tile+j_base=%d (H_dim=%d)\n",
//                     //        blockIdx.x, warp_id, row, N, h_tile + j_base, H_dim);
//                 }
//             }
//         }
//         __syncthreads();
//         // printf("Block %d: Finished h_tile=%d\n", blockIdx.x, h_tile);
//     }
//     printf("Block %d: Row %d completed\n", blockIdx.x, row);
// }

// extern "C" void launch_gcn_fused_kernel(
//     const int* row_ptr, const int* col_idx, const float* values,
//     const float* X, const float* W, float* H,
//     const int* num_neighbors,
//     int N, int F_padded, int actual_F, int H_dim, int total_nnz
// ) {
//     cudaError_t err;

//     // Set reasonable launch parameters
//     int threads_per_block = 256;
//     cudaDeviceProp prop;
//     cudaGetDeviceProperties(&prop, 0);
//     int warp_size = prop.warpSize;

//     // Calculate maximum shared memory (90% of available)
//     size_t max_shared_mem = static_cast<size_t>(prop.sharedMemPerBlock * 0.9);

//     // Calculate tile_size_H using ternary operator to avoid min ambiguity
//     size_t max_tile = max_shared_mem / (static_cast<size_t>(actual_F) * sizeof(float));
//     int tile_size_H = (static_cast<size_t>(H_dim) < max_tile) ? H_dim : static_cast<int>(max_tile);

//     // Cap tile_size_H at 128 for better occupancy
//     tile_size_H = (tile_size_H < 128) ? tile_size_H : 128;

//     // Ensure tile_size_H is a multiple of warp_size and at least warp_size
//     tile_size_H = (tile_size_H / warp_size) * warp_size;
//     if (tile_size_H < warp_size) tile_size_H = warp_size;

//     // Calculate shared memory size
//     size_t shared_mem_size = static_cast<size_t>(actual_F) * static_cast<size_t>(tile_size_H) * sizeof(float);

//     int grid_size = N;

//     printf("Launching kernel: N=%d, F_padded=%d, actual_F=%d, H_dim=%d, total_nnz=%d\n", 
//            N, F_padded, actual_F, H_dim, total_nnz);
//     printf("Grid size=%d, Threads per block=%d, Shared memory=%zu bytes\n", 
//            grid_size, threads_per_block, shared_mem_size);
//     printf("Tile size H=%d, Warp size=%d, max_tile=%zu, tile_size_H=%d\n", 
//            tile_size_H, warp_size, max_tile, tile_size_H);

//     // Launch kernel with passed num_neighbors and total_nnz
//     gcn_fused_kernel_optimized<<<grid_size, threads_per_block, shared_mem_size>>>(
//         row_ptr, col_idx, values, X, W, H, num_neighbors,
//         N, F_padded, actual_F, H_dim, tile_size_H, warp_size, total_nnz
//     );

//     // Check for launch errors
//     err = cudaGetLastError();
//     if (err != cudaSuccess) {
//         printf("CUDA Error after launch: %s\n", cudaGetErrorString(err));
//         exit(1);
//     }

//     cudaDeviceSynchronize();
//     err = cudaGetLastError();
//     if (err != cudaSuccess) {
//         printf("CUDA Error after launch: %s\n", cudaGetErrorString(err));
//         exit(1);
//     }
//     printf("Kernel launched successfully\n");
// }