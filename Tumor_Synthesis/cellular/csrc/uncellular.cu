#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <float.h>
#include <math.h>
#include <tuple>
#include <curand_kernel.h>
#include<assert.h>

__device__ float probability(
    const int target_val,
    const int organ_hu_lowerbound, 
    const int interval
){
    // 3 level of tissue area
    const int interval_1 = organ_hu_lowerbound;
    const int interval_2 = organ_hu_lowerbound + interval;
    const int interval_3 = organ_hu_lowerbound + interval * 2;

    // probability of grow in each level organ tissue
    if(target_val == interval_1){
        return 1;
    }
    else if (target_val == interval_2){
        return 0.3;
    }
    else if (target_val == interval_3){
        return 0.1;
    }
    else{
        return 0;
    }
}

__global__ void UngrowTensorKernel(
    bool* eligibility_tensor,
    float* prob_ungrow_tensor,
    int* ungrow_tensor,
    const int* state_tensor_prev,
    int* original_density_state_map,
    const int H,
    const int W,
    const int D,
    const int Y_range,
    const int X_range,
    const int Z_range,
    const int grow_per_cell,
    const int max_try,
    const int organ_hu_lowerbound,
    const int organ_standard_val,
    const int outrange_standard_val,
    const int threshold,
    const bool flag
){
    const int interval = (outrange_standard_val - organ_hu_lowerbound) / 3;
    
    const int num_threads = gridDim.x * blockDim.x;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    
    for (int pid = tid; pid < H * W * D; pid += num_threads) {
        // Commented Code is only needed if mass effect is enabled ig (I didn't recheck whether it is needed at least then)

        // if(original_density_state_map[pid] == outrange_standard_val){
        //     eligibility_tensor[pid] = (state_tensor_prev[pid] != organ_standard_val);
        // }
        // else{
            // eligibility_tensor[pid] = ((state_tensor_prev[pid] != organ_standard_val) && (state_tensor_prev[pid] != outrange_standard_val));
        // }

        eligibility_tensor[pid] = ((state_tensor_prev[pid] != organ_standard_val) && (state_tensor_prev[pid] != outrange_standard_val));

        prob_ungrow_tensor[pid] = 0.0f;
        ungrow_tensor[pid] = 0;
    }

    for (int pid = tid; pid < H * W * D; pid += num_threads) {
        const int curr_val = state_tensor_prev[pid];

        if(curr_val == threshold){
            continue; // Reverse Invasion principle
        }

        const int y = pid / (W * D);
        const int x = (pid % (W * D)) / D;
        const int z = pid % D;

        // assert(Y_range % 2 == 1);
        // assert(X_range % 2 == 1);
        // assert(Z_range % 2 == 1);
        // assert(Y_range > 0);
        // assert(X_range > 0);
        // assert(Z_range > 0);
        // assert(Y_range < H);
        // assert(X_range < W);
        // assert(Z_range < D);

        // To simplify the code for review as we are using these values anyway
        assert(Y_range == 3);
        assert(X_range == 3);
        assert(Z_range == 3);

        int window_size = Y_range * X_range * Z_range - 1;

        bool eligible[28];
        for(int i=0; i<28; ++i){
            eligible[i] = false;
        }

        int n = 0;
        int ind = -1;
        for(int dx = -1; dx <= 1; ++dx){
            for(int dy = -1; dy <= 1; ++dy){
                for(int dz = -1; dz <= 1; ++dz){
                    // skip the center cell
                    if (dx == 0 && dy == 0 && dz == 0){
                        continue;
                    }

                    ++ind;

                    int y_shift = y + dy;
                    int x_shift = x + dx;
                    int z_shift = z + dz;

                    if (y_shift < 0 || y_shift >= H || x_shift < 0 || x_shift >= W || z_shift < 0 || z_shift >= D){
                        continue;
                    }

                    // check if the cell is eligible
                    if (eligibility_tensor[(y_shift) * (W * D) + (x_shift) * D + (z_shift)]){
                        eligible[ind] = true;
                    }

                    ++n;
                }
            }
        }

        float ungrow_contribution = min(((float)max_try)/window_size, ((float)grow_per_cell)/n);
        assert(ungrow_contribution <= 1.0f);
        assert(ungrow_contribution >= 0.0f);

        ind = -1;
        for(int dx = -1; dx <= 1; ++dx){
            for(int dy = -1; dy <= 1; ++dy){
                for(int dz = -1; dz <= 1; ++dz){
                    // skip the center cell
                    if (dx == 0 && dy == 0 && dz == 0){
                        continue;
                    }

                    ++ind;

                    int y_shift = y + dy;
                    int x_shift = x + dx;
                    int z_shift = z + dz;

                    if (y_shift < 0 || y_shift >= H || x_shift < 0 || x_shift >= W || z_shift < 0 || z_shift >= D){
                        continue;
                    }

                    // check if the cell is eligible
                    if (eligible[ind]){
                        prob_ungrow_tensor[pid] += ungrow_contribution * probability(original_density_state_map[pid], organ_hu_lowerbound, interval);
                    }
                }
            }
        }
    }

    for (int pid = tid; pid < H * W * D; pid += num_threads) {
        // ungrow_tensor[pid] = round(prob_ungrow_tensor[pid]);
        ungrow_tensor[pid] = ceil(prob_ungrow_tensor[pid]);
    }
}

// tensor& collapse_ungrow_tensor(tensor& ungrow_tensor, prob_ungrow_tensor, prob_base){
//     cudaSetZeroes(ungrow_tensor)
//     cudaDivide(prob_ungrow_tensor, prob_base)
    
// }

__global__ void UnupdateCellularKernel(
    const int* ungrow_tensor,
    const int* state_tensor_prev,
    int* density_state_tensor,
    const int H,
    const int W,
    const int D,
    const int Y_range,
    const int X_range,
    const int Z_range,
    const int grow_per_cell,
    const int max_try,
    const int organ_hu_lowerbound,
    const int organ_standard_val,
    const int outrange_standard_val,
    const int threshold,
    const bool flag,
    int* state_tensor // (H, W, D)
){
    const int num_threads = gridDim.x * blockDim.x;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int interval = (outrange_standard_val - organ_hu_lowerbound) / 3;
    
    // Growth and Invasion
    if (true){
        for (int pid = tid; pid < H * W * D; pid += num_threads) {
            const int curr_val = state_tensor_prev[pid];
            if (curr_val == organ_standard_val || curr_val >= outrange_standard_val){
                continue;
            }
            
            // calculate current position
            const int y = pid / (W * D);
            const int x = (pid % (W * D)) / D;
            const int z = pid % D;

            //extend cell and proliferative itself
            if (curr_val < threshold){
                atomicAdd(state_tensor + (y) * (W * D) + (x) * D + (z), -1);
            }
            
            atomicAdd(state_tensor + (y) * (W * D) + (x) * D + (z), ungrow_tensor[pid]);
        }
    }
}


// C++ interface for the CUDA kernel
// initialize the state_tensor
at::Tensor UnupdateCellular(
    const at::Tensor& state_tensor_prev,
    const at::Tensor& density_state_tensor,
    const int Y_range,
    const int X_range,
    const int Z_range,
    const int grow_per_cell,
    const int max_try,
    const int organ_hu_lowerbound,
    const int organ_standard_val,
    const int outrange_standard_val,
    const int threshold,
    const bool flag,
    at::Tensor& state_tensor // (H, W, D)
    
){
    at::cuda::CUDAGuard device_guard(state_tensor.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    // bin_points configuration
    const int H = state_tensor.size(0);
    const int W = state_tensor.size(1);
    const int D = state_tensor.size(2);
    
    const size_t blocks = 1024;
    const size_t threads = 64;
    
    // const size_t blocks = 32;
    // const size_t threads = 8;

    const int n_pixels = H * W * D;

    bool* eligibility_tensor;
    cudaMalloc(&eligibility_tensor, sizeof(bool) * n_pixels);

    float* prob_ungrow_tensor;
    cudaMalloc(&prob_ungrow_tensor, sizeof(float) * n_pixels);

    int* ungrow_tensor;
    cudaMalloc(&ungrow_tensor, sizeof(int) * n_pixels);

    UngrowTensorKernel<<<blocks, threads, 0, stream>>>(
        eligibility_tensor,
        prob_ungrow_tensor,
        ungrow_tensor,
        state_tensor_prev.contiguous().data_ptr<int>(),
        density_state_tensor.contiguous().data_ptr<int>(),
        H,
        W,
        D,
        Y_range,
        X_range,
        Z_range,
        grow_per_cell,
        max_try,
        organ_hu_lowerbound,
        organ_standard_val,
        outrange_standard_val,
        threshold,
        flag
    );

    // collapse_ungrow_tensor(prob_ungrow_tensor, ungrow_tensor, 
    //     prob_base = max(max_try, grow_per_cell))

    // Launch the cuda kernel
    UnupdateCellularKernel<<<blocks, threads, 0, stream>>>(
        ungrow_tensor,
        state_tensor_prev.contiguous().data_ptr<int>(),
        density_state_tensor.contiguous().data_ptr<int>(),
        H,
        W,
        D,
        Y_range,
        X_range,
        Z_range,
        grow_per_cell,
        max_try,
        organ_hu_lowerbound,
        organ_standard_val,
        outrange_standard_val,
        threshold,
        flag,
        state_tensor.contiguous().data_ptr<int>() // (H, W, D)
    );

    return state_tensor;
}
