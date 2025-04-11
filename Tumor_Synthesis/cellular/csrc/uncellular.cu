__global__ void UnupdateCellularKernel(
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
            curandState state;
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

    // Launch the cuda kernel
    UnupdateCellularKernel<<<blocks, threads, 0, stream>>>(
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
