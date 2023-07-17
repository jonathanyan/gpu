const int kWarpSize = 32;
template <typename T, typename S>
__global__ void UnsortedSegmentMax(const T *input, const S *segment_ids, const S num_segments, int outer_size,
                                   int inner_size, const T init_K, T *output) {
  for (int t_idx = blockIdx.x * blockDim.x + threadIdx.x; t_idx < kWarpSize * num_segments * inner_size;
       t_idx += blockDim.x * gridDim.x) {
    int segment_id = t_idx / kWarpSize / inner_size;
    int inner_id = t_idx / kWarpSize % inner_size;
    int lane_id = threadIdx.x % kWarpSize;
    T threadK = init_K;

    for (int i = lane_id; i < outer_size; i += kWarpSize) {
      if (segment_ids[i] != segment_id) continue;
      T other_K = input[i * inner_size + inner_id];
      if (threadK < other_K) {
        threadK = other_K;
      }
    }
    __syncwarp();

    for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
      T other_K = __shfl_down_sync(0xffffffff, threadK, offset);
      if (threadK < other_K) {
        threadK = other_K;
      }
    }

    __syncwarp();

    if (lane_id == 0) {
      output[segment_id * inner_size + inner_id] = threadK;
    }
    __syncthreads();
  }
}

