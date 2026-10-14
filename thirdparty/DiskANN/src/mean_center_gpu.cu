// Copyright (c) KIOXIA Corporation. All rights reserved.
// Licensed under the MIT license.
/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cstddef>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/core/types.hpp>
#include <raft/stats/mean.cuh>
#include <raft/stats/mean_center.cuh>

#include "diskann/diskann_gpu.h"
#include "diskann/nvtx_helper.h"

// GPU implementation of the make_zero_mean step in generate_pq_pivots:
// computes the per-dimension centroid of the (num_data x dim) row-major
// training matrix and subtracts it from every vector in place. This mirrors
// the CPU loops:
//   centroid[d] = (1/num_data) * sum_p data[p * dim + d];
//   data[p * dim + d] -= centroid[d];
void mean_center_gpu(const raft::resources& dev_resources, float* d_data, size_t num_data, size_t dim,
                     float* d_centroid, bool compute_mean) {
    DISKANN_NVTX_RANGE("mean_center_gpu");
    auto stream = raft::resource::get_cuda_stream(dev_resources);

    // Populate centroid per column (divides by num_data, matching the CPU path).
    auto d_data_view = raft::make_device_matrix_view<float, int64_t>(d_data, num_data, dim);
    auto d_centroid_view = raft::make_device_vector_view<float, int64_t>(d_centroid, dim);
    if (compute_mean) {
      raft::stats::mean(dev_resources, raft::make_const_mdspan(d_data_view), d_centroid_view);
    }

    // Subtract the centroid from every row, writing the result back in place.
    raft::stats::mean_center<raft::Apply::ALONG_ROWS>(dev_resources, raft::make_const_mdspan(d_data_view),
                                                      raft::make_const_mdspan(d_centroid_view), d_data_view);
}
