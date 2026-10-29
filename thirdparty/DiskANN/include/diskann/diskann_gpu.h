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
*
* MODIFICATIONS:
* [21/08/2025] - [Gili Buzaglo]:
* Import and adjust methods from cuvs library example source to knowhere
* (vamana_build_and_write and read_bin_dataset)
*
* source repository: https://github.com/rapidsai/cuvs
* release tag: v25.06.00
* commit id: 2e20f1ec8a7e2325cdcf49d289879c2624e5bb5d
* File name: https://github.com/rapidsai/cuvs/blob/v25.06.00/examples/cpp/src/vamana_example.cu
* Function name: vamana_build_and_write
* Lines: 30 – 63
*
* source repository: https://github.com/rapidsai/cuvs
* release tag: v25.06.00
* commit id: 2e20f1ec8a7e2325cdcf49d289879c2624e5bb5d
* File name: https://github.com/rapidsai/cuvs/blob/v25.06.00/examples/cpp/src/common.cuh
* Function name: read_bin_dataset
* Lines: 106 - 129
*/

#ifndef THIRDPARTY_DISKANN_INCLUDE_DISKANN_DISKANN_GPU_H_
#define THIRDPARTY_DISKANN_INCLUDE_DISKANN_DISKANN_GPU_H_
#pragma once

#ifdef KNOWHERE_WITH_CUVS
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include <cuvs/preprocessing/quantize/pq.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/device_resources_manager.hpp>
#include <raft/core/pinned_mdarray.hpp>

template<typename T>
void gen_random_slice_gpu(const std::string data_file, double p_val,
    std::unique_ptr<raft::pinned_matrix<float, int64_t>>& sampled_data,
    size_t &slice_size, size_t &ndims,
    std::vector<uint32_t> *sampled_ids = nullptr);

template <typename T, typename idxT = std::size_t>
raft::device_matrix<T, idxT> read_bin_dataset(const raft::device_resources& dev_resources,
                                              const std::string& fname);

template <typename T>
void vamana_build_and_write(raft::device_resources const& dev_resources,
                            raft::device_matrix_view<const T, int64_t> dataset,
                            std::string out_fname,
                            int degree,
                            int visited_size,
                            float max_fraction,
                            int iters);
bool is_gpu_available();

// train function for centroids computation
void kmeans_gpu_device_data(
    const raft::resources& dev_resources,
    const float* d_chunk_data,
    size_t num_train,
    size_t chunk_size,
    size_t num_centers,
    int max_iter,
    float* d_centroids_out,
	bool is_balanced=false);

// train function for centroids computation
void kmeans_gpu(
    const raft::resources& dev_resources,
    const float* h_chunk_data,
    size_t num_train,
    size_t chunk_size,
    size_t num_centers,
    int max_iter,
    float* h_centroids_out,
	bool is_balanced=false);

// train function for centroids combined with medoids computation
void kmeans_medoids_gpu(
    const raft::resources& dev_resources,
    const float* h_chunk_data,
    size_t num_train,
    size_t dim,
    size_t num_centers,
    int max_iter,
    float* h_centroids_out,
	bool is_balanced,
	uint32_t* h_medoids);

// compute the mean (if compute_mean) then center the (num_data x dim) row-major data
// matrix in place and writes the per-dimension centroid (length dim) to d_centroid
void mean_center_gpu(const raft::resources& dev_resources,
                     float* d_data,
                     size_t num_data,
                     size_t dim,
                     float* d_centroid,
                     bool compute_mean=true);

// initialize a cuVS PQ quantizer from existing DiskANN centroid + pivot data.
// when dim is not a multiple of num_pq_chunks, every chunk (given by chunk_offsets) is zero-padded to
// padded_pq_chunk_width, and blocks must be encoded with transform_padded_pq_gpu.
// returns nullptr when the requested shape/config is not supported by this path.
std::unique_ptr<cuvs::preprocessing::quantize::pq::quantizer<float>>
initialize_pq_quantizer_gpu(const raft::resources& dev_resources,
                            const float* h_centroid,
                            const float* h_full_pivot_data,
                            size_t dim,
                            size_t num_centers,
                            size_t num_pq_chunks,
                            const uint32_t* chunk_offsets);

// GPU transform path for one block:
// 1) host->device copy
// 2) cuVS PQ transform to uint8 codes (subtracts the quantizer VQ centroid)
// 3) device->host copy of compressed codes
// if h_inflated_out is not null, also writes reconstructed vectors to host.
void transform_pq_gpu(const raft::resources& dev_resources,
                      const cuvs::preprocessing::quantize::pq::quantizer<float>& pq_quantizer,
                      const float* h_block_data,
                      size_t cur_blk_size,
                      size_t dim,
                      size_t num_pq_chunks,
                      float* d_block_data,
                      uint8_t* d_compressed_data_block,
                      uint8_t* h_codes_out,
                      float* h_inflated_out = nullptr);

// width of the widest PQ chunk, i.e. the subspace width of the padded quantizer.
size_t padded_pq_chunk_width(const uint32_t* chunk_offsets, size_t num_pq_chunks);

// same as transform_pq_gpu for a quantizer of chunks of unequal width (dim not a multiple of num_pq_chunks).
// d_block_data holds cur_blk_size x dim floats, d_padded_block_data cur_blk_size x
// (padded_pq_chunk_width * num_pq_chunks) floats.
void transform_padded_pq_gpu(const raft::resources& dev_resources,
                             const cuvs::preprocessing::quantize::pq::quantizer<float>& pq_quantizer,
                             const float* h_block_data,
                             size_t cur_blk_size,
                             size_t dim,
                             size_t num_pq_chunks,
                             const uint32_t* chunk_offsets,
                             float* d_block_data,
                             float* d_padded_block_data,
                             uint8_t* d_compressed_data_block,
                             uint8_t* h_codes_out,
                             float* h_inflated_out = nullptr);

// assign function for computes the compressed PQ vector per each vector in the dataset
// based on centroids computation
int predict_gpu(const raft::resources& dev_resources,
				const float* h_data,
                size_t n_samples,
                size_t dim,
                const float* h_centroids,
                size_t n_clusters,
				int* h_labels);

template <typename T>
int brute_force_gpu(const raft::resources& dev_resources,
		const T* h_data,
        size_t n_samples,
        size_t dim,
		size_t k,
        const T* queries,
        size_t n_queries,
		int64_t* h_labels);

void gpu_get_mem_info(const raft::resources &dev_resources, size_t &gpu_free_mem,size_t &gpu_total_mem);

#endif




#endif /* THIRDPARTY_DISKANN_INCLUDE_DISKANN_DISKANN_GPU_H_ */
