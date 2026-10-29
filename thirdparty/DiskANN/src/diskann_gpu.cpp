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
#include <cuvs/cluster/kmeans.hpp>
#include <cuvs/neighbors/vamana.hpp>
#include "diskann/diskann_gpu.h"
#include "diskann/nvtx_helper.h"
#include "diskann/utils.h"
#include "knowhere/log.h"
#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <raft/core/resources.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/util/cudart_utils.hpp>
#include <cuvs/neighbors/brute_force.hpp>
#include <cuvs/preprocessing/quantize/pq.hpp>

// same as gen_random_slice, but use pinned memory for the sampled data
template<typename T>
void gen_random_slice_gpu(const std::string data_file, double p_val,
        std::unique_ptr<raft::pinned_matrix<float, int64_t>>& sampled_data,
        size_t &slice_size, size_t &ndims,
        std::vector<uint32_t> *sampled_ids) {
  DISKANN_NVTX_RANGE("gen_random_slice_gpu(file)");
  auto const& res = raft::device_resources_manager::get_device_resources();
  size_t   npts;
  uint32_t npts32, ndims32;

  // amount to read in one shot
  _u64 read_blk_size = 64 * 1024 * 1024;
  // create cached reader + writer
  cached_ifstream base_reader(data_file.c_str(), read_blk_size);

  // metadata: npts, ndims
  base_reader.read((char *) &npts32, sizeof(unsigned));
  base_reader.read((char *) &ndims32, sizeof(unsigned));
  npts = npts32;
  ndims = ndims32;

  std::unique_ptr<T[]> cur_vector_T = std::make_unique<T[]>(ndims);
  p_val = p_val < 1 ? p_val : 1;

  std::random_device rd;  // Will be used to obtain a seed for the random number
  size_t             x = rd();
  std::mt19937       generator((unsigned) x);
  std::uniform_real_distribution<float> distribution(0, 1);
  std::vector<bool> selected(npts, false);
  slice_size = 0;

  for (size_t i = 0; i < npts; i++) {
    float rnd_val = distribution(generator);
    if (rnd_val < p_val) {
      selected[i] = true;
      slice_size++;
      if (sampled_ids) {
        sampled_ids->push_back(i);
      }
    }
  }
  
  sampled_data = std::make_unique<raft::pinned_matrix<float, int64_t>>(raft::make_pinned_matrix<float, int64_t>(res, slice_size, ndims));
  auto sampled_data_ptr = sampled_data->data_handle();
  size_t sampled_index = 0;
  for (size_t i = 0; i < npts; i++) {
    base_reader.read((char *) cur_vector_T.get(), ndims * sizeof(T));
    if (selected[i]) {
      for (size_t d = 0; d < ndims; d++) {
        sampled_data_ptr[sampled_index * ndims + d] = float(cur_vector_T[d]);
      }
      sampled_index++;
    }
  }
}

template void gen_random_slice_gpu<int8_t>(const std::string data_file, double p_val,
	std::unique_ptr<raft::pinned_matrix<float, int64_t>>& sampled_data,
	size_t &slice_size, size_t &ndims,
	std::vector<uint32_t> *sampled_ids);
template void gen_random_slice_gpu<uint8_t>(const std::string data_file, double p_val,
	std::unique_ptr<raft::pinned_matrix<float, int64_t>>& sampled_data,
	size_t &slice_size, size_t &ndims,
	std::vector<uint32_t> *sampled_ids);
template void gen_random_slice_gpu<float>(const std::string data_file, double p_val,
	std::unique_ptr<raft::pinned_matrix<float, int64_t>>& sampled_data,
	size_t &slice_size, size_t &ndims,
	std::vector<uint32_t> *sampled_ids);
template void gen_random_slice_gpu<knowhere::fp16>(const std::string data_file, double p_val,
	std::unique_ptr<raft::pinned_matrix<float, int64_t>>& sampled_data,
	size_t &slice_size, size_t &ndims,
	std::vector<uint32_t> *sampled_ids);
template void gen_random_slice_gpu<knowhere::bf16>(const std::string data_file, double p_val,
	std::unique_ptr<raft::pinned_matrix<float, int64_t>>& sampled_data,
	size_t &slice_size, size_t &ndims,
	std::vector<uint32_t> *sampled_ids);

template<typename T, typename idxT>
raft::device_matrix<T, idxT> read_bin_dataset(
		const raft::device_resources &dev_resources, const std::string &fname) {
	DISKANN_NVTX_RANGE("read_bin_dataset");
	// Open file
	std::ifstream datafile(fname, std::ios::binary);
	if (!datafile) {
		throw std::runtime_error("Failed to open file: " + fname);
	}

	// Read header
	uint32_t N = 0;
	uint32_t dim = 0;
	datafile.read(reinterpret_cast<char*>(&N), sizeof(uint32_t));
	datafile.read(reinterpret_cast<char*>(&dim), sizeof(uint32_t));

    LOG_KNOWHERE_INFO_ << "Read in file - N: " << N << ", dim: " << dim;

	std::size_t total = static_cast<std::size_t>(N) * dim;
	if (total == 0) {
		throw std::runtime_error("Invalid dataset: total size is zero");
	}

	// Allocate pinned host memory
	T *pinned_data = nullptr;
	RAFT_CUDA_TRY(
			cudaMallocHost(reinterpret_cast<void**>(&pinned_data),
					total * sizeof(T)));

	// Read data into pinned memory
	datafile.read(reinterpret_cast<char*>(pinned_data), total * sizeof(T));
	if (!datafile) {
		cudaFreeHost(pinned_data);
		throw std::runtime_error("Failed to read vector data from file");
	}
	datafile.close();

	// Create device matrix
	auto dataset = raft::make_device_matrix<T, idxT>(dev_resources, N, dim);

	// Copy to device using raft::copy
	raft::copy(dataset.data_handle(), pinned_data, total,
			raft::resource::get_cuda_stream(dev_resources));
    raft::resource::sync_stream(dev_resources);
	cudaFreeHost(pinned_data);

	return dataset;
}

template<typename T>
void vamana_build_and_write(raft::device_resources const &dev_resources,
		raft::device_matrix_view<const T, int64_t> dataset,
		std::string out_fname, int degree, int visited_size, float max_fraction,
		int iters) {
	std::stringstream ss;
	ss << "vamana_build_and_write(rows: " << dataset.extent(0) << ", cols: " << dataset.extent(1) << ")";
	DISKANN_NVTX_RANGE(ss.str());
	using namespace cuvs::neighbors;

	// use default index parameters
	vamana::index_params index_params;
	index_params.max_fraction = max_fraction;
	index_params.visited_size = visited_size;
	index_params.graph_degree = degree;
	index_params.vamana_iters = iters;

	LOG_KNOWHERE_INFO_ << "Building Vamana index (search graph)";

	auto start = std::chrono::system_clock::now();
	auto index = vamana::build(dev_resources, index_params, dataset);
	auto end = std::chrono::system_clock::now();
	std::chrono::duration<double> elapsed_seconds = end - start;

	LOG_KNOWHERE_INFO_ << "Vamana index has " << index.size() << " vectors";
	LOG_KNOWHERE_INFO_ << "Vamana graph has degree " << index.graph_degree()
			<< ", graph size [" << index.graph().extent(0) << ", "
			<< index.graph().extent(1) << "]";

	LOG_KNOWHERE_INFO_ << "Time to build index: " << elapsed_seconds.count() << "s\n";
	// Output index to file
	serialize(dev_resources, out_fname, index, false);
}

void kmeans_gpu_device_data(const raft::resources &dev_resources, const float *d_chunk_data,
		size_t num_train, size_t dim, size_t num_centers, int max_iter,
		float *d_centroids_out, bool is_balanced/*=false*/) {
	std::stringstream ss;
	ss << "kmeans_gpu_device_data(num_train: " << num_train << ", dim: " << dim << ", num_centers: " << num_centers << ", max_iter: " << max_iter << ", is_balanced: " << is_balanced << ")";
	DISKANN_NVTX_RANGE(ss.str());
	// KMeans parameters
	cuvs::cluster::kmeans::params km_params;
	int64_t n_iter;
	float inertia;
	km_params.max_iter = max_iter;
	km_params.n_init = 1;
	km_params.n_clusters = num_centers;

	auto d_data = raft::make_device_matrix_view<const float, int64_t>(d_chunk_data, num_train, dim);
	auto d_centroids = raft::make_device_matrix_view<float, int64_t>(d_centroids_out, num_centers, dim);

	// Run KMeans (fit uses RAFT-managed streams internally)
	if(!is_balanced) {
		cuvs::cluster::kmeans::fit(dev_resources, km_params,
				d_data, std::nullopt,
				d_centroids,
				raft::make_host_scalar_view<float>(&inertia),
				raft::make_host_scalar_view<int64_t>(&n_iter));
	} else {
		//use balance kmeans
		cuvs::cluster::kmeans::balanced_params b_p;
		b_p.n_iters=100;
		cuvs::cluster::kmeans::fit(dev_resources, b_p, d_data, d_centroids);
	}
}

void kmeans_gpu(const raft::resources &dev_resources, const float *h_chunk_data,
		size_t num_train, size_t dim, size_t num_centers, int max_iter,
		float *h_centroids_out, bool is_balanced/*=false*/) {
	std::stringstream ss;
	ss << "kmeans_gpu(num_train: " << num_train << ", dim: " << dim << ", num_centers: " << num_centers << ", max_iter: " << max_iter << ", is_balanced: " << is_balanced << ")";
	DISKANN_NVTX_RANGE(ss.str());

	// Allocate device matrices
	auto d_data = raft::make_device_matrix<float, int64_t>(dev_resources, num_train, dim);
	auto d_centroids = raft::make_device_matrix<float, int64_t>(dev_resources, num_centers, dim);


	raft::copy(d_data.data_handle(), h_chunk_data, num_train * dim, raft::resource::get_cuda_stream(dev_resources));
	kmeans_gpu_device_data(dev_resources, d_data.data_handle(), num_train, dim, num_centers, max_iter, d_centroids.data_handle(), is_balanced);
	raft::copy(h_centroids_out, d_centroids.data_handle(), num_centers * dim,
			raft::resource::get_cuda_stream(dev_resources));
    raft::resource::sync_stream(dev_resources);
}

void kmeans_medoids_gpu(const raft::resources &dev_resources, const float *h_chunk_data,
		size_t num_train, size_t dim, size_t num_centers, int max_iter,
		float *h_centroids_out, bool is_balanced, uint32_t* h_medoids) {
	DISKANN_NVTX_RANGE("kmeans_medoids_gpu");

	// Allocate device matrices
	auto d_data = raft::make_device_matrix<float, int64_t>(dev_resources, num_train, dim);
	auto d_centroids = raft::make_device_matrix<float, int64_t>(dev_resources, num_centers, dim);
    auto d_medoids_i64    = raft::make_device_matrix<int64_t, int64_t>(dev_resources, num_centers, 1);
	auto h_medoids_i64    = raft::make_host_vector<int64_t, int64_t>(num_centers);

	// Copy input chunk to device asynchronously
	raft::copy(d_data.data_handle(), h_chunk_data, num_train * dim, raft::resource::get_cuda_stream(dev_resources));
	kmeans_gpu_device_data(dev_resources, d_data.data_handle(), num_train, dim, num_centers, max_iter, d_centroids.data_handle(), is_balanced);

	// Compute medoids using brute force
	using namespace cuvs::neighbors;
	brute_force::index_params bf_index_params;
	brute_force::search_params bf_search_params;
	auto index = brute_force::build(dev_resources, bf_index_params, raft::make_const_mdspan(d_data.view()));
    auto distances = raft::make_device_matrix<float, int64_t>(dev_resources, num_centers, 1);

    brute_force::search(dev_resources, bf_search_params, index,
		raft::make_const_mdspan(d_centroids.view()), d_medoids_i64.view(), distances.view());
	raft::copy(h_centroids_out, d_centroids.data_handle(), num_centers * dim,
			raft::resource::get_cuda_stream(dev_resources));
	raft::copy(h_medoids_i64.data_handle(), d_medoids_i64.data_handle(), num_centers, raft::resource::get_cuda_stream(dev_resources));
    raft::resource::sync_stream(dev_resources);
	for (size_t i = 0; i < num_centers; ++i) {
		h_medoids[i] = static_cast<uint32_t>(h_medoids_i64(i));
	}
}

int predict_gpu(const raft::resources& handle,
                const float* h_data,
                size_t n_samples,
                size_t dim,
                const float* h_centroids,
                size_t n_clusters,
                int* h_labels) {
    DISKANN_NVTX_RANGE("predict_gpu");
    if (!is_gpu_available()) return -1;

    // Allocate device buffers
    auto d_data      = raft::make_device_matrix<float, int64_t>(handle, n_samples, dim);
    auto d_centroids = raft::make_device_matrix<float, int64_t>(handle, n_clusters, dim);
    auto d_labels    = raft::make_device_vector<int64_t, int64_t>(handle, n_samples);

    // Copy to device
    raft::copy(d_data.data_handle(), h_data, n_samples * dim,
               raft::resource::get_cuda_stream(handle));
    raft::copy(d_centroids.data_handle(), h_centroids, n_clusters * dim,
               raft::resource::get_cuda_stream(handle));

    float inertia = 0.0f;
    auto inertia_view = raft::make_host_scalar_view<float>(&inertia);

    cuvs::cluster::kmeans::params params;
    params.n_clusters = static_cast<int>(n_clusters);
    params.metric     = cuvs::distance::DistanceType::L2Expanded;

    cuvs::cluster::kmeans::predict(
        handle,
        params,
        raft::make_const_mdspan(d_data.view()),
        std::nullopt,
        raft::make_const_mdspan(d_centroids.view()),
        d_labels.view(),
        /*normalize_weight=*/false,
        inertia_view);

    // Copy labels back (int64_t on device -> int on host)
    std::vector<int64_t> h_labels_i64(n_samples);
    raft::copy(h_labels_i64.data(), d_labels.data_handle(), n_samples,
               raft::resource::get_cuda_stream(handle));
    raft::resource::sync_stream(handle);
    for (size_t i = 0; i < n_samples; ++i) {
        h_labels[i] = static_cast<int>(h_labels_i64[i]);
    }

    return 0;
}

bool is_gpu_available() {
	int count = 0;
	return (cudaGetDeviceCount(&count) == cudaSuccess) && (count > 0);
}

void gpu_get_mem_info(const raft::resources &dev_resources, size_t &gpu_free_mem,
		size_t &gpu_total_mem) {
    DISKANN_NVTX_RANGE("gpu_get_mem_info");
    int dev = raft::resource::get_device_id(dev_resources);

    int prev_dev;
    cudaGetDevice(&prev_dev);

    cudaSetDevice(dev);

    cudaMemGetInfo(&gpu_free_mem, &gpu_total_mem);

    // restore previous device
    cudaSetDevice(prev_dev);
}

template<typename T>
int brute_force_gpu(const raft::resources &dev_resources, const T *h_data,
		size_t n_samples, size_t dim, size_t k, const T *queries,
		size_t n_queries, int64_t *h_labels) {
	DISKANN_NVTX_RANGE("brute_force_gpu");
	using namespace cuvs::neighbors;

	brute_force::index_params index_params;
	brute_force::search_params search_params;
	auto dataset = raft::make_device_matrix<T, int64_t>(dev_resources, n_samples, dim);
	auto queries_matrix = raft::make_device_matrix<T, int64_t>(dev_resources, n_queries, dim);

	raft::copy(dataset.data_handle(), h_data, n_samples * dim,
			raft::resource::get_cuda_stream(dev_resources));
	raft::copy(queries_matrix.data_handle(), queries, n_queries * dim,
			raft::resource::get_cuda_stream(dev_resources));

	auto index = brute_force::build(dev_resources, index_params,
			raft::make_const_mdspan(dataset.view()));

	auto neighbors = raft::make_device_matrix<int64_t, int64_t>(dev_resources,
			n_queries, k);
	auto distances = raft::make_device_matrix<T, int64_t>(dev_resources, n_queries, k);

	brute_force::search(dev_resources, search_params, index,
			raft::make_const_mdspan(queries_matrix.view()), neighbors.view(), distances.view());
	// Copy back
	raft::copy(h_labels, neighbors.data_handle(), n_queries * k,
			raft::resource::get_cuda_stream(dev_resources));
	return 0;

}

static std::unique_ptr<cuvs::preprocessing::quantize::pq::quantizer<float>>
initialize_padded_pq_quantizer_gpu(const raft::resources& dev_resources,
                                   const float* h_centroid,
                                   const float* h_full_pivot_data,
                                   size_t dim,
                                   size_t num_centers,
                                   size_t num_pq_chunks,
                                   const uint32_t* chunk_offsets);

std::unique_ptr<cuvs::preprocessing::quantize::pq::quantizer<float>>
initialize_pq_quantizer_gpu(const raft::resources& dev_resources,
                            const float* h_centroid,
                            const float* h_full_pivot_data,
                            size_t dim,
                            size_t num_centers,
                            size_t num_pq_chunks,
                            const uint32_t* chunk_offsets) {
    if (num_pq_chunks != 0 && dim % num_pq_chunks != 0) {
        return initialize_padded_pq_quantizer_gpu(dev_resources, h_centroid, h_full_pivot_data, dim, num_centers,
                                                  num_pq_chunks, chunk_offsets);
    }
    DISKANN_NVTX_RANGE("initialize_pq_quantizer_gpu");
    if (num_centers != 256 || num_pq_chunks == 0) {
        return nullptr;
    }

    const uint32_t subspace_dim = static_cast<uint32_t>(dim / num_pq_chunks);
    auto d_vq_code_book =
        raft::make_device_matrix<float, uint32_t>(dev_resources, 1, static_cast<uint32_t>(dim));
    auto d_pq_code_book = raft::make_device_matrix<float, uint32_t>(
        dev_resources, static_cast<uint32_t>(num_centers * num_pq_chunks), subspace_dim);
    auto d_empty = raft::make_device_matrix<uint8_t, int64_t>(dev_resources, 0, 0);

    // DiskANN stores pivots as [center][dim]. cuVS expects subspace-major layout:
    // [chunk * num_centers + center][subspace_dim].
    std::vector<float> pq_code_book_subspace_major(num_centers * dim);
    for (size_t chunk = 0; chunk < num_pq_chunks; chunk++) {
        for (size_t center = 0; center < num_centers; center++) {
            for (size_t k = 0; k < subspace_dim; k++) {
                pq_code_book_subspace_major[(chunk * num_centers + center) * subspace_dim + k] =
                    h_full_pivot_data[center * dim + chunk * subspace_dim + k];
            }
        }
    }

    auto stream = raft::resource::get_cuda_stream(dev_resources);
    raft::copy(d_vq_code_book.data_handle(), h_centroid, dim, stream);
    raft::copy(d_pq_code_book.data_handle(), pq_code_book_subspace_major.data(), num_centers * dim, stream);
    raft::resource::sync_stream(dev_resources);

    cuvs::cluster::kmeans::params km_params;
    km_params.n_clusters = static_cast<int>(num_centers);
    km_params.max_iter = 1;
    km_params.n_init = 1;
    auto pq_params = cuvs::preprocessing::quantize::pq::params(
        8, static_cast<uint32_t>(num_pq_chunks), true, true, 1,
        cuvs::preprocessing::quantize::pq::kmeans_params_variant(km_params));

    return std::make_unique<cuvs::preprocessing::quantize::pq::quantizer<float>>(
        cuvs::preprocessing::quantize::pq::quantizer<float>{
            pq_params, cuvs::neighbors::vpq_dataset<float, int64_t>(
                           std::move(d_vq_code_book), std::move(d_pq_code_book), std::move(d_empty))});
}

void transform_pq_gpu(const raft::resources& dev_resources,
                      const cuvs::preprocessing::quantize::pq::quantizer<float>& pq_quantizer,
                      const float* h_block_data,
                      size_t cur_blk_size,
                      size_t dim,
                      size_t num_pq_chunks,
                      float* d_block_data,
                      uint8_t* d_compressed_data_block,
                      uint8_t* h_codes_out,
                      float* h_inflated_out /*=nullptr*/) {
    DISKANN_NVTX_RANGE("transform_pq_gpu");
    auto stream = raft::resource::get_cuda_stream(dev_resources);

    // No explicit mean-centering here: pq::transform subtracts the single VQ center
    // (the DiskANN centroid) from every vector before encoding.
    raft::copy(d_block_data, h_block_data, cur_blk_size * dim, stream);

    auto d_block_data_view = raft::make_device_matrix_view<const float, int64_t>(
        d_block_data, static_cast<int64_t>(cur_blk_size), static_cast<int64_t>(dim));
    auto d_compressed_data_block_view = raft::make_device_matrix_view<uint8_t, int64_t>(
        d_compressed_data_block, static_cast<int64_t>(cur_blk_size), static_cast<int64_t>(num_pq_chunks));
    cuvs::preprocessing::quantize::pq::transform(
        dev_resources, pq_quantizer, d_block_data_view, d_compressed_data_block_view);

    raft::copy(h_codes_out, d_compressed_data_block, cur_blk_size * num_pq_chunks, stream);
    if (h_inflated_out != nullptr) {
        auto d_inflated_data_block = raft::make_device_matrix<float, int64_t>(
            dev_resources, static_cast<int64_t>(cur_blk_size), static_cast<int64_t>(dim));
        auto d_centroid_labels = raft::make_device_vector<uint32_t, int64_t>(
            dev_resources, static_cast<int64_t>(cur_blk_size));
        RAFT_CUDA_TRY(cudaMemsetAsync(d_centroid_labels.data_handle(),
                                      0,
                                      cur_blk_size * sizeof(uint32_t),
                                      stream));
        cuvs::preprocessing::quantize::pq::inverse_transform(
            dev_resources,
            pq_quantizer,
            raft::make_const_mdspan(d_compressed_data_block_view),
            d_inflated_data_block.view(),
            std::make_optional(raft::make_const_mdspan(d_centroid_labels.view())));
        raft::copy(h_inflated_out, d_inflated_data_block.data_handle(), cur_blk_size * dim, stream);
    }
    raft::resource::sync_stream(dev_resources);
}

size_t padded_pq_chunk_width(const uint32_t* chunk_offsets, size_t num_pq_chunks) {
    size_t width = 0;
    for (size_t chunk = 0; chunk < num_pq_chunks; chunk++) {
        width = std::max<size_t>(width, chunk_offsets[chunk + 1] - chunk_offsets[chunk]);
    }
    return width;
}

// Every chunk is zero-padded to padded_pq_chunk_width, since cuVS requires equal-width subspaces.
static std::unique_ptr<cuvs::preprocessing::quantize::pq::quantizer<float>>
initialize_padded_pq_quantizer_gpu(const raft::resources& dev_resources,
                                   const float* h_centroid,
                                   const float* h_full_pivot_data,
                                   size_t dim,
                                   size_t num_centers,
                                   size_t num_pq_chunks,
                                   const uint32_t* chunk_offsets) {
    DISKANN_NVTX_RANGE("initialize_padded_pq_quantizer_gpu");
    if (num_centers != 256 || num_pq_chunks == 0 || chunk_offsets[num_pq_chunks] != dim) {
        return nullptr;
    }

    // Padded coordinates are zero in the centroid and in every pivot, so they add nothing to any distance.
    const size_t subspace_dim = padded_pq_chunk_width(chunk_offsets, num_pq_chunks);
    const size_t padded_dim = subspace_dim * num_pq_chunks;
    std::vector<float> padded_centroid(padded_dim, 0.0f);
    std::vector<float> pq_code_book_subspace_major(num_centers * padded_dim, 0.0f);
    for (size_t chunk = 0; chunk < num_pq_chunks; chunk++) {
        const size_t offset = chunk_offsets[chunk];
        const size_t width = chunk_offsets[chunk + 1] - offset;
        std::memcpy(padded_centroid.data() + chunk * subspace_dim, h_centroid + offset, width * sizeof(float));
        for (size_t center = 0; center < num_centers; center++) {
            std::memcpy(pq_code_book_subspace_major.data() + (chunk * num_centers + center) * subspace_dim,
                        h_full_pivot_data + center * dim + offset, width * sizeof(float));
        }
    }

    auto d_vq_code_book =
        raft::make_device_matrix<float, uint32_t>(dev_resources, 1, static_cast<uint32_t>(padded_dim));
    auto d_pq_code_book = raft::make_device_matrix<float, uint32_t>(
        dev_resources, static_cast<uint32_t>(num_centers * num_pq_chunks), static_cast<uint32_t>(subspace_dim));
    auto d_empty = raft::make_device_matrix<uint8_t, int64_t>(dev_resources, 0, 0);

    auto stream = raft::resource::get_cuda_stream(dev_resources);
    raft::copy(d_vq_code_book.data_handle(), padded_centroid.data(), padded_dim, stream);
    raft::copy(d_pq_code_book.data_handle(), pq_code_book_subspace_major.data(), num_centers * padded_dim, stream);
    raft::resource::sync_stream(dev_resources);

    cuvs::cluster::kmeans::params km_params;
    km_params.n_clusters = static_cast<int>(num_centers);
    km_params.max_iter = 1;
    km_params.n_init = 1;
    auto pq_params = cuvs::preprocessing::quantize::pq::params(
        8, static_cast<uint32_t>(num_pq_chunks), true, true, 1,
        cuvs::preprocessing::quantize::pq::kmeans_params_variant(km_params));

    return std::make_unique<cuvs::preprocessing::quantize::pq::quantizer<float>>(
        cuvs::preprocessing::quantize::pq::quantizer<float>{
            pq_params, cuvs::neighbors::vpq_dataset<float, int64_t>(
                           std::move(d_vq_code_book), std::move(d_pq_code_book), std::move(d_empty))});
}

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
                             float* h_inflated_out /*=nullptr*/) {
    DISKANN_NVTX_RANGE("transform_padded_pq_gpu");
    auto stream = raft::resource::get_cuda_stream(dev_resources);
    const size_t subspace_dim = padded_pq_chunk_width(chunk_offsets, num_pq_chunks);
    const size_t padded_dim = subspace_dim * num_pq_chunks;

    raft::copy(d_block_data, h_block_data, cur_blk_size * dim, stream);
    RAFT_CUDA_TRY(cudaMemsetAsync(d_padded_block_data, 0, cur_blk_size * padded_dim * sizeof(float), stream));
    for (size_t chunk = 0; chunk < num_pq_chunks; chunk++) {
        const size_t width = chunk_offsets[chunk + 1] - chunk_offsets[chunk];
        if (width == 0) {
            continue;
        }
        raft::copy_matrix<float>(d_padded_block_data + chunk * subspace_dim, padded_dim,
                                 d_block_data + chunk_offsets[chunk], dim, width, cur_blk_size, stream);
    }

    auto d_padded_block_data_view = raft::make_device_matrix_view<const float, int64_t>(
        d_padded_block_data, static_cast<int64_t>(cur_blk_size), static_cast<int64_t>(padded_dim));
    auto d_compressed_data_block_view = raft::make_device_matrix_view<uint8_t, int64_t>(
        d_compressed_data_block, static_cast<int64_t>(cur_blk_size), static_cast<int64_t>(num_pq_chunks));
    cuvs::preprocessing::quantize::pq::transform(
        dev_resources, pq_quantizer, d_padded_block_data_view, d_compressed_data_block_view);

    raft::copy(h_codes_out, d_compressed_data_block, cur_blk_size * num_pq_chunks, stream);
    if (h_inflated_out != nullptr) {
        auto d_centroid_labels = raft::make_device_vector<uint32_t, int64_t>(
            dev_resources, static_cast<int64_t>(cur_blk_size));
        RAFT_CUDA_TRY(cudaMemsetAsync(d_centroid_labels.data_handle(),
                                      0,
                                      cur_blk_size * sizeof(uint32_t),
                                      stream));
        cuvs::preprocessing::quantize::pq::inverse_transform(
            dev_resources,
            pq_quantizer,
            raft::make_const_mdspan(d_compressed_data_block_view),
            raft::make_device_matrix_view<float, int64_t>(
                d_padded_block_data, static_cast<int64_t>(cur_blk_size), static_cast<int64_t>(padded_dim)),
            std::make_optional(raft::make_const_mdspan(d_centroid_labels.view())));
        for (size_t chunk = 0; chunk < num_pq_chunks; chunk++) {
            const size_t width = chunk_offsets[chunk + 1] - chunk_offsets[chunk];
            if (width == 0) {
                continue;
            }
            raft::copy_matrix<float>(d_block_data + chunk_offsets[chunk], dim,
                                     d_padded_block_data + chunk * subspace_dim, padded_dim, width, cur_blk_size,
                                     stream);
        }
        raft::copy(h_inflated_out, d_block_data, cur_blk_size * dim, stream);
    }
    raft::resource::sync_stream(dev_resources);
}

template raft::device_matrix<float, int64_t> read_bin_dataset<float,
		int64_t>(const raft::device_resources &dev_resources,
		const std::string &fname);

template raft::device_matrix<uint8_t, int64_t> read_bin_dataset<uint8_t,
		int64_t>(const raft::device_resources &dev_resources,
		const std::string &fname);

template void vamana_build_and_write<float>(
		raft::device_resources const &dev_resources,
		raft::device_matrix_view<const float, int64_t> dataset,
		std::string out_fname, int degree, int visited_size, float max_fraction,
		int iters);

template void vamana_build_and_write<uint8_t>(
		raft::device_resources const &dev_resources,
		raft::device_matrix_view<const uint8_t, int64_t> dataset,
		std::string out_fname, int degree, int visited_size, float max_fraction,
		int iters);

template int brute_force_gpu<float>(const raft::resources &dev_resources,
		const float* h_data,
        size_t n_samples,
        size_t dim,
		size_t k,
        const float* queries,
        size_t n_queries,
		int64_t* h_labels);

