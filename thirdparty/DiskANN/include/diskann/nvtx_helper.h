// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

// NVTX range helper: emits a raft NVTX scope when built with cuVS/raft, and
// compiles to a no-op otherwise (raft headers are unavailable in CPU-only builds).
//
// Usage: place `DISKANN_NVTX_RANGE("name");` as the first statement of a scope.
// The range begins where it is declared and ends when the enclosing scope exits.
#ifdef KNOWHERE_WITH_CUVS
#include <string>

#include <raft/core/nvtx.hpp>
// raft::common::nvtx::range is a class template with a default Domain argument,
// so CTAD needs the explicit empty argument list (range<>). Its constructor
// also takes a printf-style `const char*`, so wrap the argument to accept both
// string literals and std::string.
#define DISKANN_NVTX_RANGE(name) raft::common::nvtx::range<> _diskann_nvtx_range__ { std::string(name).c_str() }
#else
#define DISKANN_NVTX_RANGE(name) \
    do {                         \
    } while (0)
#endif
