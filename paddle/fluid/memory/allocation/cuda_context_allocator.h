// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once
#include <memory>
#include <unordered_map>
#include "paddle/fluid/memory/allocation/allocator.h"
#include "paddle/fluid/platform/cuda_device_guard.h"
#include "paddle/fluid/platform/device_context.h"

namespace paddle {
namespace memory {
namespace allocation {

class CUDAContextAllocator
    : public Allocator,
      public std::enable_shared_from_this<CUDAContextAllocator> {
 public:
  explicit CUDAContextAllocator(
      const std::shared_ptr<platform::CUDAContext> context)
      : context_(context) {
    place_ = context->Place();
    platform::CUDADeviceGuard guard(place_.device);
    PADDLE_ENFORCE_CUDA_SUCCESS(
        cudaEventCreate(&event_, cudaEventDisableTiming),
        platform::errors::External(
            "Create event failed in CUDAContextAllocator"));
  }

  ~CUDAContextAllocator() {
    if (event_) {
      platform::CUDADeviceGuard guard(place_.device);
      PADDLE_ENFORCE_CUDA_SUCCESS(
          cudaEventDestroy(event_),
          "Destory event failed in CUDAContextAllocator destroctor");
    }
  }

  std::shared_ptr<platform::CUDAContext> CUDAContext() { return context_; }

 protected:
  Allocation* AllocateImpl(size_t size) override;
  void FreeImpl(Allocation* allocation) override;

 private:
  platform::CUDAPlace place_;
  cudaEvent_t event_{nullptr};
  const std::shared_ptr<platform::CUDAContext> context_;  // not owned.
};

class CUDAThreadLocalAllocatorPool {
 public:
  static CUDAThreadLocalAllocatorPool& Instance() {
    static thread_local CUDAThreadLocalAllocatorPool pool;
    return pool;
  }

  AllocationPtr Alloc(const std::shared_ptr<platform::CUDAContext> context,
                      size_t size) {
    if (!allocators_.count(context.get())) {
      allocators_[context.get()] = std::shared_ptr<CUDAContextAllocator>(
          new CUDAContextAllocator(context));
    }
    auto allocator = allocators_.find(context.get());
    AllocationPtr allocation = allocator->second->Allocate(size);
    return allocation;
  }

 private:
  CUDAThreadLocalAllocatorPool() = default;
  std::unordered_map<platform::CUDAContext*,
                     std::shared_ptr<CUDAContextAllocator>>
      allocators_;
};

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
