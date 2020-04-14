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
#include <vector>
#include "paddle/fluid/memory/allocation/allocator.h"
#include "paddle/fluid/memory/detail/buddy_allocator.h"
#include "paddle/fluid/memory/detail/system_allocator.h"
#include "paddle/fluid/platform/gpu_info.h"

namespace paddle {
namespace memory {
namespace allocation {

class ThreadLocalAllocatorImpl
    : public std::enable_shared_from_this<ThreadLocalAllocatorImpl> {
 public:
  explicit ThreadLocalAllocatorImpl(const platform::Place& p) : place_(p) {
    if (platform::is_gpu_place(place_)) {
      buddy_allocator_.reset(new memory::detail::BuddyAllocator(
          std::unique_ptr<memory::detail::SystemAllocator>(
              new memory::detail::GPUAllocator(
                  boost::get<platform::CUDAPlace>(place_).device)),
          platform::GpuMinChunkSize(), platform::GpuMaxChunkSize()));
    } else {
      LOG(FATAL) << "Thread local allocator only supports CUDAPlace now.";
    }
  }
  Allocation* AllocateImpl(size_t size);
  void FreeImpl(Allocation* allocation);

 private:
  std::unique_ptr<memory::detail::BuddyAllocator> buddy_allocator_;
  platform::Place place_;
};

class CUDAThreadLocalAllocatorPool {
 public:
  static CUDAThreadLocalAllocatorPool& Instance() {
    static thread_local CUDAThreadLocalAllocatorPool pool;
    return pool;
  }

  std::shared_ptr<ThreadLocalAllocatorImpl> Get(int gpu_id) {
    auto pos = std::distance(
        devices_.begin(), std::find(devices_.begin(), devices_.end(), gpu_id));
    PADDLE_ENFORCE_LT(pos, devices_.size());
    std::call_once(*init_flags_[pos], [this, pos, gpu_id] {
      platform::SetDeviceId(devices_[pos]);
      allocators_[pos].reset(
          new ThreadLocalAllocatorImpl(platform::CUDAPlace(gpu_id)));
    });
    return allocators_[pos];
  }

 private:
  CUDAThreadLocalAllocatorPool() : devices_(platform::GetSelectedDevices()) {
    auto gpu_num = devices_.size();
    allocators_.resize(gpu_num);
    init_flags_.reserve(gpu_num);
    for (size_t i = 0; i < gpu_num; ++i) {
      init_flags_.emplace_back(new std::once_flag());
    }
  }
  std::vector<int> devices_;
  std::vector<std::unique_ptr<std::once_flag>> init_flags_;
  std::vector<std::shared_ptr<ThreadLocalAllocatorImpl>> allocators_;
};

class CUDAThreadLocalAllocator : public Allocator {
 public:
  explicit CUDAThreadLocalAllocator(const platform::CUDAPlace& p)
      : gpu_id_(p.device) {}

 protected:
  Allocation* AllocateImpl(size_t size) override {
    return CUDAThreadLocalAllocatorPool::Instance().Get(gpu_id_)->AllocateImpl(
        size);
  }
  void FreeImpl(Allocation* allocation) override {
    CUDAThreadLocalAllocatorPool::Instance().Get(gpu_id_)->FreeImpl(allocation);
  }

 private:
  int gpu_id_;
};

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
