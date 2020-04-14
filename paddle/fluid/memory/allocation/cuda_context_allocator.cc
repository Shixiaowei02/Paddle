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

#include "paddle/fluid/memory/allocation/cuda_context_allocator.h"
#include <utility>
#include "paddle/fluid/memory/malloc.h"

namespace paddle {
namespace memory {
namespace allocation {

class CUDAContextAllocation : public Allocation {
 public:
  CUDAContextAllocation(AllocationPtr allocation,
                        std::shared_ptr<CUDAContextAllocator> allocator)
      : Allocation(allocation->ptr(), allocation->size(), allocation->place()),
        underlying_allocation_(std::move(allocation)),
        allocator_(allocator),
        context_(allocator_->CUDAContext()) {}

  ~CUDAContextAllocation() {
    PADDLE_ENFORCE_NOT_NULL(
        context_->RawStream(),
        "Didn't set device context for CUDAContextAllocation");
    auto* p_allocation = underlying_allocation_.release();
    VLOG(4) << "Adding callback to delete CUDAContextAllocation at "
            << p_allocation;
    context_->Stream()->AddCallback([p_allocation] {
      VLOG(4) << "Delete CUDAContextAllocation at " << p_allocation;
      AllocationDeleter()(p_allocation);
    });
  }

 private:
  AllocationPtr underlying_allocation_;
  const std::shared_ptr<CUDAContextAllocator> allocator_;
  const std::shared_ptr<platform::CUDAContext> context_;  // not owned.
};

Allocation* CUDAContextAllocator::AllocateImpl(size_t size) {
  PADDLE_ENFORCE_NOT_NULL(context_->RawStream(),
                          "Didn't set stream for CUDAContextAllocator");
  const auto& stream = context_->Stream();
  platform::CUDADeviceGuard guard(place_.device);
  auto allocation = new CUDAContextAllocation(memory::Alloc(place_, size),
                                              shared_from_this());
  stream->RecordEvent(event_);
  stream->WaitEvent(event_);
  return allocation;
}

void CUDAContextAllocator::FreeImpl(Allocation* allocation) {
  delete allocation;
}

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
