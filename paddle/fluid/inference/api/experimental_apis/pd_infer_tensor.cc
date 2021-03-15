// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include <memory>

#include "paddle/fluid/inference/api/experimental_apis/pd_infer_tensor.h"
#include "paddle/fluid/memory/allocation/allocator.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/data_type.h"

namespace paddle_infer {

struct Tensor::Impl {
public:
  size_t offset() const {
    return offset_;
  }
  void SetOffset(size_t offset) {
    offset_ = offset;
  }
  const std::vector<int64_t>& shape() const {
    return shape_;
  }
  void SetShape(std::vector<int64_t>& shape) {
    shape_ = shape;
  }
  void SetLoD(std::vector<std::vector<size_t>>& lod) {
    lod_ = lod;
  }
  const std::vector<std::vector<size_t>>& lod() const {
    return lod_;
  }
  void SetSharedBuffer(std::shared_ptr<memory::Allocation> buffer) {
    shared_buffer_ = buffer;
  }
  std::shared_ptr<memory::Allocation> GetSharedBuffer() const {
    return shared_buffer_;
  }
  const void SetType(paddle::framework::proto::VarType::Type type) {
    type_ = type;
  }
  paddle::framework::proto::VarType::Type type() const {
    return type_;
  }
  const void SetDataLayout(paddle::framework::DataLayout layout) {
    layout_ = layout;
  }
  paddle::framework::DataLayout GetLayout() const {
    return layout_;
  }
public:
  size_t memory_size() const {
    return shared_buffer == nullptr ? 0UL : shared_buffer->size() - offset;
  }
  void CheckMemorySize() const {
    PADDLE_ENFORCE_NOT_NULL(shared_buffer(), platform::errors::PreconditionNotMet(
                                        "Tensor holds no memory. "
                                        "Call Tensor::mutable_data firstly."));
    PADDLE_ENFORCE_LE(
        numel() * paddle::framework::SizeOfType(type, memory_size(),
        platform::errors::PreconditionNotMet(
            "Tensor's dimension is out of bound."
            "Tensor's dimension must be equal or less than the size of its "
            "memory."
            "But received  Tensor's dimension is d%, memory's size is %d.",
            numel() * SizeOfType(type()), memory_size()));
  }

private:
  size_t offset_;
  std::vector<int64_t> shape_;
  std::vector<std::vector<size_t>> lod_;
  std::shared_ptr<memory::Allocation> shared_buffer_;
  paddle::framework::proto::VarType::Type type_{};
  paddle::framework::DataLayout layout_{paddle::framework::DataLayout::kNCHW};
};

class Tensor::Utils {
public:
  static ShallowCopy(Tensor* dst, const paddle::framework::LoDTensor& src) {
    dst->impl_->SetOffset(lod_tensor.offset());
    dst->impl_->SetShape(lod_tensor.dims().vectorize<int64_t>());
    dst->impl_->SetSharedBuffer(lod_tensor.Holder());
    dst->impl_->SetLoD(lod_tensor.lod());
    dst->impl_->SetType(lod_tensor.type());
    dst->impl_->SetDataLayout(lod_tensor.layout());
  }

  static ShallowCopy(paddle::framework::LoDTensor* dst, const Tensor& src) {
    CHECK(src.impl_->offset());
    dst->Resize(src.impl_->shape());
    dst->ResetHolderWithType(src.impl_->shared_buffer(), src.impl_->type());
    dst->set_lod(src.impl_->lod());
    dst->set_layout(src.impl_->layout());
  }
};

Tensor::Tensor() {

}

void Tensor::Reshape(const std::vector<int64_t>& shape) {
  impl_->shape = shape;
}

const std::vector<int64_t>& Tensor::shape() const {
  return impl_->shape;
}

void Tensor::SetLoD(const std::vector<std::vector<size_t>>& lod) {
  impl_->lod = lod;
}

const std::vector<std::vector<size_t>>& Tensor::lod() const {
  return impl_->lod;
}

template <typename T>
const T* Tensor::data() const {
  return impl_->shared_buffer.
}

template <typename T>
int64_t Tensor::CopyFromHost(const T* data) {

}

template <typename T>
int64_t Tensor::CopyToHost(T* data) const {

}

void Tensor::CopyDataFrom(const Tensor& tensor) {

}


}
