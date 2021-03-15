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
#include "paddle/fluid/platform/errors.h"
#include "paddle/fluid/memory/malloc.h"

namespace paddle_infer {

struct Tensor::Impl {
public:
  size_t offset_{};
  std::string name_;
  std::vector<int64_t> shape_;
  const paddle::framework::platform::Place place_;
  std::vector<std::vector<size_t>> lod_;
  std::shared_ptr<memory::Allocation> buffer_;
  paddle::framework::proto::VarType::Type type_{};
  paddle::framework::DataLayout layout_{paddle::framework::DataLayout::kNCHW};

public:
  Impl(PlaceType place, int device_id = 0) : place_ {ConvPlaceType{place, device_id}} {}

  size_t capacity() const {
    return buffer_ == nullptr ? 0UL : buffer_->size() - offset_;
  }

  int64_t numel() const {
    return std::accumulate(shape_.begin(), shape_.end(), 1, std::multiplies<int64_t>());
  }

  size_t bytes_size() const {
    return numel() * paddle::framework::SizeOfType(type_);
  }

  bool capacity_enough() const {
    return bytes_size() <= capacity();
  }

  void* mutable_data() {
    CHECK(buffer_);
    return buffer_->ptr();
  }

  void* ReallocLazy(size_t capacity = 0) {
    if (capacity) {
      CHECK_GE(capacity, bytes_size());
      if (!capacity_enough()) {
        buffer_ = paddle::memory::AllocShared(place_, capacity);
        offset_ = 0;
      }
    } else {
      CHECK(bytes_size() > 0);
      ReallocLazy(bytes_size());
    }
    return mutable_data();
  }

};

Tensor::Tensor() : Tensor{PlaceType::kHost} {}

Tensor::Tensor(PlaceType place) : Tensor{place, 0} {}

Tensor::Tensor(PlaceType place, int device_id) : impl_{std::unique_ptr(newImpl{place, device_id})} {}

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
  CHECK(capacity());
  return impl_->buffer_->ptr();
}

template <typename T>
T* Tensor::mutable_data() {
  ReallocLazy();
  return impl_->buffer_->ptr();
}

template <typename T>
int64_t Tensor::CopyFromHost(const T* data) {

}

template <typename T>
int64_t Tensor::CopyToHost(T* data) const {

}

void Tensor::CopyDataFrom(const Tensor& tensor) {

}

void Tensor::SetName(const std::string& name) {
  impl_->name_ = name;
}

const std::string& Tensor::name() const {
  return impl_->name_;
}

int Tensor::device_id() const {
  return impl_->device_id_;
}

PlaceType Tensor::place() const {
  return ConvPlaceType(impl_->place_);
}

DataType Tensor::type() const {
  return ConvDataType(impl_->type_);
}


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

}
