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

#include "paddle/fluid/inference/api/experimental_apis/details/infer_type_conv.h"
#include "paddle/fluid/inference/api/experimental_apis/pd_infer_tensor.h"
#include "paddle/fluid/memory/allocation/allocator.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/memory/malloc.h"
#include "paddle/fluid/memory/memcpy.h"

namespace paddle_infer {

struct Tensor::Impl {
public:
  std::string name_;
  std::vector<int64_t> shape_;
  const paddle::platform::Place place_;
  std::vector<std::vector<size_t>> lod_;
  std::shared_ptr<paddle::memory::allocation::Allocation> buffer_;
  paddle::framework::proto::VarType::Type type_{};
  paddle::framework::DataLayout layout_{paddle::framework::DataLayout::kNCHW};

public:
  Impl(PlaceType place, int device_id = 0) : place_ {ConvPlaceType(place, device_id)} {}

  size_t capacity() const {
    return buffer_ == nullptr ? 0UL : buffer_->size();
  }

  int64_t numel() const {
    return std::accumulate(shape_.begin(), shape_.end(), 1, std::multiplies<int64_t>());
  }

  size_t bytes_size() const {
    return numel() * paddle::framework::SizeOfType(type_);
  }

  bool CheckCapacity() const {
    return capacity() && bytes_size() <= capacity();
  }

  void ReallocLazy(size_t capacity = 0) {
    if (capacity) {
      if (bytes_size() > capacity) {
        buffer_ = paddle::memory::AllocShared(place_, capacity);
      }
    } else {
      CHECK(bytes_size() > 0);
      ReallocLazy(bytes_size());
    }
  }

  template <typename T>
  bool CheckDataType() {
    return type_ == paddle::framework::DataTypeTrait<T>::DataType();
  }

  template <typename T>
  const T* data() const {
    CHECK(CheckDataType<T>());
    return reinterpret_cast<const T*>(reinterpret_cast<uintptr_t>(buffer_->ptr()));
  }

  template <typename T>
  T* mutable_data() {
    if (!CheckDataType<T>()) {
      type_ = paddle::framework::DataTypeTrait<T>::DataType();
    }
    ReallocLazy();
    CHECK(CheckCapacity());
    return reinterpret_cast<T*>(reinterpret_cast<uintptr_t>(buffer_->ptr()));
  }

};

Tensor::Tensor() : Tensor{PlaceType::kHost} {}

Tensor::Tensor(PlaceType place) : Tensor{place, 0} {}

Tensor::Tensor(PlaceType place, int device_id) : impl_{std::unique_ptr<Impl>(new Impl{place, device_id})} {}

void Tensor::Reshape(const std::vector<int64_t>& shape) {
  impl_->shape_ = shape;
}

const std::vector<int64_t>& Tensor::shape() const {
  return impl_->shape_;
}

void Tensor::SetLoD(const std::vector<std::vector<size_t>>& lod) {
  impl_->lod_ = lod;
}

const std::vector<std::vector<size_t>>& Tensor::lod() const {
  return impl_->lod_;
}

template <typename T>
const T* Tensor::data() const {
  return impl_->data();
}

template <typename T>
T* Tensor::mutable_data() {
  return impl_->mutable_data();
}

template <typename T>
int64_t Tensor::CopyDataFromHost(const T* src) {
  T* dst{mutable_data<T>()};
  size_t bytes_size{impl_->bytes_size()};
  paddle::memory::Copy(impl_->place_, dst, paddle::CPUPlace(), src, bytes_size);
  return bytes_size;
}

template <typename T>
int64_t Tensor::CopyDataToHost(T* dst) const {
  const T* src{data<T>()};
  size_t bytes_size{impl_->bytes_size()};
  paddle::memory::Copy(paddle::CPUPlace(), dst, impl_->place_, src, bytes_size);
  return bytes_size;
}

void Tensor::CopyDataFrom(const Tensor& tensor) {
  Reshape(tensor.shape());
  paddle::memory::Copy(impl_->place_, mutable_data<void>(), ConvPlaceType(PlaceType::kHost, tensor.device_id()), tensor.data<void>(), bytes_size(tensor));
}

void Tensor::SetName(const std::string& name) {
  impl_->name_ = name;
}

const std::string& Tensor::name() const {
  return impl_->name_;
}

int Tensor::device_id() const {
  return GetDeviceID(impl_->place_);
}

PlaceType Tensor::place() const {
  return ConvPlaceType(impl_->place_);
}

DataType Tensor::type() const {
  return ConvDataType(impl_->type_);
}

size_t Tensor::capacity() const {
  return impl_->capacity();
}


class Tensor::Utils {
public:
  static void ShallowCopy(Tensor* dst, const paddle::framework::LoDTensor& src) {
    Tensor::Impl* impl{dst->impl_.get()};
    impl->shape_ = paddle::framework::vectorize<int64_t>(src.dims());
    impl->buffer_ = src.Holder();
    impl->type_ = src.type();
    impl->layout_ = src.layout();
    for (const auto& lod: src.lod()) {
      impl->lod_.emplace_back(lod);
    }
  }

  static void ShallowCopy(paddle::framework::LoDTensor* dst, const Tensor& src) {
    dst->Resize(paddle::framework::make_ddim(src.impl_->shape_));
    dst->ResetHolderWithType(src.impl_->buffer_, src.impl_->type_);
    dst->set_lod(src.impl_->lod_);
    dst->set_layout(src.impl_->layout_);
  }
};

int64_t numel(const Tensor& tensor) {
  return std::accumulate(tensor.shape().begin(), tensor.shape().end(), 1, std::multiplies<int64_t>());
}

size_t bytes_size(const Tensor& tensor) {
  return numel(tensor) * SizeOfDataType(tensor.type());
}

}
