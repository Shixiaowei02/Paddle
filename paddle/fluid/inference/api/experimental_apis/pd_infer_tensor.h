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

#pragma once

#include <vector>
#include <memory>

#include "pd_infer_type.h" // NOLINT
#include "pd_infer_declare.h" // NOLINT

namespace paddle_infer {

class PD_INFER_DECL Tensor {
public:
  Tensor();

  explicit Tensor(PlaceType place);

  Tensor(PlaceType place, int device_id);

  void Reshape(const std::vector<int64_t>& shape);

  const std::vector<int64_t>& shape() const;

  void SetLoD(const std::vector<std::vector<size_t>>& lod);

  const std::vector<std::vector<size_t>>& lod() const;

  template <typename T>
  const T* data() const;

  template <typename T>
  T* mutable_data();

  template <typename T>
  int64_t CopyDataFromHost(const T* src);

  template <typename T>
  int64_t CopyDataToHost(T* dst) const;

  void CopyDataFrom(const Tensor& tensor);

  void SetName(const std::string& name);

  const std::string& name() const;

  int device_id() const;

  size_t capacity() const;

  PlaceType place() const;

  DataType type() const;

private:
  struct Impl;
  class Utils;
  friend class Utils;
  Tensor(const Tensor&);
  Tensor& operator=(const Tensor&);
  const std::unique_ptr<Impl> impl_;
};

int64_t numel(const Tensor& tensor);

size_t bytes_size(const Tensor& tensor);

}
