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

#include "pd_infer_type.h"
#include "pd_infer_declare.h"

namespace paddle_infer {

class PD_INFER_DECL Tensor {
public:
  explicit Tensor(const PlaceType& place);

  void Reshape(const std::vector<int64_t>& shape);

  std::vector<int64_t> shape() const;

  void SetLoD(const std::vector<std::vector<size_t>>& x);

  std::vector<std::vector<size_t>> lod() const;

  template <typename T>
  T* mutable_data();

  template <typename T>
  const T* data() const;

  template <typename T>
  void CopyFromHost(const T* data);

  template <typename T>
  void CopyToHost(T* data) const;

  void CopyDataFrom(const Tensor& tensor);

private:
  struct Impl;
  Tensor(const Tensor&);
  Tensor& operator=(const Tensor&);
  std::unique_ptr<Impl> impl_;
};

}
