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

#include "pd_infer_tensor.h"
#include "pd_infer_config.h"

namespace paddle_infer {

class PD_INFER_DECL Predictor {
public:
  const std::vector<std::string>& GetInputNames() const = 0;

  const std::vector<std::string>& GetOutputNames() const = 0;

  Tensor* GetInputHandle(const std::string& name) = 0;

  const Tensor* GetOutputHandle(const std::string& name) const = 0;

  bool Run() = 0;

  std::shared_ptr<Predictor> Clone() const = 0;

  void ClearIntermediateTensor() = 0;

  uint64_t TryShrinkMemory() = 0;
};

PD_INFER_DECL std::shared_ptr<Predictor> CreatePredictor(
    const Config& config); 

} // namespace paddle_infer
