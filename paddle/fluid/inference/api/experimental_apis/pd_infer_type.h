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

namespace paddle_infer {

/// \brief Paddle data type.
enum class DataType {
  UNK = -1,
  FLOAT32,
  INT64,
  INT32,
  UINT8,
  INT8,
  FLOAT16,
};

enum class PlaceType { kUnk = -1, kHost, kGPU, kXPU };

enum class Precision {
  kUnk = -1,
  kFloat32,  ///< fp32
  kInt8,         ///< int8
  kHalf,         ///< fp16
};

} // namespace paddle_infer
