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

#include "paddle/fluid/inference/api/experimental_apis/pd_infer_type.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle_infer {

size_t SizeOfDataType(DataType type) {
  switch (type) {
    case DataType::FLOAT32:
      return 32;
    case DataType::INT64:
      return 64;
    case DataType::INT32:
      return 32;
    case DataType::UINT8:
      return 8;
    case DataType::INT8:
      return 8;
    case DataType::FLOAT16:
      return 16;
    default:
      PADDLE_THROW(paddle::platform::errors::Unimplemented(
          "Unsupported precision type. Now only supports FP16, FP32, INT8, INT32 and "
          "INT64."));
      return 0;
  }
}

}