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

#include "paddle/fluid/inference/api/experimental_apis/details/infer_type_conv.h"

namespace paddle_infer {

DataType ConvDataType(paddle::framework::proto::VarType::Type type) {
  switch (type) {
    case paddle::framework::proto::VarType_Type_FP16:
      return DataType::FLOAT16;
    case paddle::framework::proto::VarType_Type_FP32:
      return DataType::FLOAT32;
    case paddle::framework::proto::VarType_Type_INT8:
      return DataType::INT8;
    case paddle::framework::proto::VarType_Type_INT32:
      return DataType::INT32;
    case paddle::framework::proto::VarType_Type_INT64:
      return DataType::INT64;
    default:
      PADDLE_THROW(paddle::platform::errors::Unimplemented(
          "Unsupported precision type. Now only supports FP16, FP32, INT8, INT32 and "
          "INT64."));
      return DataType::UNK;
  }
}

paddle::framework::proto::VarType::Type ConvDataType(DataType type) {
  switch (type) {
    case DataType::FLOAT16:
      return paddle::framework::proto::VarType_Type_FP16;
    case DataType::FLOAT32:
      return paddle::framework::proto::VarType_Type_FP32;
    case DataType::INT8:
      return paddle::framework::proto::VarType_Type_INT8;
    case DataType::INT32:
      return paddle::framework::proto::VarType_Type_INT32;
    case DataType::INT64:
      return paddle::framework::proto::VarType_Type_INT64;
    default:
      PADDLE_THROW(paddle::platform::errors::Unimplemented(
          "Unsupported precision type. Now only supports FP16, FP32, INT8, INT32 and "
          "INT64."));
      return static_cast<paddle::framework::proto::VarType::Type>(-1);
}

PlaceType ConvPlaceType(const paddle::platform::Place& place) {
  if (paddle::platform::is_cpu_place(place)) {
    return PlaceType::kHost;
  } else if (paddle::platform::is_gpu_place(place)) { 
    return PlaceType::kGPU;
  } else if (paddle::platform::is_xpu_place(place)) { 
    return PlaceType::kXPU;
  }
  return PlaceType::kUnk;
}

paddle::platform::Place ConvPlaceType(PlaceType place, int device_id) {
  paddle::platform::Place ret;
  switch (place) {
    case PlaceType::kHost:
      return paddle::platform::CPUPlace();
    case PlaceType::kGPU:
      return paddle::platform::CUDAPlace(device_id);
    case PlaceType::kXPU:
      return paddle::platform::XPUPlace(device_id);
  }
  return paddle::platform::Place{};
}

}