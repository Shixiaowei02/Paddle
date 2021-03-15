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
#include "paddle/fluid/framework/framework.pb.h"
#include "paddle/fluid/platform/place.h"

namespace paddle_infer {

inline DataType ConvDataType(paddle::framework::proto::VarType::Type type) {
  switch (type) {
    case framework::proto::VarType_Type_FP16:
      return DataType::FLOAT16;
    case framework::proto::VarType_Type_FP32:
      return DataType::FLOAT32;
    case framework::proto::VarType_Type_INT8:
      return DataType::INT8;
    case framework::proto::VarType_Type_INT32:
      return DataType::INT32;
    case framework::proto::VarType_Type_INT64:
      return DataType::INT64;
    default:
      PADDLE_THROW(platform::errors::Unimplemented(
          "Unsupported precision type. Now only supports FP16, FP32, INT8, INT32 and "
          "INT64."));
      return DataType::UNK;
  }
}

inline paddle::framework::proto::VarType::Type ConvDataType(DataType type) {
  switch (type) {
    case DataType::FLOAT16:
      return framework::proto::VarType_Type_FP16;
    case DataType::FLOAT32:
      return framework::proto::VarType_Type_FP32;
    case DataType::INT8:
      return framework::proto::VarType_Type_INT8;
    case DataType::INT32:
      return framework::proto::VarType_Type_INT32;
    case DataType::INT64:
      return framework::proto::VarType_Type_INT64;
    default:
      PADDLE_THROW(platform::errors::Unimplemented(
          "Unsupported precision type. Now only supports FP16, FP32, INT8, INT32 and "
          "INT64."));
      return static_cast<paddle::framework::proto::VarType::Type>(-1);
}

inline PlaceType ConvPlaceType(const paddle::framework::platform::Place& place) {
  if (paddle::platform::is_cpu_place(place)) {
    return PlaceType::kHost;
  } else if (paddle::platform::is_gpu_place(place)) { 
    return PlaceType::kGPU;
  } else if (paddle::platform::is_xpu_place(place)) { 
    return PlaceType::kXPU;
  }
  return PlaceType::kUnk;

inline paddle::framework::platform::Place ConvPlaceType(PlaceType place, int device_id) {
  paddle::framework::platform::Place ret;
  switch (place) {
    case PlaceType::kHost:
      return paddle::framework::platform::CPUPlace();
    case PlaceType::kGPU:
      return paddle::framework::platform::CUDAPlace(device_id);
    case PlaceType::kXPU:
      return paddle::framework::platform::XPUPlace(device_id);
  }
  return paddle::framework::platform::Place();
}

}