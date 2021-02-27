/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/inference/tests/api/tester_helper.h"
#include "paddle/fluid/inference/api/experimental/custom_operator.h"

using paddle_infer::Config;
using paddle_infer::Predictor;
using paddle_infer::CreatePredictor;

void run(Predictor *predictor, const std::vector<float> &input,
         const std::vector<int> &input_shape, std::vector<float> *out_data) {
  int input_num = std::accumulate(input_shape.begin(), input_shape.end(), 1,
                                  std::multiplies<int>());

  auto input_names = predictor->GetInputNames();
  auto input_t = predictor->GetInputHandle(input_names[0]);
  input_t->Reshape(input_shape);
  input_t->CopyFromCpu(input.data());

  CHECK(predictor->Run());

  auto output_names = predictor->GetOutputNames();
  // there is only one output of Resnet50
  auto output_t = predictor->GetOutputHandle(output_names[0]);
  std::vector<int> output_shape = output_t->shape();
  int out_num = std::accumulate(output_shape.begin(), output_shape.end(), 1,
                                std::multiplies<int>());

  out_data->resize(out_num);
  output_t->CopyToCpu(out_data->data());
}

int main() {
  std::cout << "here!!" << std::endl;
  const std::string dso_name{"/shixiaowei02/Paddle-custom-op-src/Paddle-Inference-Demo/c++/custom-operator/build/libpd_infer_custom_op.so"};
  paddle_infer::experimental::LoadCustomOperatorLib(dso_name);
  std::cout << "end!!" << std::endl;

  paddle::AnalysisConfig config;
  config.EnableUseGpu(100, 0);
  config.SetModel("/shixiaowei02/Paddle-custom-operator/Paddle/build_gpu/shixiaowei02/custom_op_inference/custom_relu.pdmodel",
                  "/shixiaowei02/Paddle-custom-operator/Paddle/build_gpu/shixiaowei02/custom_op_inference/custom_relu.pdiparams");

  auto predictor{paddle_infer::CreatePredictor(config)};

  std::vector<int> input_shape = {1, 1, 1, 2};

  // init 0 for the input.
  std::vector<float> input_data(1 * 1 * 1 * 2, -1);

  std::vector<float> out_data;
  run(predictor.get(), input_data, input_shape, &out_data);

  for (auto e : out_data) {
    LOG(INFO) << e << std::endl;
  }

  return 0;
}
