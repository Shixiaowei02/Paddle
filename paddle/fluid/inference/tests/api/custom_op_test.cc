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

int main() {
  std::cout << "here!!" << std::endl;
  const std::string dso_name{"/shixiaowei02/Paddle-custom-operator/Paddle/build_gpu/custom_relu_module_setup_pd_.so"};
  paddle_infer::experimental::LoadCustomOperatorLib(dso_name);
  std::cout << "end!!" << std::endl;

  paddle::AnalysisConfig config;
  config.EnableUseGpu(100, 0);
  config.SetModel("/shixiaowei02/Paddle-custom-operator/Paddle/build_gpu/shixiaowei02/custom_op_inference/custom_relu.pdmodel",
                  "/shixiaowei02/Paddle-custom-operator/Paddle/build_gpu/shixiaowei02/custom_op_inference/custom_relu.pdiparams");

   paddle::CreatePaddlePredictor(config);

  return 0;
}