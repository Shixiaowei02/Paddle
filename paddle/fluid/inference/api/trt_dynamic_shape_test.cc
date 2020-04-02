/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

//#include <gflags/gflags.h>
//#include <glog/logging.h>
//#include <gtest/gtest.h>


#include <gflags/gflags.h>
#include <glog/logging.h>
//#include "paddle/include/paddle_inference_api.h"
#include "paddle/fluid/inference/api/paddle_inference_api.h"
#include <chrono>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>
#include <utility>
#include <thread>
#include <atomic>
#include <mutex>


namespace paddle {
namespace inference {

void test_main() {

  std::mutex g_mutex;
  std::atomic<int> cnt(0);
  std::vector<std::thread> threads;
  const int num_threads = 20;
  for (int tid = 0; tid < num_threads; ++tid) {
    threads.emplace_back([&, tid]() {

  g_mutex.lock();
  std::string model_dir = "/shixiaowei02/Paddle_trt_stream/Paddle/paddle/fluid/inference/tests/api/test_trt_dy_conv";
  AnalysisConfig config;
  config.EnableUseGpu(100, 0);
  config.SetModel(model_dir);
  config.SwitchUseFeedFetchOps(false);
  config.BindGpuStreamToThread();
  // Set the input's min, max, opt shape
  std::map<std::string, std::vector<int>> min_input_shape = {
      {"image", {1, 1, 3, 3}}};
  std::map<std::string, std::vector<int>> max_input_shape = {
      {"image", {1, 1, 10, 10}}};
  std::map<std::string, std::vector<int>> opt_input_shape = {
      {"image", {1, 1, 3, 3}}};
  config.EnableTensorRtEngine(1 << 30, 1, 1,
                              AnalysisConfig::Precision::kFloat32, false, true);

  config.SetTRTDynamicShapeInfo(min_input_shape, max_input_shape,
                                opt_input_shape);
  auto predictor = CreatePaddlePredictor(config);
      g_mutex.unlock();
    std::cout << "mutex un-locked."<<std::endl;
      cnt ++;
      while(cnt != num_threads) {} 

 
  for (int cnt = 0; cnt < 1000; cnt++) {
  auto input_names = predictor->GetInputNames();
  int channels = 1;
  int height = 3;
  int width = 3;
  int input_num = channels * height * width * 1;

  float *input = new float[input_num];
  memset(input, 0, input_num * sizeof(float));
  auto input_t = predictor->GetInputTensor(input_names[0]);
  input_t->Reshape({1, channels, height, width});
  input_t->copy_from_cpu(input);

  predictor->ZeroCopyRun();

  std::vector<float> out_data;
  auto output_names = predictor->GetOutputNames();
  auto output_t = predictor->GetOutputTensor(output_names[0]);
  std::vector<int> output_shape = output_t->shape();
  int out_num = std::accumulate(output_shape.begin(), output_shape.end(), 1,
                                std::multiplies<int>());
  out_data.resize(out_num);
  output_t->copy_to_cpu(out_data.data());
  float sum = 0;
  for (auto i : out_data) {
    sum += i;
  }
  CHECK_NEAR(sum, -10.3782, 0.001);
  LOG(INFO) << "sum of data: " << sum;
  }


    });
  }
  for (int i = 0; i < num_threads; ++i) {
    threads[i].join();
  }

}
}  // namespace inference
}  // namespace paddle

int main() {
  for (int i = 0; i < 10; i ++) {
  paddle::inference::test_main();
  }
}
