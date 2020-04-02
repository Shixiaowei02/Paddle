// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

// DEFINE_string(model_dir, "./ernieL3H128_model", "model directory");
//DEFINE_string(model_dir, "./4slot_l3h128_model", "model directory");
DEFINE_string(model_dir, "/shixiaowei02/xingzhaolong/mix_dx_all_click_posttrain_dev_test_L3_step_310w_pruned", "model directory");
DEFINE_string(data, "/shixiaowei02/xingzhaolong/ras_batch_40_data", "input data path");
DEFINE_int32(repeat, 1, "repeat");
DEFINE_bool(output_prediction, false, "Whether to output the prediction results.");
DEFINE_bool(use_gpu, false, "Whether to use GPU for prediction.");
DEFINE_int32(device, 0, "device.");

using Time = decltype(std::chrono::high_resolution_clock::now());
Time time() { return std::chrono::high_resolution_clock::now(); };
double time_diff(Time t1, Time t2) {
  typedef std::chrono::microseconds ms;
  auto diff = t2 - t1;
  ms counter = std::chrono::duration_cast<ms>(diff);
  return counter.count() / 1000.0 / 1000.0;
}


template <typename T>
void GetValueFromStream(std::stringstream *ss, T *t) {
    (*ss) >> (*t);
}

template <>
void GetValueFromStream<std::string>(std::stringstream *ss, std::string *t) {
    *t = ss->str();
}

// Split string to vector
template <typename T>
void Split(const std::string &line, char sep, std::vector<T> *v) {
    std::stringstream ss;
    T t;
    for (auto c : line) {
        if (c != sep) {
            ss << c;
        } else {
            GetValueFromStream<T>(&ss, &t);
            v->push_back(std::move(t));
            ss.str({});
            ss.clear();
        }
    }

    if (!ss.str().empty()) {
        GetValueFromStream<T>(&ss, &t);
        v->push_back(std::move(t));
        ss.str({});
        ss.clear();
    }
}

template <typename T>
bool ParseData(const std::string &field, std::pair<std::vector<T>, std::vector<int>> *temp_data) {
    std::vector<std::string> data;
    Split(field, ':', &data);
    if (data.size() < 2) return false;

    std::string shape_str = data[0];

    std::vector<int> shape;
    Split(shape_str, ' ', &shape);

    std::string mat_str = data[1];

    std::vector<T> mat;
    Split(mat_str, ' ', &mat);
       
    (*temp_data) = std::make_pair(mat, shape);
    return true;
}

bool LoadInputData(
        std::vector<std::pair<std::vector<int64_t>, std::vector<int> >>* all_datas, 
        std::vector<std::pair<std::vector<float>, std::vector<int> >>* data) {
    if (FLAGS_data.empty()) {
        LOG(ERROR) << "please set input data path";
        return false;
    }

    std::ifstream fin(FLAGS_data);
    std::string line;

    int lineno = 0;
    while (std::getline(fin, line)) {
        std::vector<std::string> fields;
        Split(line, ';', &fields);
        int i = 0;
        std::pair<std::vector<int64_t>, std::vector<int>> temp_data1;
        ParseData<int64_t>(fields[i++], &temp_data1); 
        all_datas->push_back(std::move(temp_data1));

        std::pair<std::vector<int64_t>, std::vector<int>> temp_data2;
        ParseData<int64_t>(fields[i++], &temp_data2); 
        all_datas->push_back(std::move(temp_data2));

        std::pair<std::vector<int64_t>, std::vector<int>> temp_data3;
        ParseData<int64_t>(fields[i++], &temp_data3); 
        all_datas->push_back(std::move(temp_data3));

        std::pair<std::vector<float>, std::vector<int>> temp_data4;
        ParseData<float>(fields[i++], &temp_data4); 
        data->push_back(std::move(temp_data4));
    }
    return true;
}

// Bert inference demo
// Options:
//     --model_dir: bert model file directory
//     --data: data path
//     --repeat: repeat num
//     --use_gpu: use gpu
int main(int argc, char *argv[]) {
   // google::InitGoogleLogging(*argv);
   // gflags::ParseCommandLineFlags(&argc, &argv, true);

  std::mutex g_mutex;
    if (FLAGS_model_dir.empty()) {
        LOG(ERROR) << "please set model dir";
        return -1;
    }
  std::atomic<int> cnt(0);
  std::vector<std::thread> threads;
  const int num_threads = 2;
  for (int tid = 0; tid < num_threads; ++tid) {
    threads.emplace_back([&, tid]() {

    g_mutex.lock();
    std::cout << "mutex locked."<<std::endl;
    paddle::AnalysisConfig config;
    config.SetModel(FLAGS_model_dir);

    config.EnableUseGpu(700, 0);
    config.SwitchSpecifyInputNames(true);
    config.EnableMemoryOptim();
    config.SwitchUseFeedFetchOps(false);
    config.BindGpuStreamToThread();
    config.SwitchIrDebug(true);

    int batch = 40;
    int seq_len = 128;
    int head_number = 12;

    int min_seq_len = 5;
    int max_seq_len = 128;
    int opt_seq_len = 60;
    
    std::vector<int> min_in_shape = {batch, min_seq_len, 1};
    std::vector<int> max_in_shape = {batch, max_seq_len, 1};
    std::vector<int> opt_in_shape = {batch, opt_seq_len, 1};

    std::vector<int> four_shape = {batch, 12, seq_len, seq_len};
    std::string input1_name = "read_file_0.tmp_0";
    std::string input2_name = "read_file_0.tmp_1";
    std::string input3_name = "read_file_0.tmp_2";
    std::string input4_name = "stack_0.tmp_0";

    std::map<std::string, std::vector<int>> min_input_shape 
               = {{input1_name, min_in_shape}, 
                 {input2_name, min_in_shape}, 
                 {input3_name, min_in_shape}, 
                 {input4_name, {batch, head_number, min_seq_len, min_seq_len}}};
    std::map<std::string, std::vector<int>> max_input_shape 
               = {{input1_name, max_in_shape}, 
                  {input2_name, max_in_shape},
                   {input3_name, max_in_shape}, 
                 {input4_name, {batch, head_number, max_seq_len, max_seq_len}}};
    std::map<std::string, std::vector<int>> opt_input_shape 
               = {{input1_name, opt_in_shape},
                  {input2_name, opt_in_shape},
                  {input3_name, opt_in_shape},
                   {input4_name, {batch, head_number, opt_seq_len, opt_seq_len}}};

    config.EnableTensorRtEngine(1 << 30, batch, 5, /*min_subgraph_size*/
     paddle::AnalysisConfig::Precision::kHalf, false, true);
     // paddle::AnalysisConfig::Precision::kFloat32, false, true);
    config.SetTRTDynamicShapeInfo(
       min_input_shape,
       max_input_shape, 
       opt_input_shape);
       //opt_input_shape, true);   // ban trt plugin fp16

      auto predictor = CreatePaddlePredictor(config);
      g_mutex.unlock();
    std::cout << "mutex un-locked."<<std::endl;
      cnt ++;
      while(cnt != num_threads) {}


      std::vector<std::pair<std::vector<int64_t>, std::vector<int>>> inputs_idx; 
      std::vector<std::pair<std::vector<float>, std::vector<int>>> inputs_data; 
      if (!LoadInputData(&inputs_idx, &inputs_data)) {
          LOG(ERROR) << "load input data error!";
          return -1;
      }

      float total_time{0};
      // auto predict_timer = []()
      int count{0};
      int index = 0;
      int num_samples = inputs_idx.size() / 3;
      LOG(INFO) << inputs_idx.size() / 3;
      LOG(INFO) << inputs_data.size();
      auto print_shape = [](std::string name, std::vector<int> shape) {
        LOG(INFO) << name;
        for (auto i : shape) LOG(INFO) << i;  
      };

      auto input_names = predictor->GetInputNames();
      for(auto name : input_names) LOG(INFO) << name;

      auto run_once = [&](int index, std::vector<float> * out_data) {
        auto input_t = predictor->GetInputTensor(input_names[0]);
        input_t->Reshape(inputs_idx[index * 3].second);
        input_t->copy_from_cpu(inputs_idx[index*3].first.data()); 
        // print_shape(input_names[0], inputs_idx[index].second);
   
        auto input_t1 = predictor->GetInputTensor(input_names[1]);
        input_t1->Reshape(inputs_idx[index * 3 + 1].second);
        input_t1->copy_from_cpu(inputs_idx[index*3 + 1].first.data()); 
        // kprint_shape(input_names[1], inputs_idx[index + 1].second);
   
        auto input_t2 = predictor->GetInputTensor(input_names[2]);
        input_t2->Reshape(inputs_idx[index*3+2].second);
        input_t2->copy_from_cpu(inputs_idx[index*3+2].first.data()); 
        //print_shape(input_names[2], inputs_idx[index + 2].second);
   
        auto input_t3 = predictor->GetInputTensor(input_names[3]);
        input_t3->Reshape(inputs_data[index].second);
        input_t3->copy_from_cpu(inputs_data[index].first.data()); 
        //print_shape(input_names[3], inputs_idx[index].second);
        //
        predictor->ZeroCopyRun();
        
        auto output_names = predictor->GetOutputNames();
        auto output_t = predictor->GetOutputTensor(output_names[0]);
        std::vector<int> output_shape = output_t->shape();
        int out_num = std::accumulate(output_shape.begin(), output_shape.end(), 1, std::multiplies<int>());
        out_data->resize(out_num);
        LOG(INFO) << "============== one loop ===========";
      //  output_t->copy_to_cpu(out_data->data());
      };

      auto time1 = time(); 
      for (int i = 0; i < 1000; i++) { // warm up
          std::vector<float> tmp_out;
          run_once(0, &tmp_out);
      }
      auto time2 = time(); 
      std::cout << " predict cost: " << time_diff(time1, time2) / 1000.0 << "ms" << std::endl;

      std::ofstream ofi("fp32_result" + std::to_string(tid) +".txt");
      for (int i = 0; i < FLAGS_repeat; i++) {
          for (;index < num_samples; index++) {
              auto start = time();
              std::vector<float> out_data;
              run_once(index, &out_data);
              auto end = time();
              count += 1;
              total_time += time_diff(time1, time2);
              LOG(INFO) << index << " "  << out_data.size();
              for (auto ele : out_data) ofi << ele << std::endl;
            //  std::cout << std::endl;
          }
      }
      ofi.close();

      auto per_sample_ms =
          static_cast<float>(total_time) / num_samples;
      LOG(INFO) << "Run " << num_samples
          << " samples, average latency: " << per_sample_ms
          << "ms per sample.";
      LOG(INFO) << count;
    });
  }
  for (int i = 0; i < num_threads; ++i) {
    threads[i].join();
  }

    return 0;
}
