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

#include "pd_infer_type.h"

struct PD_INFER_DECL Config {

  Config() = default;

  explicit Config(const Config& other);

  explicit Config(const std::string& model_dir);

  explicit Config(const std::string& prog_file, const std::string& params_file);

  void SetModel(const std::string& model_dir);

  void SetModel(const std::string& prog_file,
                const std::string& params_file);

  void SetOptimCacheDir(const std::string& opt_cache_dir);

  const std::string& model_dir() const;

  const std::string& prog_file() const;

  const std::string& params_file() const;

  int memory_pool_init_size_mb() const;

  bool model_from_memory() const;

  void PartiallyRelease();

  struct CPU;
  struct GPU;
  struct XPU;
  struct Debug;
  struct LiteCore;
  struct MkldnnQuantizer;
};

struct Config::CPU {

  void EnableMKLDNN();

  void SetMkldnnCacheCapacity(int capacity);

  bool mkldnn_enabled() const;

  void SetCpuMathLibraryNumThreads(int cpu_math_library_num_threads);

  int cpu_math_library_num_threads() const;

  void SetMKLDNNOp(const std::unordered_set<std::string>& op_list);

  void EnableMkldnnQuantizer();

  void EnableMkldnnBfloat16();

  bool mkldnn_bfloat16_enabled() const;

  void SetBfloat16Op(const std::unordered_set<std::string>& op_list);

  bool mkldnn_quantizer_enabled() const;

  MkldnnQuantizerConfig* mkldnn_quantizer_config();
};

struct Config::GPU {

  void EnableUseGpu(uint64_t memory_pool_init_size_mb, int device_id = 0);

  void DisableGpu();

  bool use_gpu() const;

  int gpu_device_id() const;

  void EnableCUDNN();

  bool cudnn_enabled() const;

  void EnableTensorRtEngine(int workspace_size = 1 << 20,
                            int max_batch_size = 1, int min_subgraph_size = 3,
                            const Precision& precision = Precision::kFloat32,
                            bool use_static = false,
                            bool use_calib_mode = true);

  bool tensorrt_engine_enabled() const;

  void SetTRTDynamicShapeInfo(
      const std::map<std::string, std::vector<int>>& min_input_shape,
      const std::map<std::string, std::vector<int>>& max_input_shape,
      const std::map<std::string, std::vector<int>>& optim_input_shape,
      bool disable_trt_plugin_fp16 = false);

  void Exp_DisableTensorRtOPs(const std::vector<std::string>& ops);

  void EnableTensorRtOSS();

  bool tensorrt_oss_enabled() const;

  void EnableTensorRtDLA();

  bool tensorrt_dla_enabled() const;

  void EnableGpuMultiStream();

  bool thread_local_stream_enabled() const;

  float fraction_of_gpu_memory_for_pool();
};

struct Config::XPU {

  void EnableXpu(int l3_workspace_size = 0xfffc00);

  bool use_xpu() const;

  int xpu_device_id() const;
};

struct Config::Debug {

  PassStrategy* pass_builder();

  void DisableFCPadding();

  bool use_fc_padding() const;

  void SwitchIrOptim(int x = true);

  bool ir_optim() const;

  void SwitchUseFeedFetchOps(int x = true);

  bool feed_fetch_ops_enabled() const;

  void SwitchSpecifyInputNames(bool x = true);

  bool input_name_specified() const;

  void SwitchIrDebug(int x = true);

  void EnableMemoryOptim();

  bool enable_memory_optim() const;

  void EnableProfile();

  bool profile_enabled() const;

  void DisableGlogInfo();

  bool glog_info_disabled() const;
};

struct Config::LiteCore {

  void EnableLiteEngine(
      const Precision& precision_mode = Precision::kFloat32,
      bool zero_copy = false,
      const std::vector<std::string>& passes_filter = {},
      const std::vector<std::string>& ops_filter = {});

  bool lite_engine_enabled() const;
};

struct Config::MkldnnQuantizer {

  enum class ScaleAlgo {
    NONE,      ///< Do not compute scale
    MAX,       ///< Find scale based on the max absolute value
    MAX_CH,    ///< Find scale based on the max absolute value per output channel
    MAX_CH_T,  ///< Find scale based on the max absolute value per output channel
               ///< of a transposed tensor
    KL,        ///< Find scale based on KL Divergence
  };

  void SetScaleAlgo(const std::string& op_type_name, const std::string& conn_name,
                    const ScaleAlgo& algo);

  const ScaleAlgo& scale_algo(const std::string& op_type_name,
                       const std::string& conn_name);

  void SetWarmupData(std::shared_ptr<std::vector<PaddleTensor>> data);

  std::shared_ptr<std::vector<PaddleTensor>> warmup_data() const;

  void SetWarmupBatchSize(int batch_size);

  int warmup_batch_size() const;

  void SetEnabledOpTypes(const std::unordered_set<std::string>& op_list);

  const std::unordered_set<std::string>& enabled_op_types() const;

  void SetExcludedOpIds(const std::unordered_set<int>& op_ids_list);

  void SetDefaultScaleAlgo(const ScaleAlgo& algo);

  ScaleAlgo default_scale_algo() const;
};
