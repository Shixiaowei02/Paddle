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

#pragma once

#include <algorithm>
#include <string>
#include <vector>

#include "paddle/fluid/inference/tensorrt/engine.h"
#include "paddle/fluid/inference/tensorrt/plugin/trt_plugin.h"

namespace paddle {
namespace inference {
namespace tensorrt {
namespace plugin {

#if IS_TRT_VERSION_GE(6000)

class EmbEltwiseLayernormPluginDynamicImplBase {
 public:
  EmbEltwiseLayernormPluginDynamicImplBase() {}
  virtual ~EmbEltwiseLayernormPluginDynamicImplBase() {}

  virtual int initialize() = 0;
  virtual void terminate() = 0;
  virtual int enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
                      const nvinfer1::PluginTensorDesc* outputDesc,
                      const void* const* inputs, void* const* outputs,
                      void* workspace, cudaStream_t stream) = 0;
};
template<typename T>
class EmbEltwiseLayernormPluginDynamicImpl
    : public EmbEltwiseLayernormPluginDynamicImplBase {
 public:
  explicit EmbEltwiseLayernormPluginDynamicImpl(std::vector<float*> input_embs,
                                                float* bias, float* scale,
                                                std::vector<int> emb_sizes,
                                                int bias_size, int scale_size,
                                                int hidden_size, float eps)
      : embs_(input_embs),
        bias_(bias),
        scale_(scale),
        emb_sizes_(emb_sizes),
        bias_size_(bias_size),
        scale_size_(scale_size),
        hidden_size_(hidden_size),
        eps_(eps) {}

  ~EmbEltwiseLayernormPluginDynamicImpl();

  int initialize();
  void terminate();
  int enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
              const nvinfer1::PluginTensorDesc* outputDesc,
              const void* const* inputs, void* const* outputs, void* workspace,
              cudaStream_t stream);

 private:
  std::vector<float*> embs_;
  float* bias_{nullptr};
  float* scale_{nullptr};
  // data on devices
  float* bias_gpu_{nullptr};
  float* scale_gpu_{nullptr};
  std::vector<T*> embs_gpu_;

  std::vector<int> emb_sizes_;
  int bias_size_;
  int scale_size_;
  int hidden_size_;
  float eps_;

  framework::Tensor in_ptr_tensor_, emb_ptr_tensor_;
  int device_id_{0};
  uintptr_t old_input_ptr_{0};
};

class EmbEltwiseLayernormPluginDynamic : public DynamicPluginTensorRT {
 public:
  explicit EmbEltwiseLayernormPluginDynamic(std::vector<float*> input_embs,
                                            float* bias, float* scale,
                                            std::vector<int> emb_sizes,
                                            int bias_size, int scale_size,
                                            int hidden_size, float eps,
                                            bool with_fp16)
      : embs_(input_embs),
        bias_(bias),
        scale_(scale),
        emb_sizes_(emb_sizes),
        bias_size_(bias_size),
        scale_size_(scale_size),
        hidden_size_(hidden_size),
        eps_(eps),
        with_fp16_(with_fp16),
        own_host_buff_(false) {
if (with_fp16) {
#ifdef SUPPORTS_CUDA_FP16
      impl_ = new EmbEltwiseLayernormPluginDynamicImpl<half>(
          embs_, bias_, scale_, emb_sizes_, bias_size_, scale_size_,
          hidden_size_, eps_);
#else
      PADDLE_THROW(platform::errors::Fatal(
          "Unsupported data type, current GPU doesn't support half."));
#endif  // SUPPORTS_CUDA_FP16
    } else {
      impl_ = new EmbEltwiseLayernormPluginDynamicImpl<float>(
          embs_, bias_, scale_, emb_sizes_, bias_size_, scale_size_,
          hidden_size_, eps_);
    }
  }

  EmbEltwiseLayernormPluginDynamic(void const* serialData,
                                   size_t serialLength) : own_host_buff_(true) {
    DeserializeValue(&serialData, &serialLength, &emb_sizes_);

    embs_.resize(emb_sizes_.size());
    for (size_t i = 0; i < emb_sizes_.size(); i++) {
      auto size = emb_sizes_[i];
      auto ptr = new float[size];
      memcpy(ptr, serialData, sizeof(float) * size);
      embs_[i] = ptr;

      reinterpret_cast<char const*&>(serialData) +=
          emb_sizes_[i] * sizeof(float);
      serialLength -= emb_sizes_[i] * sizeof(float);
    }
    DeserializeValue(&serialData, &serialLength, &bias_size_);
    DeserializeValue(&serialData, &serialLength, &scale_size_);

    if (bias_size_) {
      bias_ = new float[bias_size_];
      memcpy(bias_, serialData, sizeof(float) * bias_size_);
    }
    reinterpret_cast<char const*&>(serialData) += bias_size_ * sizeof(float);
    serialLength -= bias_size_ * sizeof(float);

    if (scale_size_) {
      scale_ = new float[scale_size_];
      memcpy(scale_, serialData, sizeof(float) * scale_size_);
    }
    reinterpret_cast<char const*&>(serialData) += scale_size_ * sizeof(float);
    serialLength -= scale_size_ * sizeof(float);

    DeserializeValue(&serialData, &serialLength, &hidden_size_);
    DeserializeValue(&serialData, &serialLength, &eps_);
    DeserializeValue(&serialData, &serialLength, &with_fp16_);

    if (with_fp16_) {
#ifdef SUPPORTS_CUDA_FP16
      impl_ = new EmbEltwiseLayernormPluginDynamicImpl<half>(
          embs_, bias_, scale_, emb_sizes_, bias_size_, scale_size_,
          hidden_size_, eps_);
#else
      PADDLE_THROW(platform::errors::Fatal(
          "Unsupported data type, current GPU doesn't support half."));
#endif  // SUPPORTS_CUDA_FP16
    } else {
      impl_ = new EmbEltwiseLayernormPluginDynamicImpl<float>(
          embs_, bias_, scale_, emb_sizes_, bias_size_, scale_size_,
          hidden_size_, eps_);
    }
  }

  nvinfer1::IPluginV2DynamicExt* clone() const override {
    return new EmbEltwiseLayernormPluginDynamic(
        embs_, bias_, scale_, emb_sizes_, bias_size_, scale_size_, hidden_size_,
        eps_, with_fp16_);
  }

  const char* getPluginType() const override {
    return "fused_embedding_eltwise_layernorm_plugin";
  }
  int getNbOutputs() const override { return 1; }
  int initialize() override;
  void terminate() override;

  size_t getSerializationSize() const override {
    int sum_num = 0;
    sum_num += SerializedSize(emb_sizes_);

    for (size_t i = 0; i < emb_sizes_.size(); i++) {
      sum_num += emb_sizes_[i] * sizeof(float);
    }

    sum_num += SerializedSize(bias_size_);
    sum_num += SerializedSize(scale_size_);

    sum_num += (bias_size_ + scale_size_) * sizeof(float);
    sum_num += SerializedSize(hidden_size_);
    sum_num += SerializedSize(eps_);
    sum_num += SerializedSize(with_fp16_);

    return sum_num;
  }

  void serialize(void* buffer) const override {
    // SerializeValue(&buffer, with_fp16_);
    SerializeValue(&buffer, emb_sizes_);
    for (size_t i = 0; i < emb_sizes_.size(); i++) {
      auto size = emb_sizes_[i];
      for (int j = 0; j < size; ++j) {
        SerializeValue(&buffer, embs_[i][j]);
      }
    }
    SerializeValue(&buffer, bias_size_);
    SerializeValue(&buffer, scale_size_);
    for (int i = 0; i < bias_size_; ++i) {
      SerializeValue(&buffer, bias_[i]);
    }

    for (int i = 0; i < scale_size_; ++i) {
      SerializeValue(&buffer, scale_[i]);
    }

    SerializeValue(&buffer, hidden_size_);
    SerializeValue(&buffer, eps_);
    SerializeValue(&buffer, with_fp16_);
  }

  nvinfer1::DimsExprs getOutputDimensions(
      int output_index, const nvinfer1::DimsExprs* inputs, int nb_inputs,
      nvinfer1::IExprBuilder& expr_builder) override;

  bool supportsFormatCombination(int pos,
                                 const nvinfer1::PluginTensorDesc* inOut,
                                 int nbInputs, int nbOutputs) override;

  void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in,
                       int nbInputs,
                       const nvinfer1::DynamicPluginTensorDesc* out,
                       int nbOutputs) override {}

  size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs,
                          int nbInputs,
                          const nvinfer1::PluginTensorDesc* outputs,
                          int nbOutputs) const override {
    return 0;
  }

  int enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
              const nvinfer1::PluginTensorDesc* outputDesc,
              const void* const* inputs, void* const* outputs, void* workspace,
              cudaStream_t stream) override;
  nvinfer1::DataType getOutputDataType(int index,
                                       const nvinfer1::DataType* inputTypes,
                                       int nbInputs) const override;

  void destroy() override {
    if (own_host_buff_) {
      for (auto ptr : embs_) {
        delete[] ptr;
      }
      delete[] bias_;
      delete[] scale_;
    }

    delete impl_;
    delete this;
  } 

 private:
  std::vector<float*> embs_;
  float* bias_;
  float* scale_;

  std::vector<int> emb_sizes_;
  int bias_size_;
  int scale_size_;
  int hidden_size_;
  float eps_;
  bool with_fp16_;
  bool own_host_buff_{false};
  EmbEltwiseLayernormPluginDynamicImplBase* impl_{nullptr};
};

class EmbEltwiseLayernormPluginV2Creator : public nvinfer1::IPluginCreator {
 public:
  EmbEltwiseLayernormPluginV2Creator() {}
  const char* getPluginName() const override {
    return "fused_embedding_eltwise_layernorm_plugin";
  }

  const char* getPluginVersion() const override { return "1"; }

  const nvinfer1::PluginFieldCollection* getFieldNames() override {
    return &mFieldCollection;
  }

  nvinfer1::IPluginV2* createPlugin(
      const char* name, const nvinfer1::PluginFieldCollection* fc) override {
    return nullptr;
  }

  nvinfer1::IPluginV2* deserializePlugin(const char* name,
                                         const void* serialData,
                                         size_t serialLength) override {
    return new EmbEltwiseLayernormPluginDynamic(serialData, serialLength);
  }

  void setPluginNamespace(const char* libNamespace) override {
    mNamespace = libNamespace;
  }

  const char* getPluginNamespace() const override { return mNamespace.c_str(); }

 private:
  std::string mNamespace;
  std::string mPluginName;
  nvinfer1::PluginFieldCollection mFieldCollection;
  std::vector<nvinfer1::PluginField> mPluginAttributes;
};

REGISTER_TENSORRT_PLUGIN(EmbEltwiseLayernormPluginV2Creator);

#endif
}  // namespace plugin
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
