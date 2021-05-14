#pragma once 

#include <vector>
#include <string>
#include <memory>
#include "paddle_infer_declare.h"
#include "paddle_tensor.h"
#include "tensorrt_config.h"

namespace paddle_infer {
namespace tensorrt {

class PD_INFER_DECL Predictor {
 public:
  Predictor() = delete;

  ~Predictor() {}

  explicit Predictor(const Config& config);

  std::vector<std::string> GetInputNames();

  std::unique_ptr<Tensor> GetInputHandle(const std::string& name);

  bool Run();

  std::vector<std::string> GetOutputNames();

  std::unique_ptr<Tensor> GetOutputHandle(const std::string& name);

  std::unique_ptr<Predictor> Clone();

 private:
  struct Impl;
  std::shared_ptr<Impl> impl_;
  
};

PD_INFER_DECL std::shared_ptr<Predictor> CreatePredictor(
    const Config& config);

} // namespace tensorrt
} // namespace paddle_infer
