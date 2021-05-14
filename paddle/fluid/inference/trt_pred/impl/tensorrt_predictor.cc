#include "paddle/fluid/inference/trt_pred/tensorrt_predictor.h"
#include "paddle/fluid/inference/trt_pred/tensorrt_config.h"
#include <glog/logging.h>
#include <algorithm>
#include <fstream>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>
#include "paddle/fluid/framework/feed_fetch_method.h"
#include "paddle/fluid/framework/feed_fetch_type.h"
#include "paddle/fluid/framework/ir/fuse_pass_base.h"
#include "paddle/fluid/framework/ir/pass.h"
#include "paddle/fluid/framework/naive_executor.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/var_type_traits.h"
#include "paddle/fluid/framework/version.h"
#include "paddle/fluid/inference/analysis/helper.h"
#include "paddle/fluid/inference/analysis/passes/memory_optimize_pass.h"
#include "paddle/fluid/inference/api/helper.h"
#include "paddle/fluid/inference/api/paddle_inference_pass.h"
#include "paddle/fluid/inference/utils/singleton.h"
#include "paddle/fluid/memory/memcpy.h"
#include "paddle/fluid/platform/cpu_helper.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/gpu_info.h"
#include "paddle/fluid/platform/place.h"
#include "paddle/fluid/inference/analysis/analyzer.h"
#include "paddle/fluid/platform/init.h"
#include "paddle/fluid/platform/enforce.h"

#if PADDLE_WITH_TENSORRT
#include "paddle/fluid/inference/tensorrt/convert/op_converter.h"
#include "paddle/fluid/inference/tensorrt/trt_int8_calibrator.h"
#endif
#include "paddle/fluid/framework/naive_executor.h"
#include "paddle/fluid/inference/api/paddle_api.h"
#include "paddle/fluid/inference/analysis/argument.h"

#include <map>

using paddle::framework::Scope;
using paddle::framework::NaiveExecutor;
using paddle::framework::ProgramDesc;
using paddle::inference::analysis::Argument;
using paddle::ZeroCopyTensor;
using paddle::inference::analysis::Analyzer;
using paddle::framework::FeedList;
using paddle::framework::FetchList;


namespace paddle_infer {
namespace tensorrt {

struct Predictor::Impl {
Impl(const Config& config) : config_{config}{

}
bool Init(const std::shared_ptr<Scope> &parent_scope,
const std::shared_ptr<ProgramDesc> &program) {
  predictor_id_ = paddle::inference::GetUniqueId();
    if (!PrepareScope(parent_scope)) {
      return false;
    }
    if (!CreateExecutor()) {
      return false;
    }
    if (!PrepareProgram(program)) {
      return false;
    }
    // Prepare executor, create local variables.
    if (!PrepareExecutor()) {
      return true;
    }
    // Get the feed_target_names and fetch_target_names
    PrepareFeedFetch();
    return true;
  }

void PrepareFeedFetch() {
  PADDLE_ENFORCE_NOT_NULL(sub_scope_,
                          paddle::platform::errors::InvalidArgument(
                              "The sub_scope should not be nullptr."));
  CreateFeedFetchVar(sub_scope_);
  for (auto *op : inference_program_->Block(0).AllOps()) {
    if (op->Type() == "feed") {
      int idx = BOOST_GET_CONST(int, op->GetAttr("col"));
      if (feeds_.size() <= static_cast<size_t>(idx)) {
        feeds_.resize(idx + 1);
      }
      feeds_[idx] = op;
      feed_names_[op->Output("Out")[0]] = idx;
      idx2feeds_[idx] = op->Output("Out")[0];
    } else if (op->Type() == "fetch") {
      int idx = BOOST_GET_CONST(int, op->GetAttr("col"));
      if (fetches_.size() <= static_cast<size_t>(idx)) {
        fetches_.resize(idx + 1);
      }
      fetches_[idx] = op;
      idx2fetches_[idx] = op->Input("X")[0];
    }
  }
}

void CreateFeedFetchVar(Scope *scope) {
  PADDLE_ENFORCE_NOT_NULL(scope, paddle::platform::errors::InvalidArgument(
                                     "The scope should not be nullptr."));
  auto *var = scope->Var("feed");
  var->GetMutable<FeedList>();
  var = scope->Var("fetch");
  var->GetMutable<FetchList>();
}

bool PrepareScope(
    const std::shared_ptr<Scope> &parent_scope) {
  if (parent_scope) {
    PADDLE_ENFORCE_NOT_NULL(
        parent_scope,
        paddle::platform::errors::PreconditionNotMet(
            "Both program and parent_scope should be set in Clone mode."));
    scope_ = parent_scope;
    status_is_cloned_ = true;
  } else {
    paddle::framework::InitDevices();
    scope_.reset(new paddle::framework::Scope());
    status_is_cloned_ = false;
  }
  sub_scope_ = &scope_->NewScope();
  return true;
}

bool LoadProgramDesc() {
  // Initialize the inference program
  std::string filename;
  if (!config_.model_dir().empty()) {
    filename = config_.model_dir() + "/__model__";
  } else if (!config_.prog_file().empty() && !config_.params_file().empty()) {
    // All parameters are saved in a single file.
    // The file names should be consistent with that used
    // in Python API `fluid.io.save_inference_model`.
    filename = config_.prog_file();
  } else {
    if (config_.model_dir().empty() && config_.prog_file().empty()) {
      LOG(ERROR)
          << "Either model_dir or (prog_file, param_file) should be set.";
      return false;
    }
    return false;
  }

  // Create ProgramDesc
  paddle::framework::proto::ProgramDesc proto;
  if (!config_.model_from_memory()) {
    std::string pb_content;
    // Read binary
    std::ifstream fin(filename, std::ios::in | std::ios::binary);
    PADDLE_ENFORCE_EQ(
        static_cast<bool>(fin.is_open()), true,
        paddle::platform::errors::NotFound(
            "Cannot open file %s, please confirm whether the file is normal.",
            filename));
    fin.seekg(0, std::ios::end);
    pb_content.resize(fin.tellg());
    fin.seekg(0, std::ios::beg);
    fin.read(&(pb_content.at(0)), pb_content.size());
    fin.close();

    proto.ParseFromString(pb_content);
  } else {
    proto.ParseFromString(config_.prog_file());
  }
  inference_program_.reset(new ProgramDesc(proto));
  return true;
}

bool PrepareProgram(
    const std::shared_ptr<ProgramDesc> &program) {
  if (!program) {
    if (!LoadProgramDesc()) return false;
    // If not cloned, the parameters should be loaded.
    // If config_.ir_optim() is True, parameters is loaded in
    // OptimizeInferenceProgram(), but other persistable variables
    // (like RAW type var) are not created in scope.
    // If config_.ir_optim() is False, parameters is loaded in LoadParameters(),
    // still need to create other persistable variables.
    // So in both case, create persistable variables at first.
    executor_->CreateVariables(*inference_program_, 0, true, sub_scope_);

    // if enable_ir_optim_ is false,
    // the analysis pass(op fuse, graph analysis, trt subgraph, mkldnn etc) will
    // not be executed.
    OptimizeInferenceProgram();
  } else {
    // If the program is passed from external, no need to optimize it, this
    // logic is used in the clone scenario.
    inference_program_ = program;
  }

  executor_->CreateVariables(*inference_program_, 0, false, sub_scope_);
  return true;
}

bool CreateExecutor() {
  if (config_.use_gpu()) {
    PADDLE_ENFORCE_EQ(config_.use_xpu(), false,
                      paddle::platform::errors::InvalidArgument(
                          "Only one choice can be made between CPU and XPU."));
    place_ = paddle::platform::CUDAPlace(config_.gpu_device_id());
  } else {
    LOG(FATAL);
  }
  executor_.reset(new paddle::framework::NaiveExecutor(place_));
  return true;
}

bool PrepareExecutor() {
  executor_->Prepare(sub_scope_, *inference_program_, 0,
                     config_.use_feed_fetch_ops_);

  PADDLE_ENFORCE_NOT_NULL(sub_scope_,
                          paddle::platform::errors::PreconditionNotMet(
                              "The sub_scope should not be nullptr."));

  return true;
}

void PrepareArgument() {
  argument_.SetUseGPU(config_.use_gpu());
  argument_.SetUseFcPadding(config_.use_fc_padding());
  argument_.SetGPUDeviceId(config_.gpu_device_id());
  argument_.SetEnableAnalysisOptim(config_.ir_optim());
  argument_.SetEnableMemoryOptim(config_.enable_memory_optim());
  argument_.SetModelFromMemory(config_.model_from_memory());
  // Analyze inference_program
  argument_.SetPredictorID(predictor_id_);
  argument_.SetOptimCacheDir(config_.opt_cache_dir());
  if (!config_.model_dir().empty()) {
    argument_.SetModelDir(config_.model_dir());
  } else {
    PADDLE_ENFORCE_EQ(config_.params_file().empty(), false,
                      paddle::platform::errors::PreconditionNotMet(
                          "Either model_dir or param_file should be set."));
    PADDLE_ENFORCE_EQ(config_.prog_file().empty(), false,
                      paddle::platform::errors::PreconditionNotMet(
                          "Either model_dir or prog_file should be set."));
    std::string dir = paddle::inference::analysis::GetDirRoot(config_.prog_file());

    argument_.SetModelProgramPath(config_.prog_file());
    argument_.SetModelParamsPath(config_.params_file());
  }

  if (config_.use_gpu() && config_.tensorrt_engine_enabled()) {
    LOG(INFO) << "TensorRT subgraph engine is enabled";
    argument_.SetUseTensorRT(true);
    argument_.SetTensorRtWorkspaceSize(config_.tensorrt_workspace_size_);
    argument_.SetTensorRtMaxBatchSize(config_.tensorrt_max_batchsize_);
    argument_.SetTensorRtMinSubgraphSize(config_.tensorrt_min_subgraph_size_);
    argument_.SetTensorRtDisabledOPs(config_.trt_disabled_ops_);
    argument_.SetTensorRtUseDLA(config_.trt_use_dla_);
    argument_.SetTensorRtDLACore(config_.trt_dla_core_);
    argument_.SetTensorRtPrecisionMode(config_.tensorrt_precision_mode_);
    argument_.SetTensorRtUseStaticEngine(config_.trt_use_static_engine_);
    argument_.SetTensorRtUseCalibMode(config_.trt_use_calib_mode_);
    argument_.SetTensorRtUseOSS(config_.trt_use_oss_);
    argument_.SetMinInputShape(config_.min_input_shape_);
    argument_.SetMaxInputShape(config_.max_input_shape_);
    argument_.SetOptimInputShape(config_.optim_input_shape_);
    argument_.SetCloseTrtPluginFp16(config_.disable_trt_plugin_fp16_);
  }

  if (config_.use_mkldnn_) {
    LOG(INFO) << "MKLDNN is enabled";
    argument_.SetMKLDNNEnabledOpTypes(config_.mkldnn_enabled_op_types_);
  }

  auto passes = config_.pass_builder()->AllPasses();
  if (!config_.ir_optim()) {
    passes.clear();
    LOG(INFO) << "ir_optim is turned off, no IR pass will be executed";
  }
  argument_.SetDisableLogs(config_.glog_info_disabled());
  argument_.SetIrAnalysisPasses(passes);
  argument_.SetAnalysisPasses(config_.pass_builder()->AnalysisPasses());
  argument_.SetScopeNotOwned(scope_.get());
}

void OptimizeInferenceProgram() {
  PrepareArgument();
  Analyzer().Run(&argument_);

  PADDLE_ENFORCE_EQ(
      argument_.scope_valid(), true,
      paddle::platform::errors::InvalidArgument("The argument scope should be valid."));
  VLOG(5) << "to prepare executor";
  ARGUMENT_CHECK_FIELD((&argument_), ir_analyzed_program);
  inference_program_.reset(
      new paddle::framework::ProgramDesc(argument_.ir_analyzed_program()));
  // The config and argument take a lot of storage,
  // when the predictor settings are complete, we release these stores.
  argument_.PartiallyRelease();
  config_.PartiallyRelease();
  LOG(INFO) << "======= optimize end =======";
}

  Config config_;
  Argument argument_;
  std::unique_ptr<NaiveExecutor> executor_;
  paddle::platform::Place place_;
  std::shared_ptr<Scope> scope_;
  Scope *sub_scope_{nullptr};
  std::shared_ptr<ProgramDesc> inference_program_;

  std::vector<paddle::framework::OpDesc *> feeds_;
  std::map<std::string, size_t> feed_names_;
  // Sorted according to the idx.
  std::map<size_t, std::string> idx2feeds_;
  std::vector<paddle::framework::OpDesc *> fetches_;
  std::map<size_t, std::string> idx2fetches_;

  // Memory buffer for feed inputs. The temporary LoDTensor will cause serious
  // concurrency problems, wrong results and memory leak, so cache them.
  std::vector<paddle::framework::LoDTensor> feed_tensors_;
  // A mutex help to make Clone thread safe.
  std::mutex clone_mutex_;

  // For memory optimization.
  const size_t max_shape_collect_count_{1000};
  int need_collect_var_shapes_{-1};  // -1 for default, 0 for false, 1 for true.
  std::vector<std::map<std::string, std::vector<int>>> batch_var_shapes_;
  int predictor_id_;
  bool status_is_cloned_{false};
};

Predictor::Predictor(const Config& config) : impl_{new Impl{config}} {
  impl_->predictor_id_ = paddle::inference::GetUniqueId();
}

std::vector<std::string> Predictor::GetInputNames() {
  std::vector<std::string> input_names;
  for (auto &item : impl_->idx2feeds_) {
    input_names.push_back(item.second);
  }
  return input_names;
}

std::unique_ptr<Tensor> Predictor::GetInputHandle(const std::string& name) {
/*
  PADDLE_ENFORCE_NOT_NULL(
      executor_->scope()->FindVar(name),
      paddle::platform::errors::PreconditionNotMet(
          "The variable named %s is not found in the scope of the exector.",
          name));
  std::unique_ptr<ZeroCopyTensor> res(
      new ZeroCopyTensor(static_cast<void *>(impl_->executor_->scope())));
  res->input_or_output_ = true;
  res->SetName(name);
  if (paddle::platform::is_gpu_place(place_)) {
    auto gpu_place = BOOST_GET_CONST(paddle::platform::CUDAPlace, place_);
    res->SetPlace(PaddlePlace::kGPU, gpu_place.GetDeviceId());
  } else {
    LOG(FATAL);
  }
  return res;
*/
return {};
}

bool Predictor::Run() {
  impl_->executor_->Run();
  return true;
}

std::vector<std::string> Predictor::GetOutputNames() {
  std::vector<std::string> output_names;
  for (auto &item : impl_->idx2fetches_) {
    output_names.push_back(item.second);
  }
  return output_names;
}

std::unique_ptr<Tensor> Predictor::GetOutputHandle(const std::string& name) {
/*
  PADDLE_ENFORCE_NOT_NULL(
      executor_->scope()->FindVar(name),
      platform::errors::PreconditionNotMet(
          "The variable named %s is not found in the scope of the exector.",
          name));
  std::unique_ptr<ZeroCopyTensor> res(
      new ZeroCopyTensor(static_cast<void *>(impl_->executor_->scope())));
  res->input_or_output_ = false;
  res->SetName(name);
  if (paddle::platform::is_gpu_place(place_)) {
    auto gpu_place = BOOST_GET_CONST(platform::CUDAPlace, place_);
    res->SetPlace(PaddlePlace::kGPU, gpu_place.GetDeviceId());
  } else {
    LOG(FATAL);
  }
  return res;
*/
return {};
}

std::unique_ptr<Predictor> Predictor::Clone() {
  LOG(FATAL);
  return {};
}

std::shared_ptr<Predictor> CreatePredictor(
const Config& config) {
  return std::make_shared<Predictor>(config);
}

} // namespace tensorrt
} // namespace paddle_infer

