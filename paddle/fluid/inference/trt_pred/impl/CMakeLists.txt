include_directories("../")

cc_library(pd_infer_trt_pred SHARED SRCS tensorrt_predictor.cc DEPS analysis_config paddle_inference_api analysis)
cc_binary(test_tensorrt_predictor SRCS test_tensorrt_predictor.cc DEPS pd_infer_trt_pred)
