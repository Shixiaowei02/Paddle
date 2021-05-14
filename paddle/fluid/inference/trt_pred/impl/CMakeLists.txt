include_directories("../")

cc_library(pd_infer_trt_pred SHARED SRCS tensorrt_predictor.cc DEPS)
