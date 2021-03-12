# Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

cc_library(reset_tensor_array SRCS reset_tensor_array.cc DEPS lod_tensor scope)
cc_library(zero_copy_tensor SRCS zero_copy_tensor.cc DEPS scope lod_tensor enforce)
cc_library(zero_copy_tensor_dummy SRCS zero_copy_tensor_dummy.cc)

cc_test(zero_copy_tensor_test SRCS zero_copy_tensor_test.cc DEPS paddle_inference_api)

cc_library(analysis_config SRCS analysis_config.cc DEPS ${mkldnn_quantizer_cfg} lod_tensor paddle_pass_builder)
cc_library(paddle_pass_builder SRCS paddle_pass_builder.cc)

if(WITH_CRYPTO)
    cc_library(paddle_inference_api SRCS api.cc api_impl.cc helper.cc DEPS lod_tensor scope reset_tensor_array 
              analysis_config zero_copy_tensor trainer_desc_proto paddle_crypto)
else()
    cc_library(paddle_inference_api SRCS api.cc api_impl.cc helper.cc DEPS lod_tensor scope reset_tensor_array 
              analysis_config zero_copy_tensor trainer_desc_proto)
endif()

if(WIN32)
    target_link_libraries(paddle_inference_api gflags)
endif()

set(inference_deps ${analysis_deps} paddle_inference_api analysis naive_executor ${GLOB_PASS_LIB})

if(WITH_GPU AND TENSORRT_FOUND)
    set(inference_deps ${inference_deps} tensorrt_engine tensorrt_converter)
endif()

cc_library(analysis_predictor SRCS analysis_predictor.cc ${mkldnn_quantizer_src} DEPS ${inference_deps} 
          zero_copy_tensor ir_pass_manager op_compatible_info)

cc_test(test_paddle_inference_api SRCS api_tester.cc DEPS paddle_inference_api)

if(WITH_TESTING)
  if (NOT APPLE AND NOT WIN32)
    inference_base_test(test_api_impl SRCS api_impl_tester.cc DEPS paddle_inference_shared
      ARGS --word2vec_dirname=${WORD2VEC_MODEL_DIR} --book_dirname=${IMG_CLS_RESNET_INSTALL_DIR})
    set_tests_properties(test_api_impl PROPERTIES DEPENDS test_image_classification)
  elseif(WIN32)
    inference_base_test(test_api_impl SRCS api_impl_tester.cc DEPS ${inference_deps}
      ARGS --word2vec_dirname=${WORD2VEC_MODEL_DIR} --book_dirname=${IMG_CLS_RESNET_INSTALL_DIR})
    set_tests_properties(test_api_impl PROPERTIES DEPENDS test_image_classification)
  endif()

endif()

if (NOT APPLE AND NOT WIN32)
  cc_test(test_analysis_predictor SRCS analysis_predictor_tester.cc DEPS paddle_inference_shared
          ARGS --dirname=${WORD2VEC_MODEL_DIR})
elseif (WIN32)
  cc_test(test_analysis_predictor SRCS analysis_predictor_tester.cc DEPS analysis_predictor benchmark ${inference_deps}
          ARGS --dirname=${WORD2VEC_MODEL_DIR})
endif()
if(WITH_TESTING)
    if(NOT APPLE)
        set_tests_properties(test_api_impl PROPERTIES TIMEOUT 120)
    endif()
endif()
