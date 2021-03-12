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

if(APPLE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-error=pessimizing-move")
endif(APPLE)

if(WITH_MKLDNN)
  set(mkldnn_quantizer_src ${CMAKE_CURRENT_SOURCE_DIR}/mkldnn_quantizer.cc)
  set(mkldnn_quantizer_cfg mkldnn_quantizer_config)
  cc_library(${mkldnn_quantizer_cfg} SRCS mkldnn_quantizer_config.cc DEPS lod_tensor paddle_pass_builder)
  set(mkldnn_quantizer_src_file ${mkldnn_quantizer_src} PARENT_SCOPE)
  set(mkldnn_quantizer_cfg ${mkldnn_quantizer_cfg} PARENT_SCOPE)
endif()

include(details)
