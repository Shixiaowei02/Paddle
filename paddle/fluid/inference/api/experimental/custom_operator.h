/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <string>

namespace paddle_infer {
namespace experimental {
///
/// \brief A function to help load custom operator library.
///
/// Usage:
///
/// \code{.cpp}
/// const std::string dso_name{"custom_relu_module_setup_pd_.so"};
/// paddle_infer::experimental::LoadCustomOperatorLib(dso_name);
/// \endcode
///
/// FOR EXTENSION DEVELOPER:
/// For how to create a custom operator, please refer to our official website.
/// https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/index_cn.html
////
void LoadCustomOperatorLib(const std::string& dso_name);
}
}
