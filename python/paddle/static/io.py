# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function

import os
import paddle
import traceback
from paddle.fluid.framework import static_only

__all__ = [
    'save_inference_model',
    'load_inference_model',
]

def _verify_args(args, supported_args):
    caller = traceback.extract_stack(None, 2)[0][2]
    for arg in args:
        if arg not in supported_args:
            raise ValueError(
                "The argument '{}' is not supported.\nThe following arguments are supported by {}: {} ".format(arg, caller, supported_args))


@static_only
def save_inference_model(dirname, feed_vars, fetch_vars, executor, **configs):
    print('save_inference_model here..')
    supported_args = ('main_program', 'model_name', 'params_name')
    _verify_args(configs, supported_args)

def _get_abspath_compatible_api(dirname, model_name, params_name):
    model_suffix = '.pdmodel'
    params_suffix = '.pdiparams'
    abspath_model = ''
    abspath_params = ''
    model_prefix = os.path.join(dirname, model_name)
    params_prefix = os.path.join(dirname, params_name)
    usage_msg = 'Here is usage msg.'
    if os.path.exists(os.path.join(model_prefix, model_suffix)) and \
    os.path.exists(os.path.join(params_prefix, params_suffix)):
        # 1. Recommended usage: Recommended usage: the parameter represents
        # the file name prefix.
        # Abspath of model: `dirname` + `model_name` + `model_suffix`
        # Abspath of params: `dirname` + `params_name` + `params_suffix`
        abspath_model = os.path.join(model_prefix, model_suffix)
        abspath_params = os.path.join(params_prefix, params_suffix)
    elif os.path.exists(model_prefix) and os.path.exists(params_prefix):
        # 2. Compatible usage: The parameters provided by the user will
        # not be suffixed. At this time, the model format is still the
        # combined weight.
        abspath_model = model_prefix
        abspath_params = params_prefix
    else:
        # 3. Deprecated usage: Model format is discrete weight.
        if params_name is not None:
            raise ValueError("The name of params '{} is illegal.")


    return (None, None)

def _get_abspath(dirname):
    return (None, None)


@static_only
def load_inference_model(dirname, executor, **configs):
    print('save_inference_model here..')
    supported_args = ('model_name', 'params_name')
    _verify_args(configs, supported_args)
    _path_convert_compatible_api(dirname, model_name, params_name)
    _path_convert(dirname)

