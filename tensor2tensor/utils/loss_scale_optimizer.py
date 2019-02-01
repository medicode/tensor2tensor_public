# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""Loss scaling optimizer."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.eager import context
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_control_flow_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.training import optimizer
from tensorflow.python.training import distribution_strategy_context as distribute_ctx
from tensorflow.contrib.mixed_precision import LossScaleOptimizer

import tensorflow as tf


class DistributedLossScaleOptimizer(LossScaleOptimizer):
  # TODO(jamesqin): move mixed precision training explanation to __init__
  # docstring.
  """
  A wrapper for LossScaleOptimizer, since it does not support distributed
  See https://github.com/tensorflow/tensorflow/issues/25080
  """

  def apply_gradients(self, args, **kwargs):
    """Overriding parent apply_gradients to call distribution if necessary"""
    if distribute_ctx.has_distribution_strategy():
      return self.distributed_apply_gradients(*args, **kwargs)
    else:
      return super().apply_gradients(*args, **kwargs)
 
  def distributed_apply_gradients(self, grads_and_vars, global_step=None, name=None):
    grads = [g for (g, _) in grads_and_vars]

    is_finite_grad = []
    for g in grads:
      is_finite_grad.append(math_ops.reduce_all(gen_math_ops.is_finite(g)))
    is_overall_finite = math_ops.reduce_all(is_finite_grad)
    # Only update gradients when all grads are finite.
    def true_apply_gradients_fn():
      return self._opt.apply_gradients(grads_and_vars, global_step, name)
    
    #Need this instead of noop, unsure why
    def false_print_nan():
      return tf.print("One of the grads was not finite")

    #TODO: Fix cond
    print("Dist strat on")
    update_vars = true_apply_gradients_fn()

    # We need the cond below in our code
    # print("Dist strat off")
    # update_vars = control_flow_ops.cond(
    #   is_overall_finite, true_apply_gradients_fn, false_print_nan)

    # Potentially adjust gradient scale in case of finite gradients.
    return control_flow_ops.group(
        update_vars,
        self._loss_scale_manager.update_loss_scale(is_overall_finite))
