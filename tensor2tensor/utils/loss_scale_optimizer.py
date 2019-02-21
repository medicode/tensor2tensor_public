"""Loss scaling optimizer with distribution strategy support added by Fathom."""

from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.training import distribution_strategy_context as distribute_ctx
from tensorflow.contrib.mixed_precision import LossScaleOptimizer


class DistributedLossScaleOptimizer(LossScaleOptimizer):
  # TODO(jamesqin): move mixed precision training explanation to __init__
  # docstring.
  """
  A wrapper for LossScaleOptimizer, since it does not support distributed
  See https://github.com/tensorflow/tensorflow/issues/25080
  """

  def apply_gradients(self, grads_and_vars, global_step=None, name=None):
    """
      Fathom: Overriding parent apply_gradients to call
        dist_apply_gradients if necessary
    """
    # if distribute_ctx.has_distribution_strategy():
    #   # Use Fathom built distribtued_apply_gradients
    #   print("Using our dist")
    return self.dist_apply_gradients(grads_and_vars, global_step, name)
    # else:
    #   print("Using super")
    #   return super().apply_gradients(grads_and_vars, global_step, name)

  def dist_apply_gradients(self, grads_and_vars, global_step=None, name=None):
    """
      This code is necessary because control_flow_ops.cond does not work with
      Distribution Strategies.
      See: https://github.com/tensorflow/tensorflow/issues/25080
    """

    grads = [g for (g, _) in grads_and_vars]

    is_finite_grad = []
    for g in grads:
      is_finite_grad.append(math_ops.reduce_all(gen_math_ops.is_finite(g)))
    is_overall_finite = math_ops.reduce_all(is_finite_grad)
    # Only update gradients when all grads are finite.
    # def true_apply_gradients_fn():
      # return self._opt.apply_gradients(grads_and_vars, global_step, name)

    ##### Fathom changes begin #####

    #TODO:(elias) Fix cond below
    #Potentially See: https://github.com/tensorflow/tensorflow/issues/4094
    print("Dist strat on")
    update_vars = self._opt.apply_gradients(grads_and_vars, global_step, name)#true_apply_gradients_fn()
    # print("ret early")
    # return update_vars
    # This cond fails when distribution strategies are enabled, we need it on
    # to be robust to overflows.

    # update_vars = control_flow_ops.cond(
    #   is_overall_finite, true_apply_gradients_fn, gen_control_flow_ops.no_op)

    ##### Fathom changes end #####

    # Potentially adjust gradient scale in case of finite gradients.
    return control_flow_ops.group(
        update_vars,
        self._loss_scale_manager.update_loss_scale(is_overall_finite))
