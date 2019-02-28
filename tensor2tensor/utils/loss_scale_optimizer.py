"""Loss scaling optimizer with distribution strategy support added by Fathom."""

from tensor2tensor.utils.loss_scale_manager import debug_tfprint
from tensorflow.contrib.mixed_precision import LossScaleOptimizer
from tensorflow.python.ops import (control_flow_ops, gen_control_flow_ops,
                                   gen_math_ops, math_ops)
from tensorflow.python.training import \
    distribution_strategy_context as distribute_ctx
import tensorflow as tf


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
        # return super().apply_gradients(grads_and_vars, global_step, name)

    def dist_apply_gradients(self, grads_and_vars, global_step=None,
                             name=None):
        """
      This code is necessary because control_flow_ops.cond does not work with
      Distribution Strategies.
      See: https://github.com/tensorflow/tensorflow/issues/25080
    """

        grads = [g for (g, _) in grads_and_vars]

        is_finite_grad = []
        for g in grads:
            is_finite_grad.append(
                math_ops.reduce_all(gen_math_ops.is_finite(g)))
        is_overall_finite = math_ops.reduce_all(is_finite_grad)
        # Only update gradients when all grads are finite.
        no_op = gen_control_flow_ops.no_op

        ##### Fathom changes begin #####

        #TODO:(elias) Fix cond below
        #Potentially See: https://github.com/tensorflow/tensorflow/issues/4094
        # print("Dist strat on")
        update_vars = self._opt.apply_gradients(grads_and_vars, global_step, name)#true_apply_gradients_fn()
        
        # print("ret early")
        return update_vars
        # This cond fails when distribution strategies are enabled, we need it on
        # to be robust to overflows.
        # grads_and_vars[0] = debug_tfprint(message="Grads and vars", tvar=grads_and_vars[0])

        # global_step = debug_tfprint(message="Gstep", tvar=global_step)
        dummy_grad = tf.constant(0, dtype=tf.float32, shape=[1])
        dummy_var = tf.Variable(
            initial_value=0, dtype=tf.float32, expected_shape=[1], trainable=False)
        # dummy_grads_and_vars = [(dummy_grad, dummy_var)]

        print("gradvar", grads_and_vars)
        # print("dumgradvar", dummy_grads_and_vars)
        # in_grad = control_flow_ops.cond(is_overall_finite,
        #                                 lambda: grads_and_vars[0][0],
        #                                 lambda: dummy_grad)
        # in_var = control_flow_ops.cond(
        #     is_overall_finite, lambda: grads_and_vars[0][1], lambda: dummy_var)
        # in_grads_and_vars = [(in_grad, in_var)]

        apply_grad = self._opt.apply_gradients(grads_and_vars, global_step,
                                                name)
        no_op = gen_control_flow_ops.no_op()
        update_vars = tf.cond(is_overall_finite, lambda: apply_grad, lambda: no_op)
        # def true_apply_gradients_fn():
        #   return self._opt.apply_gradients(grads_and_vars, global_step, name)

        # update_vars = control_flow_ops.cond(
        #     is_overall_finite, true_apply_gradients_fn, gen_control_flow_ops.no_op)

        ##### Fathom changes end #####

        # Potentially adjust gradient scale in case of finite gradients.
        return control_flow_ops.group(
            update_vars,
            self._loss_scale_manager.update_loss_scale(is_overall_finite))
