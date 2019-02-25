"""A mixed precision LossScaleManager with Distribution Strategy support
    added by Fathom."""
import sys
import tensorflow as tf
from tensorflow.contrib.mixed_precision import ExponentialUpdateLossScaleManager
from tensorflow.python.framework import dtypes, ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_control_flow_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from typing import Union, List, Callable, Any, Set
import sys
import itertools


# pylint: disable=line-too-long
def debug_tfprint(message: str = '',
                  tvar: tf.Tensor = None,
                  print_fn: Callable = lambda x: x,
                  summarize: int = -1,
                  name: str = None,
                  disabled: bool = False) -> tf.Tensor:
    """Wraps up tf.print, adding a print_op to the graph.

    The print_op prints the message, the result of the tf operator print_fn to
    which we have given tvar in parameter.

    Note: if debug_mode is False, then this function will return tvar directly
        and will * not * print anything.

    Args:
        tvar: a tensor.
        message: a message to print before anything.
        print_fn: a function to call with tvar, the result of which is what is
            printed. By default, returns tvar, so we just print tvar directly.
            Helpful to print tvar's shape for instance by doing
            print_fn=tf.shape
        summarize: see https://github.com/tensorflow/community/pull/14/files#diff-6dc73e00aed7c8a7fbc0f53d7981f296R122
        name: A name for the operation.
        disabled: if True, will not print anything. Helps turn on/off all
            prints when debugging (or any subset).

    Returns:
        The input tensor tvar.
    """
    if disabled:
        return tvar

    if tvar is None:
        print_fn = lambda x: ""

    print_op = tf.print(
        message,
        print_fn(tvar),
        output_stream=sys.stdout,
        summarize=summarize,
        name=name)
    with tf.control_dependencies([print_op]):
        # Note: without the identity below, the print_op is not consistently
        # added to the graph.
        tvar = tf.identity(tvar)
        return tvar


class FathomDistributedExponentialUpdateLossScaleManager(
        ExponentialUpdateLossScaleManager):
    """
  This class is necessary because the base LossScaleManager doesn't suport
    distribution strategies, and there are no plans to fix that.
    See https://github.com/tensorflow/tensorflow/issues/25080
  """

    def __init__(self,
                 init_loss_scale,
                 incr_every_n_steps,
                 decr_every_n_nan_or_inf=2,
                 incr_ratio=2,
                 decr_ratio=0.8):
        """Constructor of distribution strategy exp-update loss scale manager.

    Args:
      init_loss_scale: A Python float.  The loss scale to use at the beginning.
      incr_every_n_steps: Increases loss scale every n consecutive steps with
        finite gradients.
      decr_every_n_nan_or_inf: Decreases loss scale every n accumulated steps
        with nan or inf gradients.
      incr_ratio: The multiplier to use when increasing the loss scale.
      decr_ratio: The less-than-one-multiplier to use when decreasing the loss
        scale.
    """
        self._incr_every_n_steps = incr_every_n_steps
        self._decr_every_n_nan_or_inf = decr_every_n_nan_or_inf
        self._incr_ratio = incr_ratio
        self._decr_ratio = decr_ratio
        #Fathom | For dist strat
        agg_type = tf.VariableAggregation.ONLY_FIRST_TOWER

        print("Agg type is {}".format(agg_type))
        self._loss_scale = variable_scope.variable(
            name="loss_scale",
            initial_value=ops.convert_to_tensor(init_loss_scale,
                                                dtypes.float32),
            dtype=dtypes.float32,
            trainable=False,
            aggregation=agg_type)
        self._num_good_steps = variable_scope.variable(
            name="good_steps",
            initial_value=0,
            dtype=dtypes.int32,
            trainable=False,
            aggregation=agg_type)
        self._num_bad_steps = variable_scope.variable(
            name="bad_steps",
            initial_value=0,
            dtype=dtypes.int32,
            trainable=False,
            aggregation=agg_type)

    def update_self_if_grads_are_finite(self, finite_grads):
        """This method is necessary, because we cannot wrap changes to self functions in tf conds
      
      Arguments:
        finite_grads {[type]} -- [description]
      """
        next_step_past_incr_thresh = self._num_good_steps + 1 >= self._incr_every_n_steps
        new_loss_scale = control_flow_ops.cond(
            gen_math_ops.is_finite(self._loss_scale * self._incr_ratio),
            lambda: self._loss_scale * self._incr_ratio,
            lambda: self._loss_scale)
        next_step_past_threshold_and_grads_finite = tf.math.logical_and(
            finite_grads, next_step_past_incr_thresh)
        # This incr won't matter if next step past thresh, since we'll reset
        self._num_good_steps = tf.cond(tf.math.logical_and(finite_grads, tf.math.logical_not(next_step_past_incr_thresh)),
            lambda: self._num_good_steps, lambda: self._num_good_steps + 1)
        # Update loss scale and reset stats
        # Update loss scale
        self._loss_scale = tf.cond(next_step_past_threshold_and_grads_finite,
                                   lambda: new_loss_scale,
                                   lambda: self._loss_scale)
        #Reset stats
        self._num_good_steps = tf.cond(
            next_step_past_threshold_and_grads_finite, lambda: 0,
            lambda: self._num_good_steps)
        self._num_bad_steps = tf.cond(
            next_step_past_threshold_and_grads_finite, lambda: 0,
            lambda: self._num_bad_steps)
        # For debugging
        # self._loss_scale = debug_tfprint(
        #     message="This actually ran",
        #     tvar=self._loss_scale,
        #     print_fn=tf.shape)

    def update_self_if_grads_not_finite(self, finite_grads):
        """Branch function when any grad is not finite."""
        should_execute = tf.math.logical_not(finite_grads)
        bad_steps_past_threshold = self._num_bad_steps + 1 >= self._decr_every_n_nan_or_inf

        # Step book keeping if we're not changing the scale
        should_just_update_steps = tf.math.logical_and(should_execute, tf.math.logical_not(bad_steps_past_threshold))
        self._num_bad_steps = tf.cond(should_just_update_steps,
                                      lambda: self._num_bad_steps + 1,
                                      lambda: self._num_bad_steps)
        self._num_good_steps = tf.cond(should_just_update_steps, lambda: 0,
                                       lambda: self._num_good_steps)

        # Change the loss scale and reset stats
        should_reset_loss_scale = tf.math.logical_and(bad_steps_past_threshold, should_execute)
        self._loss_scale = tf.cond(
            should_reset_loss_scale,
            lambda: gen_math_ops.maximum(1., self._loss_scale * self._decr_ratio),
            lambda: self._loss_scale,
        )
        # When loss_scale is updated, both good and bad steps are reset.
        self._num_good_steps = tf.cond(should_reset_loss_scale, lambda: 0,
                                       lambda: self._num_good_steps)
        self._num_bad_steps = tf.cond(should_reset_loss_scale, lambda: 0,
                                      lambda: self._num_bad_steps)

        # self._num_good_steps = debug_tfprint(
        #     message="These are good steps in the neg",
        #     tvar=self._num_bad_steps,
        #     print_fn=tf.shape)

    def update_loss_scale(self, finite_grads):
        """Updates loss scale based on if gradients are finite in current step."""
        #TODO: Fix
        # self._loss_scale = tf.cond(finite_grads, )

        self.update_self_if_grads_are_finite(finite_grads)
        self.update_self_if_grads_not_finite(finite_grads)
        # Incr loss scale
        #next_step >= self._incr_every_n_steps
        # Incr the loss scale
        # Else,
        # If all grads are finite

        # Do below if above is good

        # def update_if_not_finite_grads():
        #   """Branch function when any grad is not finite."""

        #   def decr_loss_scale():
        #     update_op = state_ops.assign(
        #         self._loss_scale,
        #         gen_math_ops.maximum(1., self._loss_scale * self._decr_ratio))
        #     # When loss_scale is updated, both good and bad steps are reset.
        #     return control_flow_ops.group(update_op, self._reset_stats())

        #   def just_update_steps():
        #     # When bad_steps is incremented, good_step is reset.
        #     return control_flow_ops.group(
        #         state_ops.assign_add(self._num_bad_steps, 1),
        #         state_ops.assign(self._num_good_steps, 0))

        #   return control_flow_ops.cond(
        #       next_step >= self._decr_every_n_nan_or_inf,
        #       decr_loss_scale, just_update_steps)
        return tf.group(self._num_good_steps, self._loss_scale,
                        self._num_bad_steps)
        # return control_flow_ops.cond(finite_grads, update_if_finite_grads,
        #                              update_if_not_finite_grads)
