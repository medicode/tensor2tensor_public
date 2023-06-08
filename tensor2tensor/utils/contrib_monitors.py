# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Monitors instrument the training process (deprecated).

This module and all its submodules are deprecated. See
[contrib/learn/README.md](https://www.tensorflow.org/code/tensorflow/contrib/learn/README.md)
for migration instructions.

@@get_default_monitors
@@BaseMonitor
@@CaptureVariable
@@CheckpointSaver
@@EveryN
@@ExportMonitor
@@GraphDump
@@LoggingTrainable
@@NanLoss
@@PrintTensor
@@StepCounter
@@StopAtStep
@@SummarySaver
@@ValidationMonitor
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import time
import six

from tensorflow.python.estimator import estimator as core_estimator
from tensorflow.python.framework import ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import checkpoint_management
from tensorflow.python.training import session_run_hook
from tensorflow.python.training import training_util
from tensorflow.python.util import deprecation
from tensorflow.python.util import tf_inspect


# TODO(ptucker): Split each monitor class into a separate file.
# TODO(ptucker): Fail if epoch or step does not monotonically increase?
class BaseMonitor(object):
    """Base class for Monitors.

    THIS CLASS IS DEPRECATED. See
    [contrib/learn/README.md](https://www.tensorflow.org/code/tensorflow/contrib/learn/README.md)
    for general migration instructions.

    Defines basic interfaces of Monitors.
    Monitors can either be run on all workers or, more commonly, restricted
    to run exclusively on the elected chief worker.
    """

    @deprecation.deprecated(
        "2016-12-05",
        "Monitors are deprecated. Please use tf.train.SessionRunHook.")
    def __init__(self):
        self._begun = False
        self._current_epoch = None
        self._current_step = None
        self._max_steps = None
        self._estimator = None

    @property
    def run_on_all_workers(self):
        return False

    def set_estimator(self, estimator):
        """A setter called automatically by the target estimator.

        If the estimator is locked, this method does nothing.

        Args:
          estimator: the estimator that this monitor monitors.

        Raises:
          ValueError: if the estimator is None.
        """
        if estimator is None:
            raise ValueError("Missing estimator.")
        # TODO(mdan): This should fail if called twice with the same estimator.
        self._estimator = estimator

    def begin(self, max_steps=None):
        """Called at the beginning of training.

        When called, the default graph is the one we are executing.

        Args:
          max_steps: `int`, the maximum global step this training will run until.

        Raises:
          ValueError: if we've already begun a run.
        """
        if self._begun:
            raise ValueError("begin called twice without end.")
        self._max_steps = max_steps
        self._begun = True

    def end(self, session=None):
        """Callback at the end of training/evaluation.

        Args:
          session: A `tf.Session` object that can be used to run ops.

        Raises:
          ValueError: if we've not begun a run.
        """
        _ = session
        if not self._begun:
            raise ValueError("end called without begin.")
        self._max_steps = None
        self._begun = False

    def epoch_begin(self, epoch):
        """Begin epoch.

        Args:
          epoch: `int`, the epoch number.

        Raises:
          ValueError: if we've already begun an epoch, or `epoch` < 0.
        """
        if self._current_epoch is not None:
            raise ValueError("epoch_begin called twice without epoch_end.")
        if epoch < 0:
            raise ValueError("Invalid epoch %s." % epoch)
        self._current_epoch = epoch

    def epoch_end(self, epoch):
        """End epoch.

        Args:
          epoch: `int`, the epoch number.

        Raises:
          ValueError: if we've not begun an epoch, or `epoch` number does not match.
        """
        if self._current_epoch != epoch:
            raise ValueError("epoch_end expected %s but got %s.", self._current_epoch,
                             epoch)
        self._current_epoch = None

    def step_begin(self, step):
        """Callback before training step begins.

        You may use this callback to request evaluation of additional tensors
        in the graph.

        Args:
          step: `int`, the current value of the global step.

        Returns:
          List of `Tensor` objects or string tensor names to be run.

        Raises:
          ValueError: if we've already begun a step, or `step` < 0, or
              `step` > `max_steps`.
        """
        if (step < 0) or ((self._max_steps is not None) and
                          (step > self._max_steps)):
            raise ValueError("Invalid step %s." % step)
        self._current_step = step
        return []

    def step_end(self, step, output):  # pylint: disable=unused-argument
        """Callback after training step finished.

        This callback provides access to the tensors/ops evaluated at this step,
        including the additional tensors for which evaluation was requested in
        `step_begin`.

        In addition, the callback has the opportunity to stop training by returning
        `True`. This is useful for early stopping, for example.

        Note that this method is not called if the call to `Session.run()` that
        followed the last call to `step_begin()` failed.

        Args:
          step: `int`, the current value of the global step.
          output: `dict` mapping `string` values representing tensor names to
            the value resulted from running these tensors. Values may be either
            scalars, for scalar tensors, or Numpy `array`, for non-scalar tensors.

        Returns:
          `bool`. True if training should stop.

        Raises:
          ValueError: if we've not begun a step, or `step` number does not match.
        """
        if self._current_step != step:
            raise ValueError("step_end expected %s but got %s.", self._current_step,
                             step)
        self._current_step = None
        return False

    def post_step(self, step, session):  # pylint: disable=unused-argument
        """Callback after the step is finished.

        Called after step_end and receives session to perform extra session.run
        calls. If failure occurred in the process, will be called as well.

        Args:
          step: `int`, global step of the model.
          session: `Session` object.
        """
        _ = step, session


class EveryN(BaseMonitor):
    """Base class for monitors that execute callbacks every N steps.

    THIS CLASS IS DEPRECATED. See
    [contrib/learn/README.md](https://www.tensorflow.org/code/tensorflow/contrib/learn/README.md)
    for general migration instructions.

    This class adds three new callbacks:
      - every_n_step_begin
      - every_n_step_end
      - every_n_post_step

    The callbacks are executed every n steps, or optionally every step for the
    first m steps, where m and n can both be user-specified.

    When extending this class, note that if you wish to use any of the
    `BaseMonitor` callbacks, you must call their respective super implementation:

      def step_begin(self, step):
        super(ExampleMonitor, self).step_begin(step)
        return []

    Failing to call the super implementation will cause unpredictable behavior.

    The `every_n_post_step()` callback is also called after the last step if it
    was not already called through the regular conditions.  Note that
    `every_n_step_begin()` and `every_n_step_end()` do not receive that special
    treatment.

    """

    # TODO(ipolosukhin): Add also every n seconds.

    def __init__(self, every_n_steps=100, first_n_steps=1):
        """Initializes an `EveryN` monitor.

        Args:
          every_n_steps: `int`, the number of steps to allow between callbacks.
          first_n_steps: `int`, specifying the number of initial steps during
            which the callbacks will always be executed, regardless of the value
            of `every_n_steps`. Note that this value is relative to the global step
        """
        super(EveryN, self).__init__()
        self._every_n_steps = every_n_steps
        self._first_n_steps = first_n_steps
        # Last step in the model.
        self._last_successful_step = None
        # Last step at which we called one of the every_n methods
        self._last_active_step = 0
        self._every_n_step_begin_called = False

    def every_n_step_begin(self, step):  # pylint: disable=unused-argument
        """Callback before every n'th step begins.

        Args:
          step: `int`, the current value of the global step.

        Returns:
          A `list` of tensors that will be evaluated at this step.
        """
        return []

    def every_n_step_end(self, step, outputs):  # pylint: disable=unused-argument
        """Callback after every n'th step finished.

        This callback provides access to the tensors/ops evaluated at this step,
        including the additional tensors for which evaluation was requested in
        `step_begin`.

        In addition, the callback has the opportunity to stop training by returning
        `True`. This is useful for early stopping, for example.

        Args:
          step: `int`, the current value of the global step.
          outputs: `dict` mapping `string` values representing tensor names to
            the value resulted from running these tensors. Values may be either
            scalars, for scalar tensors, or Numpy `array`, for non-scalar tensors.

        Returns:
          `bool`. True if training should stop.
        """
        return False

    def every_n_post_step(self, step, session):
        """Callback after a step is finished or `end()` is called.

        Args:
          step: `int`, the current value of the global step.
          session: `Session` object.
        """
        pass

    def step_begin(self, step):
        """Overrides `BaseMonitor.step_begin`.

        When overriding this method, you must call the super implementation.

        Args:
          step: `int`, the current value of the global step.
        Returns:
          A `list`, the result of every_n_step_begin, if that was called this step,
          or an empty list otherwise.

        Raises:
          ValueError: if called more than once during a step.
        """
        super(EveryN, self).step_begin(step)
        if (step <= self._first_n_steps or
            step >= (self._every_n_steps + self._last_active_step) or
            step == self._max_steps):  # Note: max_steps can be None here.
            self._every_n_step_begin_called = True
            return self.every_n_step_begin(step)
        self._every_n_step_begin_called = False
        return []

    def step_end(self, step, output):
        """Overrides `BaseMonitor.step_end`.

        When overriding this method, you must call the super implementation.

        Args:
          step: `int`, the current value of the global step.
          output: `dict` mapping `string` values representing tensor names to
            the value resulted from running these tensors. Values may be either
            scalars, for scalar tensors, or Numpy `array`, for non-scalar tensors.
        Returns:
          `bool`, the result of every_n_step_end, if that was called this step,
          or `False` otherwise.
        """
        super(EveryN, self).step_end(step, output)
        if self._every_n_step_begin_called:
            return self.every_n_step_end(step, output)
        return False

    def post_step(self, step, session):
        super(EveryN, self).post_step(step, session)
        if self._every_n_step_begin_called:
            self.every_n_post_step(step, session)
            self._last_active_step = step
        self._last_successful_step = step

    def end(self, session=None):
        super(EveryN, self).end(session=session)
        if self._last_successful_step != self._last_active_step:
            self.every_n_post_step(self._last_successful_step, session)


class ValidationMonitor(EveryN):
    """Runs evaluation of a given estimator, at most every N steps.

    THIS CLASS IS DEPRECATED. See
    [contrib/learn/README.md](https://www.tensorflow.org/code/tensorflow/contrib/learn/README.md)
    for general migration instructions.

    Note that the evaluation is done based on the saved checkpoint, which will
    usually be older than the current step.

    Can do early stopping on validation metrics if `early_stopping_rounds` is
    provided.
    """

    def __init__(self,
                 x=None,
                 y=None,
                 input_fn=None,
                 batch_size=None,
                 eval_steps=None,
                 every_n_steps=100,
                 metrics=None,
                 hooks=None,
                 early_stopping_rounds=None,
                 early_stopping_metric="loss",
                 early_stopping_metric_minimize=True,
                 name=None,
                 check_interval_secs=5):
        """Initializes a ValidationMonitor.

        Args:
          x: See `BaseEstimator.evaluate`.
          y: See `BaseEstimator.evaluate`.
          input_fn: See `BaseEstimator.evaluate`.
          batch_size: See `BaseEstimator.evaluate`.
          eval_steps: See `BaseEstimator.evaluate`.
          every_n_steps: Check for new checkpoints to evaluate every N steps. If a
              new checkpoint is found, it is evaluated. See `EveryN`.
          metrics: See `BaseEstimator.evaluate`.
          hooks: A list of `SessionRunHook` hooks to pass to the
            `Estimator`'s `evaluate` function.
          early_stopping_rounds: `int`. If the metric indicated by
              `early_stopping_metric` does not change according to
              `early_stopping_metric_minimize` for this many steps, then training
              will be stopped.
          early_stopping_metric: `string`, name of the metric to check for early
              stopping.
          early_stopping_metric_minimize: `bool`, True if `early_stopping_metric` is
              expected to decrease (thus early stopping occurs when this metric
              stops decreasing), False if `early_stopping_metric` is expected to
              increase. Typically, `early_stopping_metric_minimize` is True for
              loss metrics like mean squared error, and False for performance
              metrics like accuracy.
          name: See `BaseEstimator.evaluate`.
          check_interval_secs: Only check for new checkpoint if at least
              `check_interval_secs` have passed. Ignore if None. Default is 5 secs.


        Raises:
          ValueError: If both x and input_fn are provided.
        """
        super(ValidationMonitor, self).__init__(
            every_n_steps=every_n_steps, first_n_steps=-1)
        # TODO(mdan): Checks like this are already done by evaluate.
        if x is None and input_fn is None:
            raise ValueError("Either x or input_fn should be provided.")
        self.x = x
        self.y = y
        self.input_fn = input_fn
        self.batch_size = batch_size
        self.eval_steps = eval_steps
        self.metrics = metrics
        self.hooks = hooks
        self.early_stopping_rounds = early_stopping_rounds
        self.early_stopping_metric = early_stopping_metric
        self.early_stopping_metric_minimize = early_stopping_metric_minimize
        self.name = name
        self._best_value_step = None
        self._best_value = None
        self._best_metrics = None
        self._early_stopped = False
        self._latest_path = None
        self._latest_path_step = None
        self._last_checkpoint_check_time = None
        self._check_interval_secs = check_interval_secs

    @property
    def early_stopped(self):
        """Returns True if this monitor caused an early stop."""
        return self._early_stopped

    @property
    def best_step(self):
        """Returns the step at which the best early stopping metric was found."""
        return self._best_value_step

    @property
    def best_value(self):
        """Returns the best early stopping metric value found so far."""
        return self._best_value

    @property
    def best_metrics(self):
        """Returns all eval metrics computed with the best early stopping metric.

        For instance, if the metrics computed in two successive evals are
        1. {'loss':40, 'auc':0.5}
        2. {'loss':50, 'auc':0.6}
        this function would return the first dict {'loss':40, 'auc':0.5} after both
        first and second eval (if `early_stopping_metric` is 'loss' and
        `early_stopping_metric_minimize` is True).

        Returns:
          The output dict of estimator.evaluate which contains the best value of
          the early stopping metric seen so far.
        """
        return self._best_metrics

    def _evaluate_estimator(self):
        if isinstance(self._estimator, core_estimator.Estimator):
            if any((x is not None
                    for x in [self.x, self.y, self.batch_size, self.metrics])):
                raise ValueError(
                    "tf.estimator.Estimator does not support following "
                    "arguments: x, y, batch_size, metrics. Should set as `None` "
                    "in ValidationMonitor")
            return self._estimator.evaluate(
                input_fn=self.input_fn,
                steps=self.eval_steps,
                hooks=self.hooks,
                name=self.name)
        else:
            return self._estimator.evaluate(
                x=self.x,
                y=self.y,
                input_fn=self.input_fn,
                batch_size=self.batch_size,
                steps=self.eval_steps,
                metrics=self.metrics,
                hooks=self.hooks,
                name=self.name)

    def every_n_step_end(self, step, outputs):
        super(ValidationMonitor, self).every_n_step_end(step, outputs)
        # TODO(mdan): The use of step below is probably misleading.
        # The code should probably use the step from the checkpoint, because
        # that's what is being evaluated.
        if self._estimator is None:
            raise ValueError("Missing call to set_estimator.")
        current_time = time.time()
        if (self._check_interval_secs is not None and
            self._last_checkpoint_check_time is not None and
            current_time - self._last_checkpoint_check_time <=
            self._check_interval_secs):
            logging.debug(
                "Skipping evaluation since less than %d seconds have passed since "
                "last check for a new checkpoint.", self._check_interval_secs)
            return False
        self._last_checkpoint_check_time = current_time
        # Check that we are not running evaluation on the same checkpoint.
        latest_path = checkpoint_management.latest_checkpoint(
            self._estimator.model_dir)
        if latest_path is None:
            logging.debug("Skipping evaluation since model has not been saved yet "
                          "at step %d.", step)
            return False
        if latest_path is not None and latest_path == self._latest_path:
            logging.debug("Skipping evaluation due to same checkpoint %s for step %d "
                          "as for step %d.", latest_path, step,
                          self._latest_path_step)
            return False
        self._latest_path = latest_path
        self._latest_path_step = step

        # Run evaluation and log it.
        validation_outputs = self._evaluate_estimator()
        stats = []
        for name in validation_outputs:
            stats.append("%s = %s" % (name, str(validation_outputs[name])))
        logging.info("Validation (step %d): %s", step, ", ".join(stats))

        # Early stopping logic.
        if self.early_stopping_rounds is not None:
            if self.early_stopping_metric not in validation_outputs:
                raise ValueError("Metric %s missing from outputs %s." %
                                 (self.early_stopping_metric,
                                  set(validation_outputs.keys())))
            current_value = validation_outputs[self.early_stopping_metric]
            if (self._best_value is None or (self.early_stopping_metric_minimize and
                                             (current_value < self._best_value)) or
                (not self.early_stopping_metric_minimize and
                 (current_value > self._best_value))):
                self._best_value = current_value
                self._best_metrics = copy.deepcopy(validation_outputs)
                self._best_value_step = step
            stop_now = (step - self._best_value_step >= self.early_stopping_rounds)
            if stop_now:
                logging.info("Stopping. Best step: {} with {} = {}.".format(
                    self._best_value_step, self.early_stopping_metric,
                    self._best_value))
                self._early_stopped = True
                return True
        return False


def _as_graph_element(obj):
    """Retrieves Graph element."""
    graph = ops.get_default_graph()
    if not isinstance(obj, six.string_types):
        if not hasattr(obj, "graph") or obj.graph != graph:
            raise ValueError("Passed %s should have graph attribute that is equal "
                             "to current graph %s." % (obj, graph))
        return obj
    if ":" in obj:
        element = graph.as_graph_element(obj)
    else:
        element = graph.as_graph_element(obj + ":0")
        # Check that there is no :1 (e.g. it's single output).
        try:
            graph.as_graph_element(obj + ":1")
        except (KeyError, ValueError):
            pass
        else:
            raise ValueError("Name %s is ambiguous, "
                             "as this `Operation` has multiple outputs "
                             "(at least 2)." % obj)
    return element


class RunHookAdapterForMonitors(session_run_hook.SessionRunHook):
    """Wraps monitors into a SessionRunHook."""

    def __init__(self, monitors):
        self._monitors = monitors

    def begin(self):
        self._last_step = None
        self._global_step_tensor = training_util.get_global_step()
        for m in self._monitors:
            m.begin(max_steps=None)

    def before_run(self, run_context):
        if self._last_step is None:
            self._last_step = run_context.session.run(self._global_step_tensor) + 1

        request = {self._global_step_tensor: self._global_step_tensor}
        monitor_fetches = []
        for m in self._monitors:
            monitor_requests = m.step_begin(self._last_step)
            if monitor_requests:
                if not isinstance(monitor_requests, list):
                    raise ValueError("Monitor.step_begin should return a list.")
                monitor_fetches.extend(monitor_requests)
        if monitor_fetches:
            request["monitors"] = dict(
                zip(monitor_fetches, [_as_graph_element(f) for f in monitor_fetches]))

        return session_run_hook.SessionRunArgs(request)

    def after_run(self, run_context, run_values):
        result = run_values.results[
            "monitors"] if "monitors" in run_values.results else {}
        for m in self._monitors:
            induce_stop = m.step_end(self._last_step, result)
            if induce_stop:
                run_context.request_stop()

        for m in self._monitors:
            m.post_step(self._last_step, run_context.session)

        self._last_step = run_values.results[self._global_step_tensor] + 1

    def end(self, session):
        self._last_step = None
        for m in self._monitors:
            if "session" in tf_inspect.getargspec(m.end).args:
                m.end(session=session)
            else:
                m.end()


def replace_monitors_with_hooks(monitors_or_hooks, estimator):
    """Wraps monitors with a hook.

    `Monitor` is deprecated in favor of `SessionRunHook`. If you're using a
    monitor, you can wrap it with a hook using function. It is recommended to
    implement hook version of your monitor.

    Args:
      monitors_or_hooks: A `list` may contain both monitors and hooks.
      estimator: An `Estimator` that monitor will be used with.

    Returns:
      Returns a list of hooks. If there is any monitor in the given list, it is
      replaced by a hook.
    """
    monitors_or_hooks = monitors_or_hooks or []
    hooks = [
        m for m in monitors_or_hooks
        if isinstance(m, session_run_hook.SessionRunHook)
    ]

    deprecated_monitors = [
        m for m in monitors_or_hooks
        if not isinstance(m, session_run_hook.SessionRunHook)
    ]

    if not estimator.config.is_chief:
        # Prune list of monitor to the ones runnable on all workers.
        deprecated_monitors = [
            m for m in deprecated_monitors if m.run_on_all_workers
        ]

    # Setup monitors.
    for monitor in deprecated_monitors:
        monitor.set_estimator(estimator)

    if deprecated_monitors:
        hooks.append(RunHookAdapterForMonitors(deprecated_monitors))

    return hooks
