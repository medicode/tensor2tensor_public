# coding=utf-8
# Copyright 2023 The Tensor2Tensor Authors.
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

"""Wrappers around tf.contrib to dynamically import contrib packages.

This makes sure that libraries depending on T2T and TF2, do not crash at import.
"""

from __future__ import absolute_import
# from __future__ import division  # Not necessary in a Python 3-only module
# from __future__ import print_function  # Not necessary in a Python 3-only module

from absl import logging
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import estimator as tf_estimator

"""
# Check if we have contrib available
try:
    from tensorflow.contrib import slim as tf_slim  # pylint: disable=g-import-not-at-top

    is_tf2 = False
except:  # pylint: disable=bare-except
"""
# tf.contrib, including slim and certain optimizers are not available in TF2
# Some features are now available in separate packages. We shim support for
# these as needed.
import tensorflow_addons as tfa  # pylint: disable=g-import-not-at-top
import tf_slim  # pylint: disable=g-import-not-at-top


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