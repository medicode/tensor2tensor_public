# coding=utf-8
# Copyright 2018 The Tensor2Tensor Authors.
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

"""Tests for T2TModel."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os

from tensor2tensor.utils import t2t_model
from tensor2tensor.utils import trainer_lib

import tensorflow as tf


class T2TModelTest(tf.test.TestCase):

  @tf.contrib.eager.run_test_in_graph_and_eager_modes()
  def testSummarizeLosses(self):
    with tf.Graph().as_default():
      model = t2t_model.T2TModel(tf.contrib.training.HParams())
      losses = {"training": tf.random_normal([]),
                "extra": tf.random_normal([])}
      outputs = model._summarize_losses(losses)
      self.assertIsNone(outputs, None)
      self.assertEquals(
          len(tf.get_collection(tf.GraphKeys.SUMMARIES, scope="losses")),
          len(losses))

  @tf.contrib.eager.run_test_in_graph_and_eager_modes()
  def testInvalidCkptPath(self):
    with tf.Graph().as_default():
      model = t2t_model.T2TModel(tf.contrib.training.HParams())

      with self.assertRaises(AssertionError):
        model.initialize_from_ckpt("invalid+")
      
      with self.assertRaises(AssertionError):
        model.initialize_from_ckpt("invalid]")

      with self.assertRaises(AssertionError):
        model.initialize_from_ckpt("invalid[")
  
  @tf.contrib.eager.run_test_in_graph_and_eager_modes()
  def testInvalidModelPath(self):
    def get_data_dir():
      pkg, _ = os.path.split(__file__)
      pkg, _ = os.path.split(pkg)
      return os.path.join(pkg, "test_data")
    
    
    _DATA_DIR = get_data_dir()
    _CKPT_DIR = os.path.join(_DATA_DIR, "transformer_test_ckpt")
    with tf.Graph().as_default():
      hparams = tf.contrib.training.HParams()

      with self.assertRaises(AssertionError):
        hparams.model_dir = "invalid+"
        self.assertEquals(hparams.get("model_dir", None), "invalid+")

        model = t2t_model.T2TModel(hparams)
        model.set_mode(tf.estimator.ModeKeys.TRAIN)
        model.initialize_from_ckpt("valid")

      with self.assertRaises(AssertionError):
        hparams.model_dir = "invalid["
        self.assertEquals(hparams.get("model_dir", None), "invalid+")

        model = t2t_model.T2TModel(hparams)
        model.set_mode(tf.estimator.ModeKeys.TRAIN)
        model.initialize_from_ckpt("valid")
      
      with self.assertRaises(AssertionError):
        hparams.model_dir = "invalid]"
        self.assertEquals(hparams.get("model_dir", None), "invalid+")

        model = t2t_model.T2TModel(hparams)
        model.set_mode(tf.estimator.ModeKeys.TRAIN)
        model.initialize_from_ckpt("valid")
    
if __name__ == "__main__":
  tf.test.main()
