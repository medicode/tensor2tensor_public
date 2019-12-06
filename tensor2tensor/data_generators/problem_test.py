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

"""Test for common problem functionalities."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from fathomt2t_dependencies.common_t2t_utils import pad_to_next_chunk_length

from tensor2tensor.data_generators import algorithmic
from tensor2tensor.data_generators import problem as problem_module
from tensor2tensor.data_generators import problem_hparams
from tensor2tensor.data_generators.problem import default_model_hparams
from tensor2tensor.layers import modalities
from tensor2tensor.utils import registry

import tensorflow as tf


def assert_tensors_equal(sess, t1, t2, n):
  """Compute tensors `n` times and ensure that they are equal."""

  for _ in range(n):

    v1, v2 = sess.run([t1, t2])

    if v1.shape != v2.shape:
      return False

    if not np.all(v1 == v2):
      return False

  return True


class ProblemTest(tf.test.TestCase):

  @classmethod
  def setUpClass(cls):
    algorithmic.TinyAlgo.setup_for_test()

  def testNoShuffleDeterministic(self):
    problem = algorithmic.TinyAlgo()
    dataset = problem.dataset(mode=tf.estimator.ModeKeys.TRAIN,
                              data_dir=algorithmic.TinyAlgo.data_dir,
                              shuffle_files=False)

    tensor1 = dataset.make_one_shot_iterator().get_next()["targets"]
    tensor2 = dataset.make_one_shot_iterator().get_next()["targets"]

    with tf.Session() as sess:
      self.assertTrue(assert_tensors_equal(sess, tensor1, tensor2, 20))

  def testNoShufflePreprocess(self):

    problem = algorithmic.TinyAlgo()
    dataset1 = problem.dataset(mode=tf.estimator.ModeKeys.TRAIN,
                               data_dir=algorithmic.TinyAlgo.data_dir,
                               shuffle_files=False, preprocess=False)
    dataset2 = problem.dataset(mode=tf.estimator.ModeKeys.TRAIN,
                               data_dir=algorithmic.TinyAlgo.data_dir,
                               shuffle_files=False, preprocess=True)

    tensor1 = dataset1.make_one_shot_iterator().get_next()["targets"]
    tensor2 = dataset2.make_one_shot_iterator().get_next()["targets"]

    with tf.Session() as sess:
      self.assertTrue(assert_tensors_equal(sess, tensor1, tensor2, 20))


  @tf.contrib.eager.run_test_in_graph_and_eager_modes()
  def testProblemHparamsModality(self):
    problem = problem_hparams.TestProblem(input_vocab_size=2,
                                          target_vocab_size=3)
    p_hparams = problem.get_hparams()
    self.assertIsInstance(p_hparams.input_modality["inputs"],
                          modalities.SymbolModality)
    self.assertIsInstance(p_hparams.target_modality, modalities.SymbolModality)

  @tf.contrib.eager.run_test_in_graph_and_eager_modes()
  def testProblemHparamsModalityObj(self):
    class ModalityObjProblem(problem_module.Problem):

      def hparams(self, defaults, model_hparams):
        hp = defaults
        hp.input_modality = {
            "inputs": modalities.SymbolModality(model_hparams, 2)}
        hp.target_modality = modalities.SymbolModality(model_hparams, 3)

    problem = ModalityObjProblem(False, False)
    p_hparams = problem.get_hparams()
    self.assertIsInstance(p_hparams.input_modality["inputs"],
                          modalities.SymbolModality)
    self.assertIsInstance(p_hparams.target_modality, modalities.SymbolModality)

  @tf.contrib.eager.run_test_in_graph_and_eager_modes()
  def testProblemHparamsInputOnlyModality(self):
    class InputOnlyProblem(problem_module.Problem):

      def hparams(self, defaults, model_hparams):
        hp = defaults
        hp.input_modality = {"inputs": (registry.Modalities.SYMBOL, 2)}
        hp.target_modality = None

    problem = InputOnlyProblem(False, False)
    p_hparams = problem.get_hparams()
    self.assertIsInstance(p_hparams.input_modality["inputs"],
                          modalities.SymbolModality)
    self.assertIsNone(p_hparams.target_modality)

  @tf.contrib.eager.run_test_in_graph_and_eager_modes()
  def testProblemHparamsTargetOnlyModality(self):
    class TargetOnlyProblem(problem_module.Problem):

      def hparams(self, defaults, model_hparams):
        hp = defaults
        hp.input_modality = {}
        hp.target_modality = (registry.Modalities.SYMBOL, 3)

    problem = TargetOnlyProblem(False, False)
    p_hparams = problem.get_hparams()
    self.assertEqual(p_hparams.input_modality, {})
    self.assertIsInstance(p_hparams.target_modality, modalities.SymbolModality)

  def testChunkPadding(self):
    chunk_size = 4
    with tf.Session() as sess:
      ex_1_inputs = tf.convert_to_tensor([0, 1, 2])
      ex1_targets = tf.convert_to_tensor([2, 3])
      example = {'inputs': ex_1_inputs,
                 'targets': ex1_targets}

      padded_example = sess.run(
        pad_to_next_chunk_length(
            chunk_length=chunk_size,
            axis=0,
            features_to_pad=['inputs'])(example))
      assert padded_example['inputs'].shape == (4, )
      # Should be unchanged
      assert padded_example['targets'].shape == (2, )

  def testMaybeChunk(self):
    class MockedDatasetProblem(algorithmic.TinyAlgo):
      def dataset(self,
                  mode,
                  data_dir=None,
                  num_threads=None,
                  output_buffer_size=None,
                  shuffle_files=None,
                  hparams=None,
                  preprocess=True,
                  dataset_split=None,
                  shard=None,
                  partition_id=0,
                  num_partitions=1,
                  max_records=-1,
                  only_last=False):
        ex_1_inputs = tf.convert_to_tensor([0, 1])
        ex1_targets = tf.convert_to_tensor([2, 3, 5])
        example = {'inputs': ex_1_inputs, 'targets': ex1_targets}
        types = {'inputs': tf.int32, 'targets': tf.int32}
        shapes = {'inputs': (2,), 'targets': (3,)}
        def gen():
          yield example
          return

        import pdb;
        pdb.set_trace()
        return tf.data.Dataset.from_generator(gen, types, shapes)

    problem = MockedDatasetProblem()
    hparams = problem_module.default_model_hparams()
    hparams.chunk_length = 100
    hparams.batch_size = 500
    hparams.max_length = 500
    hparams.min_length = 0
    hparams.max_input_seq_length = 100
    hparams.max_target_seq_length = 100
    hparams.eval_drop_long_sequences = False
    dataset = problem.input_fn(mode=tf.estimator.ModeKeys.EVAL,
                               hparams=hparams)
    import pdb;
    pdb.set_trace()
    asserts = []
    example = dataset.make_one_shot_iterator().get_next()
    asserts.append(
      tf.Assert(
        tf.shape(example['inputs'])[0] % hparams.chunk_length == 0,
        [example['inputs']]))
    asserts.append(
      tf.Assert(
        tf.shape(example['targets'])[0] % hparams.chunk_length == 0,
        [example['targets']]))

    with tf.Session() as sess:
      with tf.control_dependencies(asserts):
        for assert_op in asserts:
          sess.run(assert_op)


if __name__ == "__main__":
  tf.test.main()
