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
r"""Decode from trained T2T models.

This binary performs inference using the Estimator API.

Example usage to decode from dataset:

  t2t-decoder \
      --data_dir ~/data \
      --problem=algorithmic_identity_binary40 \
      --model=transformer
      --hparams_set=transformer_base

Set FLAGS.decode_interactive or FLAGS.decode_from_file for alternative decode
sources.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

# Fathom
import fathomt2t
from fathomairflow.dags.dag_management.xcom_manipulation import echo_yaml_for_xcom_ingest

# Dependency imports

from tensor2tensor.bin import t2t_trainer
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.utils import decoding
from tensor2tensor.utils import registry
from tensor2tensor.utils import trainer_lib
from tensor2tensor.utils import usr_dir

import tensorflow as tf

flags = tf.flags
FLAGS = flags.FLAGS

# Additional flags in bin/t2t_trainer.py and utils/flags.py
flags.DEFINE_string("checkpoint_path", None,
                    "Path to the model checkpoint. Overrides output_dir.")
flags.DEFINE_string("decode_from_file", None,
                    "Path to the source file for decoding")
flags.DEFINE_string("decode_to_file", None,
                    "Path to the decoded (output) file")
flags.DEFINE_bool("keep_timestamp", False,
                  "Set the mtime of the decoded file to the "
                  "checkpoint_path+'.index' mtime.")
flags.DEFINE_bool("decode_interactive", False,
                  "Interactive local inference mode.")
flags.DEFINE_integer("decode_shards", 1, "Number of decoding replicas.")
flags.DEFINE_string("score_file", "", "File to score. Each line in the file "
                    "must be in the format input \t target.")


# Fathom
flags.DEFINE_bool("fathom_output_predictions", False, "Output predictions based on problem?")
flags.DEFINE_bool("use_original_input", False,
                  "Use the input that was used for validation during training?")
from fathomtf.services.model_management import fathom_t2t_model_setup

def create_hparams():
  return trainer_lib.create_hparams(
      FLAGS.hparams_set,
      FLAGS.hparams,
      data_dir=os.path.expanduser(FLAGS.data_dir),
      problem_name=FLAGS.problem)


def create_decode_hparams():
  decode_hp = decoding.decode_hparams(FLAGS.decode_hparams)
  decode_hp.add_hparam("shards", FLAGS.decode_shards)
  decode_hp.add_hparam("shard_id", FLAGS.worker_id)
  return decode_hp


def decode(estimator, hparams, decode_hp):
  if FLAGS.decode_interactive:
    if estimator.config.use_tpu:
      raise ValueError("TPU can only decode from dataset.")
    decoding.decode_interactively(estimator, hparams, decode_hp,
                                  checkpoint_path=FLAGS.checkpoint_path)
  elif FLAGS.decode_from_file:
    if estimator.config.use_tpu:
      raise ValueError("TPU can only decode from dataset.")
    decoding.decode_from_file(estimator, FLAGS.decode_from_file, hparams,
                              decode_hp, FLAGS.decode_to_file,
                              checkpoint_path=FLAGS.checkpoint_path)
    if FLAGS.checkpoint_path and FLAGS.keep_timestamp:
      ckpt_time = os.path.getmtime(FLAGS.checkpoint_path + ".index")
      os.utime(FLAGS.decode_to_file, (ckpt_time, ckpt_time))
  else:

    # Fathom
    predictions = decoding.decode_from_dataset(
        estimator,
        FLAGS.problem,
        hparams,
        decode_hp,
        decode_to_file=FLAGS.decode_to_file,
        dataset_split="test" if FLAGS.eval_use_test_set else None,
        return_generator=FLAGS.fathom_output_predictions)

    # Fathom
    if FLAGS.fathom_output_predictions:
      print('Assuming only one problem...')
      assert '-' not in FLAGS.problems
      problem = registry.problem(FLAGS.problems)
      problem.output_predictions(predictions)
    

def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  # Fathom
  checkpoint_path = fathom_t2t_model_setup()

  usr_dir.import_usr_dir(FLAGS.t2t_usr_dir)

  if FLAGS.score_file:
    filename = os.path.expanduser(FLAGS.score_file)
    if not tf.gfile.Exists(filename):
      raise ValueError("The file to score doesn't exist: %s" % filename)
    results = score_file(filename)
    if not FLAGS.decode_to_file:
      raise ValueError("To score a file, specify --decode_to_file for results.")
    write_file = open(os.path.expanduser(FLAGS.decode_to_file), "w")
    for score in results:
      write_file.write("%.6f\n" % score)
    write_file.close()
    return

  hp = create_hparams()
  decode_hp = create_decode_hparams()

  estimator = trainer_lib.create_estimator(
      FLAGS.model,
      hp,
      t2t_trainer.create_run_config(hp),
      decode_hparams=decode_hp,
      use_tpu=FLAGS.use_tpu)

  decode(estimator, hp, decode_hp)

  # Fathom
  # This xcom is here so that tasks after decode know the local path to the
  # downloaded model. Train does this same xcom echo.
  # Decode, predict, and evaluate code should
  # converge to use the same fathom_t2t_model_setup.
  echo_yaml_for_xcom_ingest({'output-dir': os.path.dirname(checkpoint_path),
                             'decode_output_file': FLAGS.decode_output_file})


if __name__ == "__main__":
  tf.app.run()
