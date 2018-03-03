# coding=utf-8
# Copyright 2017 The Tensor2Tensor Authors.
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

"""Train and evaluate."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib
import os
import sys

# Dependency imports

from tensor2tensor import models  # pylint: disable=unused-import
from tensor2tensor import problems as problems_lib  # pylint: disable=unused-import
from tensor2tensor.utils import decoding
from tensor2tensor.utils import flags as t2t_flags  # pylint: disable=unused-import
from tensor2tensor.utils import registry
from tensor2tensor.utils import trainer_lib
from tensor2tensor.utils import usr_dir


import tensorflow as tf

flags = tf.flags
FLAGS = flags.FLAGS

# See flags.py for additional command-line flags.
flags.DEFINE_string("t2t_usr_dir", None,
                    "Path to a Python module that will be imported. The "
                    "__init__.py file should include the necessary imports. "
                    "The imported files should contain registrations, "
                    "e.g. @registry.register_model calls, that will then be "
                    "available to the t2t-trainer.")
flags.DEFINE_integer("random_seed", 1234, "Random seed.")
flags.DEFINE_integer("tpu_num_shards", 8, "Number of tpu shards.")
flags.DEFINE_integer("iterations_per_loop", 1000,
                     "Number of iterations in a TPU training loop.")
flags.DEFINE_bool("use_tpu", False, "Whether to use TPU.")
flags.DEFINE_integer("tpu_infeed_sleep_secs", None,
                     "How long to sleep the infeed thread.")
flags.DEFINE_bool("generate_data", False, "Generate data before training?")
flags.DEFINE_string("tmp_dir", "/tmp/t2t_datagen",
                    "Temporary storage directory, used if --generate_data.")
flags.DEFINE_bool("profile", False, "Profile performance?")

# To maintain compatibility with some internal libs, we guard against these flag
# definitions possibly erroring. Apologies for the ugliness.
try:
  flags.DEFINE_string("master", "", "Address of TensorFlow master.")
  flags.DEFINE_string("output_dir", "", "Base output directory for run.")

  # Fathom: we changed the default here from continuous_train_and_eval
  # to train_and_evaluate. We did this because
  # continuous_train_and_eval does not work with ValidationMonitor.
  flags.DEFINE_string("schedule", "train_and_evaluate",
                      "Method of Experiment to run.")

  flags.DEFINE_integer("eval_steps", 10000,
                       "Number of steps in evaluation. By default, eval will "
                       "stop after eval_steps or when it runs through the eval "
                       "dataset once in full, whichever comes first, so this "
                       "can be a very large number.")
except:  # pylint: disable=bare-except
  pass


##################
#
# FATHOM ADDITIONS
#
##################
import fathomt2t
import fathomairflow.dags.dag_management.xcom_manipulation as xcom
from fathomairflow.dags.dag_management.task_builders.xcom_keys import (
    XCOM_GCS_MODEL_SUBPATH,
    DATA_DIR, TMP_DIR)
from fathomtf.services.model_management import (upload_model_to_gcs,
                                                fix_paths_for_workspace)
from fh_platform.model_registry import TrainedModelMetadata
import fh_platform.laika as laika
import os
flags.DEFINE_bool("debug_mode", False, "Truncate training for debug purposes")
# NOTE: this is set as REQUIRED, in main()
flags.DEFINE_string("airflow_pipeline_yaml", None,
    "For saving to assets.extra")
flags.DEFINE_string("description", "",
    "Description for this run.  Used in model name.  E.g., 'special_softmax'.")
flags.DEFINE_string("timestamp", "",
    "Timestamp for this run.  This is generally expected to be the DAG execution date,"
    " *not* the timestamp that this specific model was trained.")
flags.DEFINE_string("airflow_url", "", "URL for looking at this task's logs in airflow")
flags.DEFINE_string("airflow_ip", "", "External IP of airflow")
flags.DEFINE_string("hypothesis", "I forgot to specify a hypothesis!",
                    "Hypothesis being tested in this experiment")
flags.DEFINE_bool("debug_laika", False, "Trigger Laika even if in debug mode")
##################
#
# END FATHOM ADDS
#
##################


def get_problem_name():
  problems = FLAGS.problems.split("-")
  assert len(problems) == 1
  return problems[0]


def create_hparams():
  if FLAGS.use_tpu and "tpu" not in FLAGS.hparams_set:
    tf.logging.warn("Not all hyperparameter sets work on TPU. When available "
                    "for a given model, prefer hparams_sets with a '_tpu' "
                    "suffix, e.g. transformer_tpu.")
  return trainer_lib.create_hparams(FLAGS.hparams_set, FLAGS.hparams)


def create_experiment_fn():
  return trainer_lib.create_experiment_fn(
      model_name=FLAGS.model,
      problem_name=get_problem_name(),
      data_dir=os.path.expanduser(FLAGS.data_dir),
      train_steps=FLAGS.train_steps,
      eval_steps=FLAGS.eval_steps,
      min_eval_frequency=FLAGS.local_eval_frequency,
      schedule=FLAGS.schedule,
      export=FLAGS.export_saved_model,
      decode_hparams=decoding.decode_hparams(FLAGS.decode_hparams),
      use_tfdbg=FLAGS.tfdbg,
      use_dbgprofile=FLAGS.dbgprofile,
      eval_early_stopping_steps=FLAGS.eval_early_stopping_steps,
      eval_early_stopping_metric=FLAGS.eval_early_stopping_metric,
      eval_early_stopping_metric_delta=FLAGS.eval_early_stopping_metric_delta,
      eval_early_stopping_metric_minimize=FLAGS.
      eval_early_stopping_metric_minimize,
      use_tpu=FLAGS.use_tpu)



def create_run_config(hp):
  return trainer_lib.create_run_config(
      model_dir=os.path.expanduser(FLAGS.output_dir),
      master=FLAGS.master,
      iterations_per_loop=FLAGS.iterations_per_loop,
      num_shards=FLAGS.tpu_num_shards,
      log_device_placement=FLAGS.log_device_placement,
      save_checkpoints_steps=max(FLAGS.iterations_per_loop,
                                 FLAGS.local_eval_frequency),
      keep_checkpoint_max=FLAGS.keep_checkpoint_max,
      keep_checkpoint_every_n_hours=FLAGS.keep_checkpoint_every_n_hours,
      num_gpus=FLAGS.worker_gpu,
      gpu_order=FLAGS.gpu_order,
      shard_to_cpu=FLAGS.locally_shard_to_cpu,
      num_async_replicas=FLAGS.worker_replicas,
      gpu_mem_fraction=FLAGS.worker_gpu_memory_fraction,
      enable_graph_rewriter=FLAGS.experimental_optimize_placement,
      use_tpu=FLAGS.use_tpu,
      schedule=FLAGS.schedule,
      no_data_parallelism=hp.no_data_parallelism,
      daisy_chain_variables=hp.daisy_chain_variables,
      ps_replicas=FLAGS.ps_replicas,
      ps_job=FLAGS.ps_job,
      ps_gpu=FLAGS.ps_gpu,
      sync=FLAGS.sync,
      worker_id=FLAGS.worker_id,
      worker_job=FLAGS.worker_job,
      random_seed=FLAGS.random_seed,
      tpu_infeed_sleep_secs=FLAGS.tpu_infeed_sleep_secs)


def generate_data():
  # Generate data if requested.
  data_dir = os.path.expanduser(FLAGS.data_dir)
  tmp_dir = os.path.expanduser(FLAGS.tmp_dir)
  tf.gfile.MakeDirs(data_dir)
  tf.gfile.MakeDirs(tmp_dir)

  problem_name = get_problem_name()
  tf.logging.info("Generating data for %s" % problem_name)
  registry.problem(problem_name).generate_data(data_dir, tmp_dir)


@contextlib.contextmanager
def profile_context():
  if FLAGS.profile:
    with tf.contrib.tfprof.ProfileContext("t2tprof",
                                          trace_steps=range(100),
                                          dump_steps=range(100)) as pctx:
      opts = tf.profiler.ProfileOptionBuilder.time_and_memory()
      pctx.add_auto_profiling("op", opts, range(100))
      yield
  else:
    yield


def log_registry():
  if FLAGS.registry_help:
    tf.logging.info(registry.help_string())
    sys.exit(0)


def is_chief():
  schedules = ["train", "train_and_evaluate", "continuous_train_and_eval"]
  return FLAGS.worker_id == 0 and FLAGS.schedule in schedules


def save_metadata(hparams):
  """Saves FLAGS and hparams to output_dir."""
  output_dir = os.path.expanduser(FLAGS.output_dir)
  if not tf.gfile.Exists(output_dir):
    tf.gfile.MakeDirs(output_dir)

  # Save FLAGS in txt file
  if hasattr(FLAGS, "flags_into_string"):
    flags_str = FLAGS.flags_into_string()
    t2t_flags_str = "\n".join([
        "--%s=%s" % (f.name, f.value)
        for f in FLAGS.flags_by_module_dict()[
            "tensor2tensor.utils.flags"]
    ])
  else:
    flags_dict = FLAGS.__dict__["__flags"]
    flags_str = "\n".join(
        ["--%s=%s" % (name, str(f)) for (name, f) in flags_dict.items()])
    t2t_flags_str = None

  flags_txt = os.path.join(output_dir, "flags.txt")
  with tf.gfile.Open(flags_txt, "w") as f:
    f.write(flags_str)

  if t2t_flags_str:
    t2t_flags_txt = os.path.join(output_dir, "flags_t2t.txt")
    with tf.gfile.Open(t2t_flags_txt, "w") as f:
      f.write(t2t_flags_str)

  # Save hparams as hparams.json
  hparams_fname = os.path.join(output_dir, "hparams.json")
  with tf.gfile.Open(hparams_fname, "w") as f:
    f.write(hparams.to_json())


def execute_schedule(exp):
  if not hasattr(exp, FLAGS.schedule):
    raise ValueError(
        "Experiment has no method %s, from --schedule" % FLAGS.schedule)
  with profile_context():
    getattr(exp, FLAGS.schedule)()


# Fathom
def _pick_optimal_model() -> None:
    """Update the checkpoint so that it points to the best model that was
    encountered during training. Here "best" is defined as the lowest or
    highest value of the chosen early stopping metric. (By default,
    lowest loss.)

    We do this automatically based on knowledge of how early stopping
    works; i.e., we take the model that prevented early stopping from
    stopping before it did.
    """

    #if FLAGS.debug_mode:
        #return

    checkpoint_state = tf.train.get_checkpoint_state(FLAGS.output_dir)
    all_checkpoint_paths = list(checkpoint_state.all_model_checkpoint_paths)

    def extract_step(path):
        """Extract the step number from a checkpoint path

        Args:
            path: a path, e.g., model.ckpt-17

        Returns:
            step: the step number as an int, e.g., 17
        """
        return int(path[path.rindex('-') + 1:])

    # get available step numbers
    steps = [(extract_step(path), path) for path in all_checkpoint_paths]
    steps = sorted(steps)
    steps, all_checkpoint_paths = zip(*steps)
    all_checkpoint_paths = list(all_checkpoint_paths)

    # the step we want is the last one that would have allowed us to
    # stop when we did (at steps[-1])
    thresh = steps[-1] - FLAGS.eval_early_stopping_steps

    # get the last step that is <= thresh. Note that the early
    # stopping flags are phrased in terms of step number, not how many
    # times we've run eval.
    best_step_index = [step <= thresh for step in steps].index(False) - 1
    if not FLAGS.debug_mode:
        assert best_step_index >= 0, 'Early stopping stopped before it should have'


    # this is the checkpoint we want
    checkpoint_path = all_checkpoint_paths[best_step_index]

    print('Early stopping chose checkpoint', checkpoint_path)

    tf.train.update_checkpoint_state(
      FLAGS.output_dir,
      checkpoint_path,
      [checkpoint_path])


def main(_):
  # Fathom
  if FLAGS.debug_laika or not FLAGS.debug_mode:
    laika_model = TrainedModelMetadata(
      shortname=FLAGS.description,
      hypothesis=FLAGS.hypothesis,
      gcs_model_path=None,
      debug=FLAGS.debug_mode)
    laika.start_of_training(
      model=laika_model,
      airflow_ip=FLAGS.airflow_ip,
      airflow_url=FLAGS.airflow_url)

  # Fathom
  fix_paths_for_workspace(FLAGS, get_problem_name())

  tf.logging.set_verbosity(tf.logging.INFO)
  trainer_lib.set_random_seed(FLAGS.random_seed)
  usr_dir.import_usr_dir(FLAGS.t2t_usr_dir)
  log_registry()

  if FLAGS.generate_data:
    generate_data()

  # Fathom
  if FLAGS.debug_mode:
    FLAGS.train_steps = 1
    FLAGS.eval_steps = 1

  hparams = create_hparams()
  run_config = create_run_config(hparams)

  if is_chief():
    save_metadata(hparams)

  exp_fn = create_experiment_fn()
  exp = exp_fn(run_config, hparams)
  execute_schedule(exp)

  # Fathom
  #if not FLAGS.debug_mode and FLAGS.eval_early_stopping_steps is not None:
  if FLAGS.eval_early_stopping_steps is not None:
    _pick_optimal_model()
  dir_path, model_name = upload_model_to_gcs(FLAGS=FLAGS)

  # Fathom
  if FLAGS.debug_laika or not FLAGS.debug_mode:
    laika_model = TrainedModelMetadata(
      shortname=FLAGS.description,
      hypothesis=FLAGS.hypothesis,
      gcs_model_path=model_name,
      debug=FLAGS.debug_mode,
      results_dir=FLAGS.output_dir)

    laika.training_succeeded(
      model=laika_model,
      airflow_ip=FLAGS.airflow_ip,
      airflow_url=FLAGS.airflow_url,
    )    
  
  # Fathom
  # NOTE: this must run LAST in the process, to make sure STDOUT is
  # appropriately populated.
  xcom.echo_yaml_for_xcom_ingest({'output_dir': dir_path,
                                  XCOM_GCS_MODEL_SUBPATH: model_name,
                                  DATA_DIR: FLAGS.data_dir,
                                  TMP_DIR: FLAGS.tmp_dir})

if __name__ == "__main__":
  # Fathom
  tf.flags.mark_flag_as_required('airflow_pipeline_yaml')
  tf.flags.mark_flag_as_required('timestamp')

  # Fathom
  try:
    tf.app.run(main)
  except SystemExit:
    pass
  except:
    if FLAGS.debug_laika or not FLAGS.debug_mode:
      laika_model = TrainedModelMetadata(
        shortname=FLAGS.description,
        hypothesis=FLAGS.hypothesis,
        gcs_model_path=None,
        debug=FLAGS.debug_mode,
        results_dir=FLAGS.output_dir)

      laika.training_crashed(
        model=laika_model,
        airflow_ip=FLAGS.airflow_ip,
        airflow_url=FLAGS.airflow_url)
      
    raise
  
