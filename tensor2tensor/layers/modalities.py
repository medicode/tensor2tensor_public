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

"""Modalities define the bottom and top of the model (not the body)."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from six.moves import range  # pylint: disable=redefined-builtin

# commmon_audio is effectively removed from t2t-lite, but if i remove this
# import, there is an error that pops up with eager execution.
from tensor2tensor.layers import common_audio  # pylint: disable=unused-import
from tensor2tensor.layers import common_layers
from tensor2tensor.utils import modality
from tensor2tensor.utils import registry

import tensorflow as tf

import tensorflow_probability as tfp

class SymbolModality(modality.Modality):
  """Modality for sets of discrete symbols.

  Input:
    Embedding.

  Output:
    Linear transformation + softmax.
  """

  @property
  def name(self):
    return "symbol_modality_%d_%d" % (self._vocab_size, self._body_input_depth)

  @property
  def top_is_pointwise(self):
    return True

  @property
  def targets_weights_fn(self):
    weights_fn = common_layers.weights_nonzero

    hp = self._model_hparams
    if hp and hp.prepend_mode != "none":
      assert (hp.prepend_mode == "prepend_inputs_masked_attention" or
              hp.prepend_mode == "prepend_inputs_full_attention")

      if (
          # In masked attention mode, during training, the network try to
          # autoregressively predicting the inputs portion, while the
          # evaluation is only done on the output
          hp.prepend_mode != "prepend_inputs_masked_attention" or
          hp.mode != tf.estimator.ModeKeys.TRAIN):
        weights_fn = common_layers.weights_prepend_inputs_to_targets

    return weights_fn

  def _get_weights(self, hidden_dim=None):
    """Create or get concatenated embedding or softmax variable.

    Args:
      hidden_dim: dim of the variable. Defaults to self._body_input_depth

    Returns:
       a list of self._num_shards Tensors.
    """
    if hidden_dim is None:
      hidden_dim = self._body_input_depth
    num_shards = self._model_hparams.symbol_modality_num_shards
    shards = []
    for i in range(num_shards):
      shard_size = (self._vocab_size // num_shards) + (
          1 if i < self._vocab_size % num_shards else 0)
      var_name = "weights_%d" % i
      shards.append(
          tf.compat.v1.get_variable(
              var_name, [shard_size, hidden_dim],
              initializer=tf.compat.v1.random_normal_initializer(0.0, hidden_dim**-0.5)))
    if num_shards == 1:
      ret = shards[0]
    else:
      ret = tf.concat(shards, 0)
    # Convert ret to tensor.
    if not tf.executing_eagerly():
      ret = common_layers.convert_gradient_to_tensor(ret)
    return ret

  def bottom_simple(self, x, name, reuse):
    with tf.compat.v1.variable_scope(name, reuse=reuse):
      # Ensure the inputs are 3-D
      if len(x.get_shape()) == 4:
        x = tf.squeeze(x, axis=3)
      while len(x.get_shape()) < 3:
        x = tf.expand_dims(x, axis=-1)

      var = self._get_weights()
      x = common_layers.dropout_no_scaling(
          x, 1.0 - self._model_hparams.symbol_dropout)
      ret = common_layers.gather(var, x)
      if self._model_hparams.multiply_embedding_mode == "sqrt_depth":
        ret *= self._body_input_depth**0.5
      ret *= tf.expand_dims(tf.cast(tf.not_equal(x, 0), dtype=tf.float32), -1)
      return ret

  def bottom(self, x):
    if (self._model_hparams.shared_embedding_and_softmax_weights or
        self._model_hparams.get("shared_embedding")):
      return self.bottom_simple(x, "shared", reuse=None)
    return self.bottom_simple(x, "input_emb", reuse=None)

  def targets_bottom(self, x):
    if (self._model_hparams.shared_embedding_and_softmax_weights or
        self._model_hparams.get("shared_embedding")):
      try:
        return self.bottom_simple(x, "shared", reuse=True)
      except ValueError:
        # perhaps there were no inputs, and this is a new variable.
        return self.bottom_simple(x, "shared", reuse=None)
    else:
      return self.bottom_simple(x, "target_emb", reuse=None)

  def top(self, body_output, _):
    """Generate logits.

    Args:
      body_output: A Tensor with shape [batch, p0, p1, body_input_depth]
    Returns:
      logits: A Tensor with shape  [batch, p0, p1, ?, vocab_size].
    """
    if self._model_hparams.symbol_modality_skip_top:
      return tf.expand_dims(body_output, 3)

    if self._model_hparams.shared_embedding_and_softmax_weights:
      scope_name = "shared"
      reuse = True
    else:
      scope_name = "softmax"
      reuse = False

    with tf.compat.v1.variable_scope(scope_name, reuse=reuse):
      body_output_shape = common_layers.shape_list(body_output)
      var = self._get_weights(body_output_shape[-1])
      if (self._model_hparams.factored_logits and
          self._model_hparams.mode == tf.estimator.ModeKeys.TRAIN):
        # insert channels dimension
        body_output = tf.expand_dims(body_output, 3)
        return common_layers.FactoredTensor(body_output, var)
      else:
        body_output = tf.reshape(body_output, [-1, body_output_shape[-1]])
        logits = tf.matmul(body_output, var, transpose_b=True)
        # Just reshape like upstream t2t so that body_output
        # is the expected shape of [B, 1, D]
        # https://github.com/tensorflow/tensor2tensor/blob/d600c8bb196193596fdb38c2b6e5393c4e240564/tensor2tensor/layers/modalities.py#L1135
        return tf.reshape(logits,
                          body_output_shape[:-1] + [1, var.shape[0]])


class SymbolModalityWeightsAll(SymbolModality):
  """SymbolModality for features that do not have 0-padding."""

  @property
  def targets_weights_fn(self):
    return common_layers.weights_all


class SymbolModalityOneHot(SymbolModality):
  """Simple SymbolModality with one hot as embeddings."""

  def bottom(self, x):
    return tf.one_hot(x, self._vocab_size)

  def targets_bottom(self, x):
    return tf.one_hot(x, self._vocab_size)

  def top(self, body_output, _):
    return body_output

  def loss(self, top_out, targets):
    labels = tf.one_hot(targets, self._vocab_size)
    loss = tf.nn.softmax_cross_entropy_with_logits(
        logits=top_out, labels=tf.stop_gradient(labels))
    return tf.reduce_mean(loss), tf.constant(1.0)


class CTCSymbolModality(SymbolModality):
  """SymbolModality that uses CTC loss."""

  def loss(self, top_out, targets):
    """Compute the CTC loss."""
    logits = top_out
    with tf.compat.v1.name_scope("ctc_loss", values=[logits, targets]):
      # For CTC we assume targets are 1d, [batch, length, 1, 1] here.
      targets_shape = targets.get_shape().as_list()
      assert len(targets_shape) == 4
      assert targets_shape[2] == 1
      assert targets_shape[3] == 1
      targets = tf.squeeze(targets, axis=[2, 3])
      logits = tf.squeeze(logits, axis=[2, 3])
      targets_mask = 1 - tf.cast(tf.equal(targets, 0), dtype=tf.int32)
      targets_lengths = tf.reduce_sum(targets_mask, axis=1)
      sparse_targets = tf.keras.backend.ctc_label_dense_to_sparse(
          targets, targets_lengths)
      xent = tf.compat.v1.nn.ctc_loss(
          sparse_targets,
          logits,
          targets_lengths,
          time_major=False,
          preprocess_collapse_repeated=False,
          ctc_merge_repeated=False)
      weights = self.targets_weights_fn(targets)  # pylint: disable=not-callable
      return tf.reduce_sum(xent), tf.reduce_sum(weights)


class ClassLabelModality(modality.Modality):
  """Used for label data."""

  @property
  def name(self):
    return "class_label_modality_%d_%d" % (self._vocab_size,
                                           self._body_input_depth)

  def bottom(self, x):
    with tf.compat.v1.variable_scope(self.name):
      return common_layers.embedding(
          x,
          self._vocab_size,
          self._body_input_depth,
          multiplier=self._body_input_depth**0.5 if
          self._model_hparams.multiply_embedding_mode == "sqrt_depth" else 1.0)

  def targets_bottom(self, x):
    with tf.compat.v1.variable_scope(self.name):
      return tf.zeros(
          [common_layers.shape_list(x)[0], 1, 1, self._body_input_depth])

  def top(self, body_output, _):
    """Transform inputs from model space to target space.

    Average over inner dims and a linear layer to logits.

    Args:
      body_output: A Tensor with shape [batch, ?, ?, body_output_size].

    Returns:
      a Tensors, each with shape [batch_size, ?, ?, vocab_size]
    """
    with tf.compat.v1.variable_scope(self.name):
      x = body_output
      x = tf.reduce_mean(x, axis=[1, 2], keepdims=True)
      res = tf.compat.v1.layers.dense(x, self._vocab_size)
      return tf.expand_dims(res, 3)


class MultiLabelModality(ClassLabelModality):
  """Used for multi label task."""

  @property
  def targets_weights_fn(self):
    """Target weight function for multi label, defaults to nonzero labels."""
    return common_layers.weights_nonzero

  def loss(self, top_out, targets):
    """Average loss over the labels."""
    logits = top_out
    num_labels = tf.shape(targets)[1]
    logits = tf.tile(logits, [1, num_labels, 1, 1, 1])

    xent, weights = common_layers.padded_cross_entropy(
        logits,
        targets,
        self._model_hparams.label_smoothing,
        weights_fn=self.targets_weights_fn,
        reduce_sum=False,
    )
    xent = tf.squeeze(xent, [2, 3])
    weights = tf.squeeze(weights, [2, 3])
    # average loss over all labels
    loss = tf.reduce_sum(xent, axis=1)
    weights = tf.reduce_sum(weights, axis=1)
    loss /= (weights + 1e-8)
    weights = tf.cast(tf.greater(weights, 0.), dtype=tf.float32)

    return tf.reduce_sum(loss*weights), tf.reduce_sum(weights)


class OneHotClassLabelModality(ClassLabelModality):
  """Used for one-hot encoded class labels."""

  def loss(self, top_out, targets):
    """Apply softmax cross-entropy between outputs and targets.

    Args:
      top_out: logits Tensor with shape [batch, ?, ?, num_classes]
      targets: one-hot encoding Tensor with shape [batch, ?, ?, num_classes]
    Returns:
      loss_scale (cross-entropy), loss_denom
    """
    loss_scale = tf.compat.v1.losses.softmax_cross_entropy(
        onehot_labels=targets, logits=top_out)
    weights = self.targets_weights_fn(targets)
    loss_denom = tf.reduce_sum(weights)
    return loss_scale, loss_denom


class IdentityModality(modality.Modality):
  """Does nothing."""

  def bottom(self, x):
    return tf.cast(x, dtype=tf.float32)

  def top(self, body_output, _):
    return body_output


class GenericL2LossModality(IdentityModality):
  """Generic modality with L2 as Loss."""

  def targets_bottom(self, x):
    return tf.cast(x, dtype=tf.float32)

  def loss(self, body_output, targets):
    loss = tf.square(body_output - tf.cast(targets, dtype=tf.float32))
    return tf.reduce_mean(loss), tf.constant(1.0)


class RealModality(modality.Modality):
  """Base class for real (i.e. float) vectors.

  * Bottom is a linear projection layer to hparams.hidden_size.
  * Top is a linear projection layer to vocab_size.
  """

  @property
  def top_is_pointwise(self):
    return True

  def bottom(self, x):
    with tf.compat.v1.variable_scope("real"):
      return tf.compat.v1.layers.dense(
          tf.cast(x, dtype=tf.float32), self._body_input_depth, name="bottom")

  def top(self, body_output, _):
    with tf.compat.v1.variable_scope("real"):
      return tf.compat.v1.layers.dense(body_output, self._vocab_size, name="top")

  def loss(self, top_out, targets):
    raise NotImplementedError()


class RealL2LossModality(RealModality):
  """Modality for real (i.e. float) vectors with L2 (Gaussian) loss."""

  def loss(self, top_out, targets):
    predictions = top_out
    if (len(common_layers.shape_list(top_out)) != len(
        common_layers.shape_list(targets))):
      predictions = tf.squeeze(top_out, axis=[-1])
    with tf.compat.v1.name_scope("l2"):
      weights = self.targets_weights_fn(targets)
      l2 = tf.pow(predictions - targets, 2)
      return tf.reduce_sum(l2 * weights), tf.reduce_sum(weights)


class RealLogPoissonLossModality(RealModality):
  """Modality for real (i.e. float) vectors with log Poisson regression loss."""

  def loss(self, top_out, targets):
    predictions = top_out
    if (len(common_layers.shape_list(top_out)) != len(
        common_layers.shape_list(targets))):
      predictions = tf.squeeze(top_out, axis=[-1])
    with tf.compat.v1.name_scope("log_possion"):
      weights = self.targets_weights_fn(targets)
      lp_loss = tf.nn.log_poisson_loss(targets, predictions)
      return tf.reduce_sum(lp_loss * weights), tf.reduce_sum(weights)


class IdentitySymbolModality(SymbolModality):
  """Symbol modality with identity top and bottom transformations.

  Uses the weights_fn from SymbolModality so that loss/metrics ignore padding.
  """

  def bottom(self, x):
    return tf.cast(x, dtype=tf.float32)

  def top(self, body_output, _):
    return body_output

  def targets_bottom(self, x):
    """SymbolModality overrides targets_bottom, so need to override here too."""
    return self.bottom(x)

  @property
  def top_is_pointwise(self):
    # pointwise mode manipulates body output, not logits, so it fails here.
    return False


class SigmoidClassLabelModality(ClassLabelModality):
  """Sigmoid cross-entropy for independent class labels."""

  @property
  def name(self):
    return "sigmoid_class_symbol_modality_%d_%d" % (self._vocab_size,
                                                    self._body_input_depth)

  def loss(self, top_out, targets):
    # Expect inputs of size [batch-size, timesteps, 1, num-classes], where the
    # last dimension of num-classes represents logits for binary labels
    loss_scale = tf.compat.v1.losses.sigmoid_cross_entropy(
        multi_class_labels=targets, logits=top_out)
    # Weigh all classes equally
    weights = self.targets_weights_fn(targets)
    loss_denom = tf.reduce_sum(weights)
    return loss_scale, loss_denom


class SigmoidMaxPoolingClassLabelModality(ClassLabelModality):
  """Sigmoid cross-entropy applied on max-pooling over timesteps."""

  @property
  def name(self):
    return "sigmoid_max_pooling_class_symbol_modality_%d_%d" % (
        self._vocab_size, self._body_input_depth)

  def top(self, body_output, _):
    """Transform inputs from model space to target space.
    
    Average over inner dims and a linear layer to logits.
    
    Args:
      body_output: A Tensor with shape [batch, timesteps, 1, body_output_size].
    
    Returns:
      a Tensors, each with shape [batch_size, 1, 1, vocab_size]
    """
    with tf.compat.v1.variable_scope(self.name):
      x = body_output
      x = tf.reduce_max(x, axis=1, keepdims=True)
      return tf.compat.v1.layers.dense(x, self._vocab_size)

  def loss(self, top_out, targets):
    # Expect inputs of size [batch-size, 1, 1, num-classes], where the
    # last dimension of num-classes represents logits for binary labels
    loss_scale = tf.compat.v1.losses.sigmoid_cross_entropy(
        multi_class_labels=targets, logits=top_out)
    # Weigh all classes equally
    weights = self.targets_weights_fn(targets)
    loss_denom = tf.reduce_sum(weights)
    return loss_scale, loss_denom


class SoftmaxMaxPoolingClassLabelModality(OneHotClassLabelModality):
  """Softmax cross-entropy applied on max-pooling over timesteps."""

  @property
  def name(self):
    return "softmax_max_pooling_onehot_class_label_modality_%d_%d" % (
        self._vocab_size, self._body_input_depth)

  def top(self, body_output, _):
    with tf.compat.v1.variable_scope(self.name):
      x = body_output
      x = tf.reduce_max(x, axis=1, keepdims=True)
      return tf.compat.v1.layers.dense(x, self._vocab_size)


class SoftmaxAveragePoolingClassLabelModality(OneHotClassLabelModality):
  """Softmax cross-entropy applied on average-pooling over timesteps."""

  @property
  def name(self):
    return "softmax_average_pooling_onehot_class_label_modality_%d_%d" % (
        self._vocab_size, self._body_input_depth)

  def top(self, body_output, _):
    with tf.compat.v1.variable_scope(self.name):
      x = body_output
      x = tf.reduce_mean(x, axis=1, keepdims=True)
      return tf.compat.v1.layers.dense(x, self._vocab_size)


class SoftmaxLastTimestepClassLabelModality(OneHotClassLabelModality):
  """Softmax cross-entropy applied on last-timestep encoding."""

  @property
  def name(self):
    return "softmax_last_timestep_onehot_class_label_modality_%d_%d" % (
        self._vocab_size, self._body_input_depth)

  def top(self, body_output, _):
    with tf.compat.v1.variable_scope(self.name):
      x = body_output
      x = tf.expand_dims(x[:, -1], 1)  # Pick the last timestep
      return tf.compat.v1.layers.dense(x, self._vocab_size)


def create_modality(modality_spec, model_hparams):
  """Creates modality.

  Args:
    modality_spec: tuple ("modality_type:modality_name", vocab_size).
    model_hparams: tf.contrib.training.HParams.

  Returns:
    Modality.

  Raises:
    LookupError: if modality_type is not recognized. See registry.Modalities for
      accepted types.
  """
  modality_full_name, vocab_size = modality_spec
  modality_type, modality_name = parse_modality_name(modality_full_name)

  if modality_type == registry.Modalities.SYMBOL:
    modality_collection = {
        "default": SymbolModality,
        "identity": IdentitySymbolModality,
        "weights_all": SymbolModalityWeightsAll,
        "one_hot": SymbolModalityOneHot,
        "ctc": CTCSymbolModality,
    }
  elif modality_type == registry.Modalities.CLASS_LABEL:
    modality_collection = {
        "default": ClassLabelModality,
        "identity": IdentityModality,
        "multi_label": MultiLabelModality,
        "onehot": OneHotClassLabelModality,
        "sigmoid": SigmoidClassLabelModality,
        "sigmoid_max_pooling": SigmoidMaxPoolingClassLabelModality,
        "onehot_softmax_max_pooling": SoftmaxMaxPoolingClassLabelModality,
        "onehot_softmax_average_pooling":
            SoftmaxAveragePoolingClassLabelModality,
        "onehot_softmax_last_timestep": SoftmaxLastTimestepClassLabelModality,
    }
  elif modality_type == registry.Modalities.GENERIC:
    modality_collection = {
        "default": IdentityModality,
        "l2_loss": GenericL2LossModality,
    }
  elif modality_type == registry.Modalities.REAL:
    modality_collection = {
        "default": RealL2LossModality,
        "identity": IdentityModality,
        "l2_loss": RealL2LossModality,
        "log_poisson_loss": RealLogPoissonLossModality,
    }
  else:
    modality_types = ("symbol", "class_label", "generic", "real")
    raise LookupError("Modality type %s not recognized. Options are: %s" %
                      (modality_type, list(modality_types)))

  return modality_collection[modality_name](model_hparams, vocab_size)


def parse_modality_name(name):
  name_parts = name.split(":")
  if len(name_parts) < 2:
    name_parts.append("default")
  modality_type, modality_name = name_parts
  return modality_type, modality_name
