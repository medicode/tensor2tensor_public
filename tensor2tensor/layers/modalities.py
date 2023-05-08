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

"""Modalities, which specify a feature's domain.

T2TModel applies a default transformation to each feature according to its
modality. Override them by specifying a model's
hparams.{bottom,loss,top,weights_fn}.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import range  # pylint: disable=redefined-builtin

from tensor2tensor.layers import common_attention
from tensor2tensor.layers import common_audio
from tensor2tensor.layers import common_image_attention as cia
from tensor2tensor.layers import common_layers
from tensor2tensor.layers import common_video
from tensor2tensor.layers import discretization

import tensorflow as tf
from tensorflow.compat.v1 import estimator as tf_estimator
import tensorflow_probability as tfp


class ModalityType(object):
  """Types of modalities."""

  AUDIO = "audio"
  AUDIO_SPECTRAL = "audio_spectral"
  CLASS_LABEL = "class_label"
  CTC_SYMBOL = "ctc_symbol"  # symbol with CTC loss
  GENERIC_L2_LOSS = "generic_l2"  # identity modality with L2 loss
  IDENTITY = "identity"  # identity top and bottom
  IDENTITY_SYMBOL = "identity_symbol"  # symbol with identity top and bottom
  IMAGE = "image"
  # images using channel compression for generation
  IMAGE_CHANNEL_BOTTOM_IDENTITY = "image_channel_bottom_identity"
  # images using channel compression for generation
  IMAGE_CHANNEL_COMPRESS = "image_channel_compress"
  IMAGE_CHANNEL_EMBEDDINGS_BOTTOM = "image_channel_embeddings_bottom"
  MULTI_LABEL = "multi_label"
  ONE_HOT_CLASS_LABEL = "one_hot_class_label"
  REAL = "real"  # real vectors
  REAL_L2_LOSS = "real_l2"  # real vectors with L2 as loss
  # real vectors with log Poisson regression loss
  REAL_LOG_POISSON_LOSS = "real_log_poisson"
  SIGMOID_CLASS_LABEL = "sigmoid_class_label"  # sigmoid cross-entropy loss
  # sigmoid cross-entropy applied on max-pooling over timesteps
  SIGMOID_MAX_POOLING_CLASS_LABEL = "sigmoid_max_pooling_class_label"
  # softmax cross-entropy applied on average-pooling over timesteps
  SOFTMAX_AVERAGE_POOLING_CLASS_LABEL = "softmax_average_pooling_class_label"
  # softmax cross-entropy applied on last-timestep encoding
  SOFTMAX_LAST_TIMESTEP_CLASS_LABEL = "softmax_last_timestep_class_label"
  # softmax cross-entropy applied on max-pooling over timesteps
  SOFTMAX_MAX_POOLING_CLASS_LABEL = "softmax_max_pooling_class_label"
  SPEECH_RECOGNITION = "speech_recognition"
  SYMBOL = "symbol"
  SYMBOL_WEIGHTS_ALL = "symbol_weights_all"  # symbol for features w/o 0-padding
  SYMBOL_ONE_HOT = "symbol_one_hot"  # symbol with one hot as embeddings
  VIDEO = "video"
  VIDEO_BITWISE = "video_bitwise"  # video where bottom embeds pixels bitwise
  VIDEO_IDENTITY = "video_identity"  # video with identity top and bottom
  VIDEO_L1 = "video_l1"  # video with L2 loss
  VIDEO_L2 = "video_l2"  # video with L1 loss
  # video with L1 loss and raw input (sequences of frames)
  VIDEO_L1_RAW = "video_l1_raw"
  # video with L2 loss and raw input (sequences of frames)
  VIDEO_L2_RAW = "video_l2_raw"
  # video with pixel noise on input during training
  VIDEO_PIXEL_NOISE = "video_pixel_noise"

  @staticmethod
  def get_choices():
    return [
        ModalityType.AUDIO,
        ModalityType.AUDIO_SPECTRAL,
        ModalityType.CLASS_LABEL,
        ModalityType.CTC_SYMBOL,
        ModalityType.GENERIC_L2_LOSS,
        ModalityType.IDENTITY,
        ModalityType.IDENTITY_SYMBOL,
        ModalityType.IMAGE,
        ModalityType.IMAGE_CHANNEL_BOTTOM_IDENTITY,
        ModalityType.IMAGE_CHANNEL_COMPRESS,
        ModalityType.IMAGE_CHANNEL_EMBEDDINGS_BOTTOM,
        ModalityType.MULTI_LABEL,
        ModalityType.ONE_HOT_CLASS_LABEL,
        ModalityType.REAL,
        ModalityType.REAL_L2_LOSS,
        ModalityType.REAL_LOG_POISSON_LOSS,
        ModalityType.SIGMOID_CLASS_LABEL,
        ModalityType.SIGMOID_MAX_POOLING_CLASS_LABEL,
        ModalityType.SOFTMAX_AVERAGE_POOLING_CLASS_LABEL,
        ModalityType.SOFTMAX_LAST_TIMESTEP_CLASS_LABEL,
        ModalityType.SOFTMAX_MAX_POOLING_CLASS_LABEL,
        ModalityType.SPEECH_RECOGNITION,
        ModalityType.SYMBOL,
        ModalityType.SYMBOL_ONE_HOT,
        ModalityType.SYMBOL_WEIGHTS_ALL,
        ModalityType.VIDEO,
        ModalityType.VIDEO_BITWISE,
        ModalityType.VIDEO_IDENTITY,
        ModalityType.VIDEO_L1,
        ModalityType.VIDEO_L2,
        ModalityType.VIDEO_L1_RAW,
        ModalityType.VIDEO_L2_RAW,
        ModalityType.VIDEO_PIXEL_NOISE,
    ]


# Bottom transformations, applied to all features


def audio_bottom(x, model_hparams, vocab_size):
  """Transform input from data space to model space.

  Args:
    x: A Tensor with shape [batch, ...]
    model_hparams: HParams, model hyperparmeters.
    vocab_size: int, vocabulary size.

  Returns:
    body_input: A Tensor with shape [batch, ?, ?,
      model_hparams.hidden_size].
  """
  del vocab_size  # unused arg
  inputs = x
  with tf.compat.v1.variable_scope("audio_modality"):
    # TODO(aidangomez): Will need to sort out a better audio pipeline
    def xnet_resblock(x, filters, res_relu, name):
      """Xception block."""
      with tf.compat.v1.variable_scope(name):
        # Typically audio samples are >100k samples in length and have a width
        # of 2 or 4. Mono audio has a single channel while stereo has 2.
        y = common_layers.separable_conv_block(
            x,
            filters, [((1, 1), (3, 3)), ((1, 1), (3, 3))],
            first_relu=True,
            padding="SAME",
            force2d=True,
            name="sep_conv_block")
        y = common_layers.pool(y, (3, 3), "MAX", "SAME", strides=(2, 2))
        return y + common_layers.conv_block(
            x,
            filters, [((1, 1), (1, 1))],
            padding="SAME",
            strides=(2, 2),
            first_relu=res_relu,
            force2d=True,
            name="res_conv0")

    x = tf.cast(inputs, dtype=tf.float32) / 255.
    x.set_shape([None, None, None, 1])
    for i in range(model_hparams.audio_compression):
      x = xnet_resblock(x, 2**(i + 1), True, "compress_block_%d" % i)
    return xnet_resblock(x,
                         model_hparams.hidden_size,
                         False,
                         "compress_block_final")


def audio_spectral_bottom(x, model_hparams, vocab_size):
  """Transform input from data space to model space.

  Args:
    x: A Tensor with shape [batch, ...]
    model_hparams: HParams, model hyperparmeters.
    vocab_size: int, vocabulary size.

  Returns:
    body_input: A Tensor with shape [batch, ?, ?,
      model_hparams.hidden_size].
  """
  del vocab_size  # unused arg
  inputs = x
  with tf.compat.v1.variable_scope("audio_spectral_modality"):
    # TODO(aidangomez): Will need to sort out a better audio pipeline
    def xnet_resblock(x, filters, res_relu, name):
      """Xception-like block."""
      with tf.compat.v1.variable_scope(name):
        # We only stride along the length dimension to preserve the spectral
        # bins (which are tiny in dimensionality relative to length)
        y = common_layers.separable_conv_block(
            x,
            filters, [((1, 1), (3, 3)), ((1, 1), (3, 3))],
            first_relu=True,
            padding="SAME",
            force2d=True,
            name="sep_conv_block")
        y = common_layers.pool(y, (3, 3), "MAX", "SAME", strides=(2, 1))
        return y + common_layers.conv_block(
            x,
            filters, [((1, 1), (1, 1))],
            padding="SAME",
            strides=(2, 1),
            first_relu=res_relu,
            force2d=True,
            name="res_conv0")

    # Bitcast back from int32
    x = tf.bitcast(inputs, tf.float32)
    x.set_shape([None, None, None, 1])
    for i in range(model_hparams.audio_compression):
      x = xnet_resblock(x, 2 ** (i + 1), True, "compress_block_%d" % i)
    return xnet_resblock(x,
                         model_hparams.hidden_size,
                         False,
                         "compress_block_final")


def class_label_bottom(x, model_hparams, vocab_size):
  with tf.compat.v1.variable_scope("class_label_modality_%d_%d" % (
      vocab_size, model_hparams.hidden_size)):
    multiplier = 1.0
    if model_hparams.multiply_embedding_mode == "sqrt_depth":
      multiplier = model_hparams.hidden_size ** 0.5
    return common_layers.embedding(x,
                                   vocab_size,
                                   model_hparams.hidden_size,
                                   multiplier=multiplier)


def class_label_targets_bottom(x, model_hparams, vocab_size):
  with tf.compat.v1.variable_scope("class_label_modality_%d_%d" % (
      vocab_size, model_hparams.hidden_size)):
    return tf.zeros([common_layers.shape_list(x)[0],
                     1,
                     1,
                     model_hparams.hidden_size])


def identity_bottom(x, model_hparams, vocab_size):
  del model_hparams, vocab_size  # unused arg
  return tf.cast(x, dtype=tf.float32)


def image_bottom(x, model_hparams, vocab_size):
  del model_hparams, vocab_size  # unused arg
  with tf.compat.v1.variable_scope("image_modality"):
    if not tf.executing_eagerly():
      tf.compat.v1.summary.image(
          "inputs", common_layers.tpu_safe_image_summary(x), max_outputs=2)
    return tf.cast(x, dtype=tf.float32)


def image_targets_bottom(x, model_hparams, vocab_size):
  """Bottom transformation for target images."""
  pixel_embedding_size = 64
  inputs = x
  with tf.compat.v1.variable_scope("image_modality"):
    if not tf.executing_eagerly():
      tf.compat.v1.summary.image(
          "targets_bottom",
          common_layers.tpu_safe_image_summary(inputs),
          max_outputs=1)
    inputs_shape = common_layers.shape_list(inputs)
    if len(inputs_shape) != 4:
      raise ValueError("Assuming images given as int tensors in the format "
                       "[batch, height, width, channels] (256 values).")
    # We embed each of 256=vocab_size possible pixel values.
    embedding_var = tf.compat.v1.get_variable(
        "pixel_embedding",
        [vocab_size, pixel_embedding_size])
    hot_inputs = tf.one_hot(tf.cast(inputs, dtype=tf.int32), vocab_size)
    hot_inputs = tf.reshape(hot_inputs, [-1, vocab_size])
    embedded = tf.matmul(hot_inputs, embedding_var)
    # Let's now merge all channels that were embedded into a single vector.
    merged_size = pixel_embedding_size * inputs_shape[3]
    embedded = tf.reshape(embedded, inputs_shape[:3] + [merged_size])
    merged = tf.compat.v1.layers.dense(
        embedded,
        model_hparams.hidden_size,
        name="merge_pixel_embedded_channels")
    return merged


def _image_channel_compress_bottom(inputs, model_hparams, name="bottom"):
  """Compresses channel-wise input pixels into whole pixel representions.

  Perform conversion of RGB pixel values to a real number in the range -1 to
  1. This combines pixel channels to form a representation of shape
  [img_len, img_len].

  Args:
    inputs: Tensor representing RGB pixel intensities as integers, of shape
      [batch, img_len, img_len, channels].
    model_hparams: HParams, model hyperparmeters.
    name: string, scope.

  Returns:
    body_input: Tensor of shape
      [batch, img_len, img_len, model_hparams.hidden_size].
  """
  num_channels = 3
  with tf.compat.v1.variable_scope(name):
    inputs = tf.cast(inputs, dtype=tf.float32)
    hp = model_hparams
    if hp.mode != tf_estimator.ModeKeys.PREDICT:
      tf.compat.v1.summary.image(
          "inputs",
          common_layers.tpu_safe_image_summary(inputs),
          max_outputs=2)
    inputs = common_layers.convert_rgb_to_symmetric_real(inputs)

    # Reshape inputs to apply convolutions across [img_len, img_len*channels].
    inputs_shape = common_layers.shape_list(inputs)
    inputs = tf.reshape(
        inputs, [-1, inputs_shape[1], inputs_shape[2] * inputs_shape[3], 1])

    # Compress RGB intensities for each pixel using a convolution.
    outputs = tf.compat.v1.layers.conv2d(
        inputs,
        model_hparams.hidden_size,
        kernel_size=(1, num_channels),
        padding="VALID",
        strides=(1, num_channels),
        activation=tf.nn.relu,
        name="conv_input")
    return outputs


def image_channel_compress_bottom(x, model_hparams, vocab_size):
  del vocab_size  # unused arg
  return _image_channel_compress_bottom(x, model_hparams, "input_bottom")


def image_channel_compress_targets_bottom(x, model_hparams, vocab_size):
  del vocab_size  # unused arg
  return _image_channel_compress_bottom(x, model_hparams, "output_bottom")


def image_channel_embeddings_bottom(x, model_hparams, vocab_size):
  """Bottom transformation for image targets."""
  del vocab_size  # unused arg
  inputs = tf.cast(x, dtype=tf.int32)
  io_depth = model_hparams.num_channels
  tshape = common_layers.shape_list(inputs)
  hidden_size = model_hparams.hidden_size
  target_embeddings = cia.get_channel_embeddings(
      io_depth, inputs, hidden_size, "input_bottom")
  return tf.reshape(target_embeddings,
                    [tshape[0], tshape[1], tshape[2] * io_depth, hidden_size])


def make_targets_bottom(bottom):
  def targets_bottom(x, model_hparams, vocab_size):
    with tf.compat.v1.variable_scope("targets_bottom"):
      return bottom(x, model_hparams, vocab_size)

  return targets_bottom


def real_bottom(x, model_hparams, vocab_size):
  del vocab_size  # unused arg
  with tf.compat.v1.variable_scope("real"):
    return tf.compat.v1.layers.dense(
        tf.cast(x, dtype=tf.float32), model_hparams.hidden_size, name="bottom")


def speech_recognition_bottom(x, model_hparams, vocab_size):
  """Use batchnorm instead of CMVN and shorten the stft with strided convs.

  Args:
    x: float32 tensor with shape [batch_size, len, 1, freqs * channels]
    model_hparams: HParams, model hyperparmeters.
    vocab_size: int, vocabulary size.

  Returns:
    float32 tensor with shape [batch_size, shorter_len, 1, hidden_size]
  """
  del vocab_size  # unused arg
  inputs = x
  p = model_hparams

  num_mel_bins = p.audio_num_mel_bins
  num_channels = 3 if p.audio_add_delta_deltas else 1

  with tf.compat.v1.variable_scope("speech_recognition_modality"):
    if p.audio_preproc_in_bottom:
      # Compute filterbanks
      with tf.compat.v1.variable_scope("fbanks"):
        waveforms = tf.squeeze(inputs, [2, 3])
        mel_fbanks = common_audio.compute_mel_filterbank_features(
            waveforms,
            sample_rate=p.audio_sample_rate,
            dither=p.audio_dither,
            preemphasis=p.audio_preemphasis,
            frame_length=p.audio_frame_length,
            frame_step=p.audio_frame_step,
            lower_edge_hertz=p.audio_lower_edge_hertz,
            upper_edge_hertz=p.audio_upper_edge_hertz,
            num_mel_bins=p.audio_num_mel_bins,
            apply_mask=True)
        if p.audio_add_delta_deltas:
          mel_fbanks = common_audio.add_delta_deltas(mel_fbanks)
        x = tf.reshape(mel_fbanks,
                       common_layers.shape_list(mel_fbanks)[:2] +
                       [num_mel_bins, num_channels])

        nonpadding_mask = 1. - common_attention.embedding_to_padding(x)
        num_of_nonpadding_elements = tf.reduce_sum(
            nonpadding_mask) * num_mel_bins * num_channels

        # This replaces CMVN estimation on data
        var_epsilon = 1e-09
        mean = tf.reduce_sum(
            x, axis=[1], keepdims=True) / num_of_nonpadding_elements
        variance = (num_of_nonpadding_elements * mean ** 2. -
                    2. * mean * tf.reduce_sum(x, axis=[1], keepdims=True) +
                    tf.reduce_sum(x ** 2, axis=[1], keepdims=True)
                    ) / num_of_nonpadding_elements
        x = (x - mean) * tf.math.rsqrt(variance + var_epsilon) * tf.expand_dims(
            nonpadding_mask, -1)
    else:
      x = inputs

    # The convention is that the models are flattened along the spatial,
    # dimensions, thus the speech preprocessor treats frequencies and
    # channels as image colors (last axis)
    x.set_shape([None, None, num_mel_bins, num_channels])

    # TODO(chorowski): how to specify bottom's hparams and avoid hardcoding?
    x = tf.pad(x, [[0, 0], [0, 8], [0, 0], [0, 0]])
    for _ in range(2):
      x = tf.compat.v1.layers.conv2d(
          x, 128, (3, 3), (2, 2), use_bias=False)
      x = common_layers.layer_norm(x)
      x = tf.nn.relu(x)

    xshape = common_layers.shape_list(x)
    # apply a conv that will remove all frequencies and at the same time
    # project the output into desired hidden_size
    x = tf.pad(x, [[0, 0], [0, 2], [0, 0], [0, 0]])
    x = tf.compat.v1.layers.conv2d(x, p.hidden_size, (3, xshape[2]), use_bias=False)

    assert common_layers.shape_list(x)[2] == 1
    x = common_layers.layer_norm(x)
    x = tf.nn.relu(x)
  return x


def get_weights(model_hparams, vocab_size, hidden_dim=None):
  """Create or get concatenated embedding or softmax variable.

  Args:
    model_hparams: HParams, model hyperparmeters.
    vocab_size: int, vocabulary size.
    hidden_dim: dim of the variable. Defaults to _model_hparams' hidden_size

  Returns:
     a list of num_shards Tensors.
  """
  if hidden_dim is None:
    hidden_dim = model_hparams.hidden_size
  num_shards = model_hparams.symbol_modality_num_shards
  shards = []
  for i in range(num_shards):
    shard_size = (vocab_size // num_shards) + (
        1 if i < vocab_size % num_shards else 0)
    var_name = "weights_%d" % i
    shards.append(
        tf.compat.v1.get_variable(
            var_name, [shard_size, hidden_dim],
            initializer=tf.compat.v1.random_normal_initializer(0.0, hidden_dim ** -0.5)))
  if num_shards == 1:
    ret = shards[0]
  else:
    ret = tf.concat(shards, 0)
  # Convert ret to tensor.
  if not tf.executing_eagerly():
    ret = common_layers.convert_gradient_to_tensor(ret)
  return ret


def _symbol_bottom_simple(x, model_hparams, vocab_size, name, reuse):
  """Bottom transformation for symbols."""
  with tf.compat.v1.variable_scope(name, reuse=reuse):
    # Ensure the inputs are 3-D
    if len(x.get_shape()) == 4:
      x = tf.squeeze(x, axis=3)
    while len(x.get_shape()) < 3:
      x = tf.expand_dims(x, axis=-1)

    var = get_weights(model_hparams, vocab_size)
    x = common_layers.dropout_no_scaling(
        x, 1.0 - model_hparams.symbol_dropout)
    ret = common_layers.gather(var, x)
    if model_hparams.multiply_embedding_mode == "sqrt_depth":
      ret *= model_hparams.hidden_size ** 0.5
    ret *= tf.expand_dims(
        common_layers.cast_like(tf.not_equal(x, 0), ret), -1)
    return ret


def symbol_bottom(x, model_hparams, vocab_size):
  if (model_hparams.shared_embedding_and_softmax_weights or
      model_hparams.get("shared_embedding")):
    return _symbol_bottom_simple(
        x, model_hparams, vocab_size, "shared", reuse=None)
  return _symbol_bottom_simple(
      x, model_hparams, vocab_size, "input_emb", reuse=None)


def symbol_targets_bottom(x, model_hparams, vocab_size):
  """Bottom transformation for target symbols."""
  if (model_hparams.shared_embedding_and_softmax_weights or
      model_hparams.get("shared_embedding")):
    try:
      return _symbol_bottom_simple(
          x, model_hparams, vocab_size, "shared", reuse=True)
    except ValueError:
      # perhaps there were no inputs, and this is a new variable.
      return _symbol_bottom_simple(
          x, model_hparams, vocab_size, "shared", reuse=None)
  else:
    return _symbol_bottom_simple(
        x, model_hparams, vocab_size, "target_emb", reuse=None)


def symbol_one_hot_bottom(x, model_hparams, vocab_size):
  del model_hparams  # unused arg
  return tf.one_hot(x, vocab_size)


def video_bottom(x, model_hparams, vocab_size):
  del model_hparams, vocab_size  # unused arg
  common_video.gif_summary("inputs", x, max_outputs=1)
  x = common_layers.standardize_images(x)
  return x


def video_targets_bottom(x, model_hparams, vocab_size):
  del model_hparams, vocab_size  # unused arg
  common_video.gif_summary("targets", x, max_outputs=1)
  x = common_layers.standardize_images(x)
  return x


def video_bitwise_bottom(x, model_hparams, vocab_size):
  """Bottom transformation for embedding video bitwise."""
  pixel_embedding_size = 64
  inputs = x
  with tf.compat.v1.variable_scope("video_modality_bitwise", reuse=tf.compat.v1.AUTO_REUSE):
    common_layers.summarize_video(inputs, "bottom")
    # Embed bitwise.
    assert vocab_size == 256
    embedded = discretization.int_to_bit_embed(inputs, 8,
                                               pixel_embedding_size)
    # Project.
    return tf.compat.v1.layers.dense(
        embedded,
        model_hparams.hidden_size,
        name="merge_pixel_embedded_frames")


def video_bitwise_targets_bottom(x, model_hparams, vocab_size):
  """Bottom transformation for embedding target video bitwise."""
  pixel_embedding_size = 64
  inputs = x
  with tf.compat.v1.variable_scope("video_modality_bitwise", reuse=tf.compat.v1.AUTO_REUSE):
    common_layers.summarize_video(inputs, "targets_bottom")
    # Embed bitwise.
    assert vocab_size == 256
    embedded = discretization.int_to_bit_embed(inputs, 8,
                                               pixel_embedding_size)
    # Transpose and project.
    transposed = common_layers.time_to_channels(embedded)
    return tf.compat.v1.layers.dense(
        transposed,
        model_hparams.hidden_size,
        name="merge_pixel_embedded_frames")


def video_identity_bottom(x, model_hparams, vocab_size):
  del model_hparams, vocab_size  # unused arg
  common_video.gif_summary("inputs", x, max_outputs=1)
  return x


def video_identity_targets_bottom(x, model_hparams, vocab_size):
  del model_hparams, vocab_size  # unused arg
  common_video.gif_summary("targets", x, max_outputs=1)
  return x


def video_pixel_noise_bottom(x, model_hparams, vocab_size):
  """Bottom transformation for video."""
  input_noise = getattr(model_hparams, "video_modality_input_noise", 0.25)
  inputs = x
  if model_hparams.mode == tf_estimator.ModeKeys.TRAIN:
    background = tfp.stats.percentile(inputs, 50., axis=[0, 1, 2, 3])
    input_shape = common_layers.shape_list(inputs)
    input_size = tf.reduce_prod(input_shape[:-1])
    input_mask = tf.random.categorical(
        tf.math.log([[input_noise, 1. - input_noise]]), input_size)
    input_mask = tf.reshape(tf.cast(input_mask, tf.int32),
                            input_shape[:-1] + [1])
    inputs = inputs * input_mask + background * (1 - input_mask)
  return video_bottom(inputs, model_hparams, vocab_size)


def convert_rgb_to_real(prediction, targets):
  """Convert prediction and target from rgb to real."""
  prediction = tf.squeeze(prediction, axis=-1)
  prediction = common_layers.convert_rgb_to_real(prediction)
  targets = common_layers.convert_rgb_to_real(targets)
  return prediction, targets


def video_raw_bottom(x, model_hparams, vocab_size):
  del model_hparams, vocab_size  # unused arg
  common_video.gif_summary("inputs", x)
  return common_layers.convert_rgb_to_real(x)


def video_raw_targets_bottom(x, model_hparams, vocab_size):
  del model_hparams, vocab_size  # unused arg
  common_video.gif_summary("targets_bottom", x)
  return common_layers.convert_rgb_to_real(x)


# Loss transformations, applied to target features


def ctc_symbol_loss(top_out, targets, model_hparams, vocab_size, weight_fn):
  """Compute the CTC loss."""
  del model_hparams, vocab_size  # unused arg
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
    weights = weight_fn(targets)
    return tf.reduce_sum(xent), tf.reduce_sum(weights)


def generic_loss(top_out, targets, model_hparams, vocab_size, weights_fn):
  """Compute loss numerator and denominator for one shard of output."""
  del vocab_size  # unused arg
  logits = top_out
  logits = common_attention.maybe_upcast(logits, hparams=model_hparams)
  cutoff = getattr(model_hparams, "video_modality_loss_cutoff", 0.0)
  return common_layers.padded_cross_entropy(
      logits,
      targets,
      model_hparams.label_smoothing,
      cutoff=cutoff,
      weights_fn=weights_fn)


def generic_l2_loss(body_output,
                    targets,
                    model_hparams,
                    vocab_size,
                    weights_fn):
  del model_hparams, vocab_size, weights_fn  # unused arg
  loss = tf.math.squared_difference(body_output, tf.cast(targets, dtype=tf.float32))
  return tf.reduce_mean(loss), tf.constant(1.0)


def multi_label_loss(top_out, targets, model_hparams, vocab_size, weights_fn):
  """Average loss over the labels."""
  del vocab_size  # unused arg
  logits = top_out
  num_labels = tf.shape(targets)[1]
  logits = tf.tile(logits, [1, num_labels, 1, 1, 1])

  xent, weights = common_layers.padded_cross_entropy(
      logits,
      targets,
      model_hparams.label_smoothing,
      weights_fn=weights_fn,
      reduce_sum=False,
  )
  xent = tf.squeeze(xent, [2, 3])
  weights = tf.squeeze(weights, [2, 3])
  # average loss over all labels
  loss = tf.reduce_sum(xent, axis=1)
  weights = tf.reduce_sum(weights, axis=1)
  loss /= (weights + 1e-8)
  weights = tf.cast(tf.greater(weights, 0.), dtype=tf.float32)

  return tf.reduce_sum(loss * weights), tf.reduce_sum(weights)


def one_hot_class_label_loss(top_out,
                             targets,
                             model_hparams,
                             vocab_size,
                             weights_fn):
  """Apply softmax cross-entropy between outputs and targets.

  Args:
    top_out: logits Tensor with shape [batch, ?, ?, num_classes]
    targets: one-hot encoding Tensor with shape [batch, ?, ?, num_classes]
    model_hparams: HParams, model hyperparmeters.
    vocab_size: int, vocabulary size.
    weights_fn:

  Returns:
    loss_scale (cross-entropy), loss_denom
  """
  del model_hparams, vocab_size  # unused arg
  loss_scale = tf.compat.v1.losses.softmax_cross_entropy(
      onehot_labels=targets, logits=top_out)
  weights = weights_fn(targets)
  loss_denom = tf.reduce_sum(weights)
  return loss_scale, loss_denom


def real_l2_loss(top_out, targets, model_hparams, vocab_size, weights_fn):
  del model_hparams, vocab_size  # unused arg
  predictions = top_out
  if (len(common_layers.shape_list(top_out)) != len(
      common_layers.shape_list(targets))):
    predictions = tf.squeeze(top_out, axis=[-1])
  with tf.compat.v1.name_scope("l2"):
    weights = weights_fn(targets)
    l2 = tf.pow(predictions - targets, 2)
    return tf.reduce_sum(l2 * weights), tf.reduce_sum(weights)


def real_log_poisson_loss(top_out,
                          targets,
                          model_hparams,
                          vocab_size,
                          weights_fn):
  """Poisson loss for real."""
  del model_hparams, vocab_size  # unused arg
  predictions = top_out
  if (len(common_layers.shape_list(top_out)) != len(
      common_layers.shape_list(targets))):
    predictions = tf.squeeze(top_out, axis=[-1])
  with tf.compat.v1.name_scope("log_possion"):
    weights = weights_fn(targets)
    lp_loss = tf.nn.log_poisson_loss(targets, predictions)
    return tf.reduce_sum(lp_loss * weights), tf.reduce_sum(weights)


def sigmoid_class_label_loss(top_out,
                             targets,
                             model_hparams,
                             vocab_size,
                             weights_fn):
  """Loss for class label."""
  # Expect inputs of size [batch-size, timesteps, 1, num-classes], where the
  # last dimension of num-classes represents logits for binary labels
  del model_hparams, vocab_size  # unused arg
  loss_scale = tf.compat.v1.losses.sigmoid_cross_entropy(
      multi_class_labels=targets, logits=top_out)
  weights = weights_fn(targets)
  loss_denom = tf.reduce_sum(weights)
  return loss_scale, loss_denom


def sigmoid_max_pooling_class_label_loss(top_out,
                                         targets,
                                         model_hparams,
                                         vocab_size,
                                         weights_fn):
  """Loss for class label."""
  # Expect inputs of size [batch-size, 1, 1, num-classes], where the
  # last dimension of num-classes represents logits for binary labels
  del model_hparams, vocab_size  # unused arg
  loss_scale = tf.compat.v1.losses.sigmoid_cross_entropy(
      multi_class_labels=targets, logits=top_out)
  weights = weights_fn(targets)
  loss_denom = tf.reduce_sum(weights)
  return loss_scale, loss_denom


def symbol_one_hot_loss(top_out,
                        targets,
                        model_hparams,
                        vocab_size,
                        weights_fn):
  del model_hparams, weights_fn  # unused arg
  labels = tf.one_hot(targets, vocab_size)
  loss = tf.nn.softmax_cross_entropy_with_logits(
      logits=top_out, labels=tf.stop_gradient(labels))
  return tf.reduce_mean(loss), tf.constant(1.0)


def video_loss(top_out, targets, model_hparams, vocab_size, weights_fn):
  """Compute loss numerator and denominator for one shard of output."""
  del vocab_size  # unused arg
  logits = top_out
  logits = tf.reshape(logits, [-1] + common_layers.shape_list(logits)[2:])
  targets = tf.reshape(targets, [-1] + common_layers.shape_list(targets)[2:])
  cutoff = getattr(model_hparams, "video_modality_loss_cutoff", 0.01)
  return common_layers.padded_cross_entropy(
      logits,
      targets,
      model_hparams.label_smoothing,
      cutoff=cutoff,
      weights_fn=weights_fn)


def video_identity_loss(top_out,
                        targets,
                        model_hparams,
                        vocab_size,
                        weights_fn):
  """Compute loss numerator and denominator for one shard of output."""
  del vocab_size  # unused arg
  # TODO(nikip): Try L2 loss
  logits = top_out
  logits = tf.reshape(logits, [-1] + common_layers.shape_list(logits)[2:])
  targets = tf.reshape(targets, [-1] + common_layers.shape_list(targets)[2:])
  cutoff = getattr(model_hparams, "video_modality_loss_cutoff", 0.01)
  return common_layers.padded_cross_entropy(
      logits,
      targets,
      model_hparams.label_smoothing,
      cutoff=cutoff,
      weights_fn=weights_fn)


def video_l1_internal_loss(logits, targets, model_hparams):
  cutoff = getattr(model_hparams, "video_modality_loss_cutoff", 0.2)
  return tf.nn.relu(tf.abs(logits - targets) - cutoff)


def video_l1_loss(top_out, targets, model_hparams, vocab_size, weights_fn):
  """Compute loss numerator and denominator for one shard of output."""
  del vocab_size  # unused arg
  logits = top_out
  logits = tf.reshape(logits, [-1] + common_layers.shape_list(logits)[2:-1])
  targets = tf.reshape(targets, [-1] + common_layers.shape_list(targets)[2:])
  weights = weights_fn(targets)
  # Shift targets by 0.5 so later just casting to int gives the prediction.
  # So for int targets, say 0 and 7, we actually train to predict 0.5 and 7.5.
  # Later (in merics or infer) this is cast to int anyway. Also, we have no
  # loss beyond cutoff = 0.2 as these are already correct predictions.
  targets = tf.cast(targets, dtype=tf.float32) + 0.5
  loss = video_l1_internal_loss(logits, targets, model_hparams)
  return tf.reduce_sum(loss * weights), tf.reduce_sum(weights)


def video_l2_internal_loss(logits, targets, model_hparams):
  cutoff = getattr(model_hparams, "video_modality_loss_cutoff", 0.2)
  return tf.nn.relu(
      tf.math.squared_difference(logits, targets) - cutoff * cutoff)


def video_l2_loss(top_out, targets, model_hparams, vocab_size, weights_fn):
  """Compute loss numerator and denominator for one shard of output."""
  del vocab_size  # unused arg
  logits = top_out
  logits = tf.reshape(logits, [-1] + common_layers.shape_list(logits)[2:-1])
  targets = tf.reshape(targets, [-1] + common_layers.shape_list(targets)[2:])
  weights = weights_fn(targets)
  # Shift targets by 0.5 so later just casting to int gives the prediction.
  # So for int targets, say 0 and 7, we actually train to predict 0.5 and 7.5.
  # Later (in merics or infer) this is cast to int anyway. Also, we have no
  # loss beyond cutoff = 0.2 as these are already correct predictions.
  targets = tf.cast(targets, dtype=tf.float32) + 0.5
  loss = video_l2_internal_loss(logits, targets, model_hparams)
  return tf.reduce_sum(loss * weights), tf.reduce_sum(weights)


def video_l2_raw_loss(top_out, targets, model_hparams, vocab_size, weights_fn):
  del model_hparams, vocab_size, weights_fn  # unused arg
  prediction, groundtruth = convert_rgb_to_real(top_out, targets)
  loss = tf.compat.v1.losses.mean_squared_error(prediction, groundtruth)
  return loss, tf.constant(1.0)


def video_l1_raw_loss(top_out, targets, model_hparams, vocab_size, weights_fn):
  del model_hparams, vocab_size, weights_fn  # unused arg
  prediction, groundtruth = convert_rgb_to_real(top_out, targets)
  loss = tf.compat.v1.losses.absolute_difference(prediction, groundtruth)
  return loss, tf.constant(1.0)


# Top transformations, applied to target features


def is_pointwise(func):
  """Decorator for whether the function is pointwise.

  An example of a pointwise function is a linear layer followed by
  a softmax. Given a tensor [batch, length, height, depth] it operates
  only on the last axis, on every point in [batch, length, height] fully
  independently. In contrast, a classifier that first averages over length
  and height is not pointwise, as it depends on the whole field. It is useful
  to know if top functions are pointwise to speed up decoding in certain models.

  Args:
    func: Function to decorate.

  Returns:
    Original function with an attribute pointwise set to True.
  """
  func.pointwise = True
  return func


def class_label_top(body_output, targets, model_hparams, vocab_size):
  """Transform inputs from model space to target space.

  Average over inner dims and a linear layer to logits.

  Args:
    body_output: A Tensor with shape [batch, ?, ?, body_output_size].
    targets:
    model_hparams: HParams, model hyperparmeters.
    vocab_size: int, vocabulary size.

  Returns:
    a Tensors, each with shape [batch_size, 1, 1, 1, vocab_size]
  """
  del targets  # unused arg
  with tf.compat.v1.variable_scope("class_label_modality_%d_%d" % (
      vocab_size, model_hparams.hidden_size)):
    x = body_output
    x = tf.reduce_mean(x, axis=[1, 2], keepdims=True)
    res = tf.compat.v1.layers.dense(x, vocab_size)
    return tf.expand_dims(res, 3)


def identity_top(body_output, targets, model_hparams, vocab_size):
  del targets, model_hparams, vocab_size  # unused arg
  return body_output


def image_top(body_output, targets, model_hparams, vocab_size):
  """Top transformation for images."""
  del targets  # unused arg
  # TODO(lukaszkaiser): is this a universal enough way to get channels?
  num_channels = model_hparams.problem.num_channels
  with tf.compat.v1.variable_scope("rgb_softmax"):
    body_output_shape = common_layers.shape_list(body_output)
    reshape_shape = body_output_shape[:3]
    reshape_shape.extend([num_channels, vocab_size])
    res = tf.compat.v1.layers.dense(body_output, vocab_size * num_channels)
    res = tf.reshape(res, reshape_shape)
    if not tf.compat.v1.get_variable_scope().reuse:
      res_argmax = tf.argmax(res, axis=-1)
      tf.compat.v1.summary.image(
          "result",
          common_layers.tpu_safe_image_summary(res_argmax),
          max_outputs=1)
    return res


def image_channel_compress_top(body_output, targets, model_hparams, vocab_size):
  """Transforms body output to return logits.

  Args:
    body_output: Tensor of shape [batch, img_len, img_len, depth].
    targets:
    model_hparams: HParams, model hyperparmeters.
    vocab_size: int, vocabulary size.

  Returns:
    Tensor of shape [batch, img_len, img_len, channels, vocab_size].
  """
  del targets  # unused arg
  with tf.compat.v1.variable_scope("image_channel_compress_modality"):
    hidden_size = model_hparams.hidden_size
    img_len = model_hparams.img_len
    channels = 3  # RGB
    batch = common_layers.shape_list(body_output)[0]
    x = tf.compat.v1.layers.conv2d(
        body_output,
        hidden_size * channels,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding="VALID",
        activation=tf.nn.relu,
        name="decompress_conv")
    x = tf.reshape(x, [batch, img_len, img_len * channels, hidden_size])
    x = common_layers.layer_preprocess(x, model_hparams)
    x = tf.compat.v1.layers.dense(x,
                        vocab_size,
                        use_bias=True,
                        activation=None,
                        name="output_conv")
    x = tf.reshape(
        x, [batch, img_len, img_len, channels, vocab_size])
    return x


def image_channel_embeddings_top(body_output,
                                 targets,
                                 model_hparams,
                                 vocab_size):
  """Top transformation for images."""
  del targets  # unused arg
  with tf.compat.v1.variable_scope("image_channel_embeddings_bottom"):
    img_len = model_hparams.img_len
    channels = model_hparams.num_channels
    x = tf.compat.v1.layers.dense(
        body_output, 256, use_bias=True, activation=None, name="output_conv")
    x = tf.reshape(x,
                   [-1, img_len, img_len, channels, vocab_size])
    return x


@is_pointwise
def real_top(body_output, targets, model_hparams, vocab_size):
  del targets, model_hparams  # unused arg
  with tf.compat.v1.variable_scope("real"):
    return tf.compat.v1.layers.dense(body_output, vocab_size, name="top")


def sigmoid_max_pooling_class_label_top(body_output,
                                        targets,
                                        model_hparams,
                                        vocab_size):
  """Transform inputs from model space to target space.

  Average over inner dims and a linear layer to logits.

  Args:
    body_output: A Tensor with shape [batch, timesteps, 1, body_output_size].
    targets:
    model_hparams: HParams, model hyperparmeters.
    vocab_size: int, vocabulary size.

  Returns:
    a Tensors, each with shape [batch_size, 1, 1, vocab_size]
  """
  del targets  # unused arg
  with tf.compat.v1.variable_scope(
      "sigmoid_max_pooling_class_symbol_modality_%d_%d" % (
          vocab_size, model_hparams.hidden_size)):
    x = body_output
    x = tf.reduce_max(x, axis=1, keepdims=True)
    return tf.compat.v1.layers.dense(x, vocab_size)


def softmax_average_pooling_class_label_top(body_output,
                                            targets,
                                            model_hparams,
                                            vocab_size):
  """Loss for class label."""
  del targets  # unused arg
  with tf.compat.v1.variable_scope(
      "softmax_average_pooling_onehot_class_label_modality_%d_%d" % (
          vocab_size, model_hparams.hidden_size)):
    x = body_output
    x = tf.reduce_mean(x, axis=1, keepdims=True)
    return tf.compat.v1.layers.dense(x, vocab_size)


def softmax_last_timestep_class_label_top(body_output,
                                          targets,
                                          model_hparams,
                                          vocab_size):
  """Loss for class label."""
  del targets  # unused arg
  with tf.compat.v1.variable_scope(
      "softmax_last_timestep_onehot_class_label_modality_%d_%d" % (
          vocab_size, model_hparams.hidden_size)):
    x = body_output
    x = tf.expand_dims(x[:, -1], 1)  # Pick the last timestep
    return tf.compat.v1.layers.dense(x, vocab_size)


def softmax_max_pooling_class_label_top(body_output,
                                        targets,
                                        model_hparams,
                                        vocab_size):
  """Loss for class label."""
  del targets  # unused arg
  with tf.compat.v1.variable_scope(
      "softmax_max_pooling_onehot_class_label_modality_%d_%d" % (
          vocab_size, model_hparams.hidden_size)):
    x = body_output
    x = tf.reduce_max(x, axis=1, keepdims=True)
    return tf.compat.v1.layers.dense(x, vocab_size)


@is_pointwise
def symbol_top(body_output, targets, model_hparams, vocab_size):
  """Generate logits.

  Args:
    body_output: A Tensor with shape
      [batch, p0, p1, model_hparams.hidden_size].
    targets: Unused.
    model_hparams: HParams, model hyperparmeters.
    vocab_size: int, vocabulary size.

  Returns:
    logits: A Tensor with shape  [batch, p0, p1, ?, vocab_size].
  """
  del targets  # unused arg
  if model_hparams.shared_embedding_and_softmax_weights:
    scope_name = "shared"
    reuse = tf.compat.v1.AUTO_REUSE
  else:
    scope_name = "softmax"
    reuse = False
  with tf.compat.v1.variable_scope(scope_name, reuse=reuse):
    body_output_shape = common_layers.shape_list(body_output)
    var = get_weights(model_hparams, vocab_size, body_output_shape[-1])
    if (model_hparams.factored_logits and
        model_hparams.mode == tf_estimator.ModeKeys.TRAIN):
      # insert channels dimension
      body_output = tf.expand_dims(body_output, 3)
      return common_layers.FactoredTensor(body_output, var)
    else:
      body_output = tf.reshape(body_output, [-1, body_output_shape[-1]])
      logits = tf.matmul(body_output, var, transpose_b=True)
      return tf.reshape(logits,
                        body_output_shape[:-1] + [1, vocab_size])


@is_pointwise
def symbol_one_hot_top(body_output, targets, model_hparams, vocab_size):
  del targets, model_hparams, vocab_size  # unused arg
  return body_output


def video_top(body_output, targets, model_hparams, vocab_size):
  """Top transformation for video."""
  del targets  # unused arg
  num_channels = model_hparams.problem.num_channels
  shape = common_layers.shape_list(body_output)
  reshape_shape = shape[:-1] + [num_channels, vocab_size]
  res = tf.reshape(body_output, reshape_shape)
  # Calculate argmax so as to have a summary with the produced images.
  x = tf.argmax(tf.reshape(res, [-1, vocab_size]), axis=-1)
  x = tf.reshape(x, shape[:-1] + [num_channels])
  common_video.gif_summary("results", x, max_outputs=1)
  return res


def video_l1_top(body_output, targets, model_hparams, vocab_size):
  """Top transformation for video."""
  del targets, vocab_size  # unused arg
  num_channels = model_hparams.problem.num_channels
  num_frames = model_hparams.video_num_target_frames
  with tf.compat.v1.variable_scope("rgb"):
    body_output_shape = common_layers.shape_list(body_output)
    res = tf.compat.v1.layers.dense(body_output, num_channels * num_frames, name="cast")
    res = tf.reshape(res, body_output_shape[:3] + [num_channels, num_frames])
    res = tf.transpose(res, [0, 4, 1, 2, 3])  # Move frames next to batch.
    if not tf.compat.v1.get_variable_scope().reuse:
      res_argmax = res[:, -1, :, :, :]
      tf.compat.v1.summary.image(
          "result",
          common_layers.tpu_safe_image_summary(res_argmax),
          max_outputs=1)
    return tf.expand_dims(res, axis=-1)  # Add an axis like in perplexity.


def video_raw_top(body_output, targets, model_hparams, vocab_size):
  del targets, model_hparams, vocab_size  # unused arg
  frames = body_output
  if isinstance(body_output, list):
    frames = tf.stack(body_output, axis=1)
  rgb_frames = common_layers.convert_real_to_rgb(frames)
  common_video.gif_summary("body_output", rgb_frames)
  return tf.expand_dims(rgb_frames, axis=-1)


# Utility functions similar to tf.keras for default transformations


def get_bottom(modality_type, value=None):
  """Gets default bottom transformation; if none available, return value."""
  if modality_type == ModalityType.AUDIO:
    return audio_bottom
  elif modality_type == ModalityType.AUDIO_SPECTRAL:
    return audio_spectral_bottom
  elif modality_type in (ModalityType.CLASS_LABEL,
                         ModalityType.MULTI_LABEL,
                         ModalityType.ONE_HOT_CLASS_LABEL,
                         ModalityType.SIGMOID_CLASS_LABEL,
                         ModalityType.SIGMOID_MAX_POOLING_CLASS_LABEL,
                         ModalityType.SOFTMAX_AVERAGE_POOLING_CLASS_LABEL,
                         ModalityType.SOFTMAX_LAST_TIMESTEP_CLASS_LABEL,
                         ModalityType.SOFTMAX_MAX_POOLING_CLASS_LABEL):
    return class_label_bottom
  elif modality_type in (ModalityType.CTC_SYMBOL,
                         ModalityType.SYMBOL,
                         ModalityType.SYMBOL_WEIGHTS_ALL):
    return symbol_bottom
  elif modality_type in (ModalityType.GENERIC_L2_LOSS,
                         ModalityType.IDENTITY,
                         ModalityType.IDENTITY_SYMBOL,
                         ModalityType.IMAGE_CHANNEL_EMBEDDINGS_BOTTOM):
    return identity_bottom
  elif modality_type == ModalityType.IMAGE:
    return image_bottom
  elif modality_type in (ModalityType.IMAGE_CHANNEL_BOTTOM_IDENTITY,
                         ModalityType.IMAGE_CHANNEL_COMPRESS):
    return image_channel_compress_bottom
  elif modality_type in (ModalityType.REAL,
                         ModalityType.REAL_L2_LOSS,
                         ModalityType.REAL_LOG_POISSON_LOSS):
    return real_bottom
  elif modality_type == ModalityType.SPEECH_RECOGNITION:
    return speech_recognition_bottom
  elif modality_type == ModalityType.SYMBOL_ONE_HOT:
    return symbol_one_hot_bottom
  elif modality_type in (ModalityType.VIDEO,
                         ModalityType.VIDEO_L1,
                         ModalityType.VIDEO_L2):
    return video_bottom
  elif modality_type == ModalityType.VIDEO_BITWISE:
    return video_bitwise_bottom
  elif modality_type == ModalityType.VIDEO_IDENTITY:
    return video_identity_bottom
  elif modality_type in (ModalityType.VIDEO_L1_RAW,
                         ModalityType.VIDEO_L2_RAW):
    return video_raw_bottom
  elif modality_type == ModalityType.VIDEO_PIXEL_NOISE:
    return video_pixel_noise_bottom
  return value


def get_loss(modality_type, value=None):
  """Gets default loss transformation; if none available, return value."""
  if modality_type in (ModalityType.AUDIO,
                       ModalityType.AUDIO_SPECTRAL,
                       ModalityType.CLASS_LABEL,
                       ModalityType.IDENTITY,
                       ModalityType.IDENTITY_SYMBOL,
                       ModalityType.IMAGE,
                       ModalityType.IMAGE_CHANNEL_BOTTOM_IDENTITY,
                       ModalityType.IMAGE_CHANNEL_COMPRESS,
                       ModalityType.IMAGE_CHANNEL_EMBEDDINGS_BOTTOM,
                       ModalityType.REAL,
                       ModalityType.SPEECH_RECOGNITION,
                       ModalityType.SYMBOL,
                       ModalityType.SYMBOL_WEIGHTS_ALL):
    return generic_loss
  elif modality_type == ModalityType.CTC_SYMBOL:
    return ctc_symbol_loss
  elif modality_type == ModalityType.GENERIC_L2_LOSS:
    return generic_l2_loss
  elif modality_type == ModalityType.MULTI_LABEL:
    return multi_label_loss
  elif modality_type in (ModalityType.ONE_HOT_CLASS_LABEL,
                         ModalityType.SOFTMAX_AVERAGE_POOLING_CLASS_LABEL,
                         ModalityType.SOFTMAX_LAST_TIMESTEP_CLASS_LABEL,
                         ModalityType.SOFTMAX_MAX_POOLING_CLASS_LABEL):
    return one_hot_class_label_loss
  elif modality_type == ModalityType.REAL_L2_LOSS:
    return real_l2_loss
  elif modality_type == ModalityType.REAL_LOG_POISSON_LOSS:
    return real_log_poisson_loss
  elif modality_type == ModalityType.SIGMOID_CLASS_LABEL:
    return sigmoid_class_label_loss
  elif modality_type == ModalityType.SIGMOID_MAX_POOLING_CLASS_LABEL:
    return sigmoid_max_pooling_class_label_loss
  elif modality_type == ModalityType.SYMBOL_ONE_HOT:
    return symbol_one_hot_loss
  elif modality_type in (ModalityType.VIDEO,
                         ModalityType.VIDEO_BITWISE,
                         ModalityType.VIDEO_PIXEL_NOISE):
    return video_loss
  elif modality_type == ModalityType.VIDEO_IDENTITY:
    return video_identity_loss
  elif modality_type == ModalityType.VIDEO_L1:
    return video_l1_loss
  elif modality_type == ModalityType.VIDEO_L1_RAW:
    return video_l1_raw_loss
  elif modality_type == ModalityType.VIDEO_L2:
    return video_l2_loss
  elif modality_type == ModalityType.VIDEO_L2_RAW:
    return video_l2_raw_loss
  return value


def get_name(modality_type, value=None):
  """Gets default name for transformations; if none available, return value."""
  # For legacy reasons, modalities vary in their naming scheme. Future plans are
  # to remove any need for get_name. We do not recommend using it.
  if modality_type == ModalityType.AUDIO:
    return lambda model_hparams, vocab_size: "audio_modality"
  elif modality_type == ModalityType.AUDIO_SPECTRAL:
    return lambda model_hparams, vocab_size: "audio_spectral_modality"
  elif modality_type == ModalityType.GENERIC_L2_LOSS:
    return lambda model_hparams, vocab_size: "generic_l2_loss_modality"
  elif modality_type == ModalityType.IDENTITY:
    return lambda model_hparams, vocab_size: "identity_modality"
  elif modality_type == ModalityType.IMAGE:
    return lambda model_hparams, vocab_size: "image_modality"
  elif modality_type == ModalityType.IMAGE_CHANNEL_BOTTOM_IDENTITY:
    return (lambda model_hparams, vocab_size:  # pylint: disable=g-long-lambda
            "image_channel_bottom_identity_modality")
  elif modality_type == ModalityType.IMAGE_CHANNEL_COMPRESS:
    return lambda model_hparams, vocab_size: "image_channel_compress_modality"
  elif modality_type == ModalityType.IMAGE_CHANNEL_EMBEDDINGS_BOTTOM:
    return lambda model_hparams, vocab_size: "image_channel_embeddings_bottom"
  elif modality_type == ModalityType.REAL:
    return lambda model_hparams, vocab_size: "real_modality"
  elif modality_type == ModalityType.REAL_L2_LOSS:
    return lambda model_hparams, vocab_size: "real_l2_loss_modality"
  elif modality_type == ModalityType.REAL_LOG_POISSON_LOSS:
    return lambda model_hparams, vocab_size: "real_log_poisson_loss_modality"
  elif modality_type == ModalityType.SPEECH_RECOGNITION:
    return lambda model_hparams, vocab_size: "speech_recognition_modality"
  elif modality_type == ModalityType.VIDEO:
    return lambda model_hparams, vocab_size: "video_modality"
  elif modality_type == ModalityType.VIDEO_BITWISE:
    return lambda model_hparams, vocab_size: "video_modality_bitwise"
  elif modality_type == ModalityType.VIDEO_IDENTITY:
    return lambda model_hparams, vocab_size: "video_modality_identity"
  elif modality_type == ModalityType.VIDEO_L1:
    return lambda model_hparams, vocab_size: "video_modality_l1"
  elif modality_type == ModalityType.VIDEO_L1_RAW:
    return lambda model_hparams, vocab_size: "video_modality_l1_raw"
  elif modality_type == ModalityType.VIDEO_L2:
    return lambda model_hparams, vocab_size: "video_modality_l2"
  elif modality_type == ModalityType.VIDEO_L2_RAW:
    return lambda model_hparams, vocab_size: "video_modality_l2_raw"
  elif modality_type == ModalityType.VIDEO_PIXEL_NOISE:
    return lambda model_hparams, vocab_size: "video_modality_pixel_noise"
  elif modality_type in (ModalityType.CLASS_LABEL,
                         ModalityType.MULTI_LABEL,
                         ModalityType.ONE_HOT_CLASS_LABEL):
    def name(model_hparams, vocab_size):
      return "class_label_modality_%d_%d" % (vocab_size,
                                             model_hparams.hidden_size)

    return name
  elif modality_type in (ModalityType.CTC_SYMBOL,
                         ModalityType.IDENTITY_SYMBOL,
                         ModalityType.SYMBOL,
                         ModalityType.SYMBOL_WEIGHTS_ALL,
                         ModalityType.SYMBOL_ONE_HOT):
    def name(model_hparams, vocab_size):
      return "symbol_modality_%d_%d" % (vocab_size, model_hparams.hidden_size)

    return name
  elif modality_type == ModalityType.SIGMOID_CLASS_LABEL:
    def name(model_hparams, vocab_size):
      return "sigmoid_class_symbol_modality_%d_%d" % (vocab_size,
                                                      model_hparams.hidden_size)

    return name
  elif modality_type == ModalityType.SIGMOID_MAX_POOLING_CLASS_LABEL:
    def name(model_hparams, vocab_size):
      return "sigmoid_max_pooling_class_symbol_modality_%d_%d" % (
          vocab_size, model_hparams.hidden_size)

    return name
  elif modality_type == ModalityType.SOFTMAX_AVERAGE_POOLING_CLASS_LABEL:
    def name(model_hparams, vocab_size):
      return "softmax_average_pooling_onehot_class_label_modality_%d_%d" % (
          vocab_size, model_hparams.hidden_size)

    return name
  elif modality_type == ModalityType.SOFTMAX_LAST_TIMESTEP_CLASS_LABEL:
    def name(model_hparams, vocab_size):
      return "softmax_last_timestep_onehot_class_label_modality_%d_%d" % (
          vocab_size, model_hparams.hidden_size)

    return name
  elif modality_type == ModalityType.SOFTMAX_MAX_POOLING_CLASS_LABEL:
    def name(model_hparams, vocab_size):
      return "softmax_max_pooling_onehot_class_label_modality_%d_%d" % (
          vocab_size, model_hparams.hidden_size)

    return name
  return value


def get_targets_bottom(modality_type, value=None):
  """Gets default bottom transformation for targets; if none, return value."""
  if modality_type == ModalityType.AUDIO:
    return make_targets_bottom(audio_bottom)
  elif modality_type == ModalityType.AUDIO_SPECTRAL:
    return make_targets_bottom(audio_spectral_bottom)
  elif modality_type in (ModalityType.CLASS_LABEL,
                         ModalityType.MULTI_LABEL,
                         ModalityType.ONE_HOT_CLASS_LABEL,
                         ModalityType.SIGMOID_CLASS_LABEL,
                         ModalityType.SIGMOID_MAX_POOLING_CLASS_LABEL,
                         ModalityType.SOFTMAX_AVERAGE_POOLING_CLASS_LABEL,
                         ModalityType.SOFTMAX_LAST_TIMESTEP_CLASS_LABEL,
                         ModalityType.SOFTMAX_MAX_POOLING_CLASS_LABEL):
    return class_label_targets_bottom
  elif modality_type in (ModalityType.CTC_SYMBOL,
                         ModalityType.SYMBOL,
                         ModalityType.SYMBOL_WEIGHTS_ALL):
    return symbol_targets_bottom
  elif modality_type in (ModalityType.GENERIC_L2_LOSS,
                         ModalityType.IDENTITY_SYMBOL):
    return identity_bottom
  elif modality_type == ModalityType.IDENTITY:
    return make_targets_bottom(identity_bottom)
  elif modality_type == ModalityType.IMAGE:
    return image_targets_bottom
  elif modality_type in (ModalityType.IMAGE_CHANNEL_BOTTOM_IDENTITY,
                         ModalityType.IMAGE_CHANNEL_COMPRESS):
    return image_channel_compress_targets_bottom
  elif modality_type == ModalityType.IMAGE_CHANNEL_EMBEDDINGS_BOTTOM:
    return image_channel_embeddings_bottom
  elif modality_type in (ModalityType.REAL,
                         ModalityType.REAL_L2_LOSS,
                         ModalityType.REAL_LOG_POISSON_LOSS):
    return make_targets_bottom(real_bottom)
  elif modality_type == ModalityType.SPEECH_RECOGNITION:
    return make_targets_bottom(speech_recognition_bottom)
  elif modality_type == ModalityType.SYMBOL_ONE_HOT:
    return symbol_one_hot_bottom
  elif modality_type in (ModalityType.VIDEO,
                         ModalityType.VIDEO_L1,
                         ModalityType.VIDEO_L2):
    return video_targets_bottom
  elif modality_type == ModalityType.VIDEO_BITWISE:
    return video_bitwise_targets_bottom
  elif modality_type == ModalityType.VIDEO_IDENTITY:
    return video_identity_targets_bottom
  elif modality_type in (ModalityType.VIDEO_L1_RAW,
                         ModalityType.VIDEO_L2_RAW):
    return video_raw_targets_bottom
  elif modality_type == ModalityType.VIDEO_PIXEL_NOISE:
    return make_targets_bottom(video_pixel_noise_bottom)
  return value


def get_top(modality_type, value=None):
  """Gets default top transformation; if none available, return value."""
  if modality_type in (ModalityType.AUDIO,
                       ModalityType.AUDIO_SPECTRAL,
                       ModalityType.GENERIC_L2_LOSS,
                       ModalityType.IDENTITY,
                       ModalityType.IDENTITY_SYMBOL,
                       ModalityType.IMAGE_CHANNEL_BOTTOM_IDENTITY,
                       ModalityType.SPEECH_RECOGNITION,
                       ModalityType.VIDEO_IDENTITY):
    return identity_top
  elif modality_type in (ModalityType.CLASS_LABEL,
                         ModalityType.MULTI_LABEL,
                         ModalityType.ONE_HOT_CLASS_LABEL,
                         ModalityType.SIGMOID_CLASS_LABEL):
    return class_label_top
  elif modality_type in (ModalityType.CTC_SYMBOL,
                         ModalityType.SYMBOL,
                         ModalityType.SYMBOL_WEIGHTS_ALL):
    return symbol_top
  elif modality_type == ModalityType.IMAGE:
    return image_top
  elif modality_type == ModalityType.IMAGE_CHANNEL_COMPRESS:
    return image_channel_compress_top
  elif modality_type == ModalityType.IMAGE_CHANNEL_EMBEDDINGS_BOTTOM:
    return image_channel_embeddings_top
  elif modality_type in (ModalityType.REAL,
                         ModalityType.REAL_L2_LOSS,
                         ModalityType.REAL_LOG_POISSON_LOSS):
    return real_top
  elif modality_type == ModalityType.SIGMOID_MAX_POOLING_CLASS_LABEL:
    return sigmoid_max_pooling_class_label_top
  elif modality_type == ModalityType.SOFTMAX_AVERAGE_POOLING_CLASS_LABEL:
    return softmax_average_pooling_class_label_top
  elif modality_type == ModalityType.SOFTMAX_LAST_TIMESTEP_CLASS_LABEL:
    return softmax_last_timestep_class_label_top
  elif modality_type == ModalityType.SOFTMAX_MAX_POOLING_CLASS_LABEL:
    return softmax_max_pooling_class_label_top
  elif modality_type == ModalityType.SYMBOL_ONE_HOT:
    return symbol_one_hot_top
  elif modality_type in (ModalityType.VIDEO,
                         ModalityType.VIDEO_BITWISE,
                         ModalityType.VIDEO_PIXEL_NOISE):
    return video_top
  elif modality_type in (ModalityType.VIDEO_L1,
                         ModalityType.VIDEO_L2):
    return video_l1_top
  elif modality_type in (ModalityType.VIDEO_L1_RAW,
                         ModalityType.VIDEO_L2_RAW):
    return video_raw_top
  return value


def get_weights_fn(modality_type, value=None):
  """Gets default weights function; if none available, return value."""
  if modality_type in (ModalityType.CTC_SYMBOL,
                       ModalityType.IDENTITY_SYMBOL,
                       ModalityType.MULTI_LABEL,
                       ModalityType.SYMBOL,
                       ModalityType.SYMBOL_ONE_HOT):
    return common_layers.weights_nonzero
  elif modality_type in ModalityType.get_choices():
    return common_layers.weights_all
  return value

class SymbolModality(modality.Modality):
  """
  Modality
  for sets of discrete symbols.

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
          hp.mode != tf_estimator.ModeKeys.TRAIN):
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
    if not tf.contrib.eager.in_eager_mode():
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
          self._model_hparams.mode == tf_estimator.ModeKeys.TRAIN):
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


class ImageModality(modality.Modality):
  """Modality for images."""
  PIXEL_EMBEDDING_SIZE = 64

  def bottom(self, x):
    with tf.compat.v1.variable_scope(self.name):
      if not tf.contrib.eager.in_eager_mode():
        tf.compat.v1.summary.image(
            "inputs", common_layers.tpu_safe_image_summary(x), max_outputs=2)
      return tf.cast(x, dtype=tf.float32)

  def targets_bottom(self, x):
    inputs = x
    with tf.compat.v1.variable_scope(self.name):
      if not tf.contrib.eager.in_eager_mode():
        tf.compat.v1.summary.image(
            "targets_bottom",
            common_layers.tpu_safe_image_summary(inputs),
            max_outputs=1)
      inputs_shape = common_layers.shape_list(inputs)
      if len(inputs_shape) != 4:
        raise ValueError("Assuming images given as int tensors in the format "
                         "[batch, height, width, channels] (256 values).")
      # We embed each of 256=self.top_dimensionality possible pixel values.
      embedding_var = tf.compat.v1.get_variable(
          "pixel_embedding",
          [self.top_dimensionality, self.PIXEL_EMBEDDING_SIZE])
      hot_inputs = tf.one_hot(tf.cast(inputs, dtype=tf.int32), self.top_dimensionality)
      hot_inputs = tf.reshape(hot_inputs, [-1, self.top_dimensionality])
      embedded = tf.matmul(hot_inputs, embedding_var)
      # Let's now merge all channels that were embedded into a single vector.
      merged_size = self.PIXEL_EMBEDDING_SIZE * inputs_shape[3]
      embedded = tf.reshape(embedded, inputs_shape[:3] + [merged_size])
      merged = tf.compat.v1.layers.dense(
          embedded,
          self._body_input_depth,
          name="merge_pixel_embedded_channels")
      return merged

  def top(self, body_output, _):
    # TODO(lukaszkaiser): is this a universal enough way to get channels?
    num_channels = self._model_hparams.problem.num_channels
    with tf.compat.v1.variable_scope("rgb_softmax"):
      body_output_shape = common_layers.shape_list(body_output)
      reshape_shape = body_output_shape[:3]
      reshape_shape.extend([num_channels, self.top_dimensionality])
      res = tf.compat.v1.layers.dense(body_output, self.top_dimensionality * num_channels)
      res = tf.reshape(res, reshape_shape)
      if not tf.compat.v1.get_variable_scope().reuse:
        res_argmax = tf.argmax(res, axis=-1)
        tf.compat.v1.summary.image(
            "result",
            common_layers.tpu_safe_image_summary(res_argmax),
            max_outputs=1)
      return res

  def loss(self, top_out, targets):
    """Compute loss numerator and denominator for one shard of output."""
    logits = top_out
    cutoff = getattr(self._model_hparams, "video_modality_loss_cutoff", 0.0)
    return common_layers.padded_cross_entropy(
        logits,
        targets,
        self._model_hparams.label_smoothing,
        cutoff=cutoff,
        weights_fn=self.targets_weights_fn)


class ImageChannelCompressModality(modality.Modality):
  """Modality for images using channel compression for generation."""

  @property
  def num_channels(self):
    return 3

  def bottom_compress(self, inputs, name="bottom"):
    """Compresses channel-wise input pixels into whole pixel representions.

    Perform conversion of RGB pixel values to a real number in the range -1 to
    1. This combines pixel channels to form a representation of shape
    [img_len, img_len].

    Args:
      inputs: Tensor representing RGB pixel intensities as integers, of shape
        [batch, img_len, img_len, channels].
      name: string, scope.

    Returns:
      body_input: Tensor of shape [batch, img_len, img_len, body_input_depth].
    """
    with tf.compat.v1.variable_scope(name):
      inputs = tf.cast(inputs, dtype=tf.float32)
      hp = self._model_hparams
      if hp.mode != tf_estimator.ModeKeys.PREDICT:
        tf.compat.v1.summary.image(
            "inputs",
            common_layers.tpu_safe_image_summary(inputs),
            max_outputs=2)
      inputs = common_layers.convert_rgb_to_symmetric_real(inputs)

      # Reshape inputs to apply convolutions across [img_len, img_len*channels].
      inputs_shape = common_layers.shape_list(inputs)
      inputs = tf.reshape(
          inputs, [-1, inputs_shape[1], inputs_shape[2] * inputs_shape[3], 1])

      # Compress RGB intensities for each pixel using a convolution.
      outputs = tf.compat.v1.layers.conv2d(
          inputs,
          self._body_input_depth,
          kernel_size=(1, self.num_channels),
          padding="VALID",
          strides=(1, self.num_channels),
          activation=tf.nn.relu,
          name="conv_input")
      return outputs

  def bottom(self, x):
    return self.bottom_compress(x, "input_bottom")

  def targets_bottom(self, x):
    return self.bottom_compress(x, "output_bottom")

  def top(self, body_output, _):
    """Transforms body output to return logits.

    Args:
      body_output: Tensor of shape [batch, img_len, img_len, depth].

    Returns:
      Tensor of shape [batch, img_len, img_len, channels, top_dimensionality].
    """
    with tf.compat.v1.variable_scope(self.name):
      hidden_size = self._model_hparams.hidden_size
      img_len = self._model_hparams.img_len
      channels = self.num_channels  # RGB
      batch = common_layers.shape_list(body_output)[0]
      x = tf.compat.v1.layers.conv2d(
          body_output,
          hidden_size * channels,
          kernel_size=(1, 1),
          strides=(1, 1),
          padding="VALID",
          activation=tf.nn.relu,
          name="decompress_conv")
      x = tf.reshape(x, [batch, img_len, img_len * channels, hidden_size])
      x = common_layers.layer_preprocess(x, self._model_hparams)
      x = tf.compat.v1.layers.dense(x,
                          self.top_dimensionality,
                          use_bias=True,
                          activation=None,
                          name="output_conv")
      x = tf.reshape(
          x, [batch, img_len, img_len, channels, self.top_dimensionality])
      return x


class ImageChannelBottomIdentityModality(ImageChannelCompressModality):

  def top(self, body_output, _):
    return body_output


class ImageChannelEmbeddingsBottom(modality.Modality):
  """Modality for images using channel compression for generation."""

  def get_channel_embeddings(self,
                             io_depth,
                             targets,
                             hidden_size,
                             name="channel"):
    """Get separate embedding for each of the channels."""
    targets_split = tf.split(targets, io_depth, axis=3)
    rgb_embedding_var = tf.compat.v1.get_variable("rgb_target_emb_%s" % name,
                                        [256 * io_depth, hidden_size])
    rgb_embedding_var = tf.identity(rgb_embedding_var)
    rgb_embedding_var *= float(hidden_size)**0.5
    channel_target_embs = []
    for i in range(io_depth):
      # Adding the channel offsets to get the right embedding since the
      # embedding tensor has shape 256 * io_depth, hidden_size
      target_ids = tf.squeeze(targets_split[i], axis=3) + i * 256
      target_embs = common_layers.gather(rgb_embedding_var, target_ids)
      channel_target_embs.append(target_embs)

    return tf.concat(channel_target_embs, axis=-1)

  def targets_bottom(self, x):
    inputs = x
    io_depth = self._model_hparams.num_channels
    tshape = common_layers.shape_list(inputs)
    hidden_size = self._model_hparams.hidden_size
    target_embeddings = self.get_channel_embeddings(io_depth, inputs,
                                                    hidden_size, "input_bottom")
    return tf.reshape(target_embeddings,
                      [tshape[0], tshape[1], tshape[2] * io_depth, hidden_size])

  def top(self, body_output, _):
    with tf.compat.v1.variable_scope(self.name):
      img_len = self._model_hparams.img_len
      channels = self._model_hparams.num_channels
      x = tf.compat.v1.layers.dense(
          body_output, 256, use_bias=True, activation=None, name="output_conv")
      x = tf.reshape(x,
                     [-1, img_len, img_len, channels, self.top_dimensionality])
      return x


class AudioModality(modality.Modality):
  """Performs strided conv compressions for audio data."""

  def bottom(self, x):
    """Transform input from data space to model space.

    Args:
      x: A Tensor with shape [batch, ...]
    Returns:
      body_input: A Tensor with shape [batch, ?, ?, body_input_depth].
    """
    inputs = x
    with tf.compat.v1.variable_scope(self.name):
      # TODO(aidangomez): Will need to sort out a better audio pipeline
      def xnet_resblock(x, filters, res_relu, name):
        """Xception block."""
        with tf.compat.v1.variable_scope(name):
          # Typically audio samples are >100k samples in length and have a width
          # of 2 or 4. Mono audio has a single channel while stereo has 2.
          y = common_layers.separable_conv_block(
              x,
              filters, [((1, 1), (3, 3)), ((1, 1), (3, 3))],
              first_relu=True,
              padding="SAME",
              force2d=True,
              name="sep_conv_block")
          y = common_layers.pool(y, (3, 3), "MAX", "SAME", strides=(2, 2))
          return y + common_layers.conv_block(
              x,
              filters, [((1, 1), (1, 1))],
              padding="SAME",
              strides=(2, 2),
              first_relu=res_relu,
              force2d=True,
              name="res_conv0")

      x = tf.cast(inputs, dtype=tf.float32) / 255.
      x.set_shape([None, None, None, 1])
      for i in range(self._model_hparams.audio_compression):
        x = xnet_resblock(x, 2**(i + 1), True, "compress_block_%d" % i)
      return xnet_resblock(x, self._body_input_depth, False,
                           "compress_block_final")


class AudioSpectralModality(modality.Modality):
  """Performs strided conv compressions for audio spectral data."""

  def bottom(self, x):
    """Transform input from data space to model space.

    Args:
      x: A Tensor with shape [batch, ...]
    Returns:
      body_input: A Tensor with shape [batch, ?, ?, body_input_depth].
    """
    inputs = x
    with tf.compat.v1.variable_scope(self.name):
      # TODO(aidangomez): Will need to sort out a better audio pipeline
      def xnet_resblock(x, filters, res_relu, name):
        """Xception-like block."""
        with tf.compat.v1.variable_scope(name):
          # We only stride along the length dimension to preserve the spectral
          # bins (which are tiny in dimensionality relative to length)
          y = common_layers.separable_conv_block(
              x,
              filters, [((1, 1), (3, 3)), ((1, 1), (3, 3))],
              first_relu=True,
              padding="SAME",
              force2d=True,
              name="sep_conv_block")
          y = common_layers.pool(y, (3, 3), "MAX", "SAME", strides=(2, 1))
          return y + common_layers.conv_block(
              x,
              filters, [((1, 1), (1, 1))],
              padding="SAME",
              strides=(2, 1),
              first_relu=res_relu,
              force2d=True,
              name="res_conv0")

      # Bitcast back from int32
      x = tf.bitcast(inputs, tf.float32)
      x.set_shape([None, None, None, 1])
      for i in range(self._model_hparams.audio_compression):
        x = xnet_resblock(x, 2**(i + 1), True, "compress_block_%d" % i)
      return xnet_resblock(x, self._body_input_depth, False,
                           "compress_block_final")


class SpeechRecognitionModality(modality.Modality):
  """Common ASR filterbank processing."""

  def bottom(self, x):
    """Use batchnorm instead of CMVN and shorten the stft with strided convs.

    Args:
      x: float32 tensor with shape [batch_size, len, 1, freqs * channels]

    Returns:
      float32 tensor with shape [batch_size, shorter_len, 1, hidden_size]
    """
    inputs = x
    p = self._model_hparams

    num_mel_bins = p.audio_num_mel_bins
    num_channels = 3 if p.audio_add_delta_deltas else 1

    with tf.compat.v1.variable_scope(self.name):
      if p.audio_preproc_in_bottom:
        # Compute filterbanks
        with tf.compat.v1.variable_scope("fbanks"):
          waveforms = tf.squeeze(inputs, [2, 3])
          mel_fbanks = common_audio.compute_mel_filterbank_features(
              waveforms,
              sample_rate=p.audio_sample_rate,
              dither=p.audio_dither,
              preemphasis=p.audio_preemphasis,
              frame_length=p.audio_frame_length,
              frame_step=p.audio_frame_step,
              lower_edge_hertz=p.audio_lower_edge_hertz,
              upper_edge_hertz=p.audio_upper_edge_hertz,
              num_mel_bins=p.audio_num_mel_bins,
              apply_mask=True)
          if p.audio_add_delta_deltas:
            mel_fbanks = common_audio.add_delta_deltas(mel_fbanks)
          x = tf.reshape(mel_fbanks,
                         common_layers.shape_list(mel_fbanks)[:2] +
                         [num_mel_bins, num_channels])

          nonpadding_mask = 1. - common_attention.embedding_to_padding(x)
          num_of_nonpadding_elements = tf.reduce_sum(
              nonpadding_mask) * num_mel_bins * num_channels

          # This replaces CMVN estimation on data
          var_epsilon = 1e-09
          mean = tf.reduce_sum(
              x, axis=[1], keepdims=True) / num_of_nonpadding_elements
          variance = (num_of_nonpadding_elements * mean**2. -
                      2. * mean * tf.reduce_sum(x, axis=[1], keepdims=True) +
                      tf.reduce_sum(x**2, axis=[1], keepdims=True)
                     ) / num_of_nonpadding_elements
          x = (x - mean) * tf.math.rsqrt(variance + var_epsilon) * tf.expand_dims(
              nonpadding_mask, -1)
      else:
        x = inputs

      # The convention is that the models are flattened along the spatial,
      # dimensions, thus the speech preprocessor treats frequencies and
      # channels as image colors (last axis)
      x.set_shape([None, None, num_mel_bins, num_channels])

      # TODO(chorowski): how to specify bottom's hparams and avoid hardcoding?
      x = tf.pad(x, [[0, 0], [0, 8], [0, 0], [0, 0]])
      for _ in range(2):
        x = tf.compat.v1.layers.conv2d(
            x, 128, (3, 3), (2, 2), use_bias=False)
        x = common_layers.layer_norm(x)
        x = tf.nn.relu(x)

      xshape = common_layers.shape_list(x)
      # apply a conv that will remove all frequencies and at the same time
      # project the output into desired hidden_size
      x = tf.pad(x, [[0, 0], [0, 2], [0, 0], [0, 0]])
      x = tf.compat.v1.layers.conv2d(x, p.hidden_size, (3, xshape[2]), use_bias=False)

      assert common_layers.shape_list(x)[2] == 1
      x = common_layers.layer_norm(x)
      x = tf.nn.relu(x)
    return x


class VideoModality(modality.Modality):
  """Modality for videos, i.e., time-sequences of frames."""

  def bottom(self, x):
    common_video.gif_summary("inputs", x, max_outputs=1)
    x = common_layers.standardize_images(x)
    return x

  def targets_bottom(self, x):
    common_video.gif_summary("targets", x, max_outputs=1)
    x = common_layers.standardize_images(x)
    return x

  def top(self, body_output, targets):
    num_channels = self._model_hparams.problem.num_channels
    shape = common_layers.shape_list(body_output)
    reshape_shape = shape[:-1] + [num_channels, self.top_dimensionality]
    res = tf.reshape(body_output, reshape_shape)
    # Calculate argmax so as to have a summary with the produced images.
    x = tf.argmax(tf.reshape(res, [-1, self.top_dimensionality]), axis=-1)
    x = tf.reshape(x, shape[:-1] + [num_channels])
    common_video.gif_summary("results", x, max_outputs=1)
    return res

  def loss(self, top_out, targets):
    """Compute loss numerator and denominator for one shard of output."""
    logits = top_out
    logits = tf.reshape(logits, [-1] + common_layers.shape_list(logits)[2:])
    targets = tf.reshape(targets, [-1] + common_layers.shape_list(targets)[2:])
    cutoff = getattr(self._model_hparams, "video_modality_loss_cutoff", 0.01)
    return common_layers.padded_cross_entropy(
        logits,
        targets,
        self._model_hparams.label_smoothing,
        cutoff=cutoff,
        weights_fn=self.targets_weights_fn)


class VideoModalityBitwise(VideoModality):
  """Video Modality where bottom embeds pixels bitwise."""
  PIXEL_EMBEDDING_SIZE = 64

  def bottom(self, x):
    inputs = x
    with tf.compat.v1.variable_scope(self.name, reuse=tf.compat.v1.AUTO_REUSE):
      common_layers.summarize_video(inputs, "bottom")
      # Embed bitwise.
      assert self.top_dimensionality == 256
      embedded = discretization.int_to_bit_embed(inputs, 8,
                                                 self.PIXEL_EMBEDDING_SIZE)
      # Project.
      return tf.compat.v1.layers.dense(
          embedded,
          self._body_input_depth,
          name="merge_pixel_embedded_frames")

  def targets_bottom(self, x):  # pylint: disable=arguments-differ
    inputs = x
    with tf.compat.v1.variable_scope(self.name, reuse=tf.compat.v1.AUTO_REUSE):
      common_layers.summarize_video(inputs, "targets_bottom")
      # Embed bitwise.
      assert self.top_dimensionality == 256
      embedded = discretization.int_to_bit_embed(inputs, 8,
                                                 self.PIXEL_EMBEDDING_SIZE)
      # Transpose and project.
      transposed = common_layers.time_to_channels(embedded)
      return tf.compat.v1.layers.dense(
          transposed,
          self._body_input_depth,
          name="merge_pixel_embedded_frames")


class VideoModalityPixelNoise(VideoModality):
  """Video modality that introduces pixel noise on input during training."""

  def bottom(self, x):
    inputs = x
    if self._model_hparams.mode == tf_estimator.ModeKeys.TRAIN:
      background = tf.contrib.distributions.percentile(inputs, 50.,
                                                       axis=[0, 1, 2, 3])
      input_shape = common_layers.shape_list(inputs)
      input_size = tf.reduce_prod(input_shape[:-1])
      input_mask = tf.random.categorical(
          tf.math.log([[self.input_noise, 1.-self.input_noise]]), input_size)
      input_mask = tf.reshape(tf.cast(input_mask, tf.int32),
                              input_shape[:-1]+[1])
      inputs = inputs * input_mask + background * (1 - input_mask)
    return super(VideoModalityPixelNoise, self).bottom(inputs)

  @property
  def input_noise(self):
    return getattr(self._model_hparams, "video_modality_input_noise", 0.25)


class VideoModalityL1(VideoModality):
  """Video modality that predicts a scalar per channel with an L1 loss."""

  def top(self, body_output, _):
    num_channels = self._model_hparams.problem.num_channels
    num_frames = self._model_hparams.video_num_target_frames
    with tf.compat.v1.variable_scope("rgb"):
      body_output_shape = common_layers.shape_list(body_output)
      res = tf.compat.v1.layers.dense(body_output, num_channels * num_frames, name="cast")
      res = tf.reshape(res, body_output_shape[:3] + [num_channels, num_frames])
      res = tf.transpose(res, [0, 4, 1, 2, 3])  # Move frames next to batch.
      if not tf.compat.v1.get_variable_scope().reuse:
        res_argmax = res[:, -1, :, :, :]
        tf.compat.v1.summary.image(
            "result",
            common_layers.tpu_safe_image_summary(res_argmax),
            max_outputs=1)
      return tf.expand_dims(res, axis=-1)  # Add an axis like in perplexity.

  @property
  def cutoff(self):
    return getattr(self._model_hparams, "video_modality_loss_cutoff", 0.2)

  def internal_loss(self, logits, targets):
    return tf.nn.relu(tf.abs(logits - targets) - self.cutoff)

  def loss(self, top_out, targets):
    """Compute loss numerator and denominator for one shard of output."""
    logits = top_out
    logits = tf.reshape(logits, [-1] + common_layers.shape_list(logits)[2:-1])
    targets = tf.reshape(targets, [-1] + common_layers.shape_list(targets)[2:])
    weights = self.targets_weights_fn(targets)
    # Shift targets by 0.5 so later just casting to int gives the prediction.
    # So for int targets, say 0 and 7, we actually train to predict 0.5 and 7.5.
    # Later (in merics or infer) this is cast to int anyway. Also, we have no
    # loss beyond self.cutoff = 0.2 as these are already correct predictions.
    targets = tf.cast(targets, dtype=tf.float32) + 0.5
    loss = self.internal_loss(logits, targets)
    return tf.reduce_sum(loss * weights), tf.reduce_sum(weights)


class VideoModalityL2(VideoModalityL1):
  """Modality for videos with L2 loss."""

  def internal_loss(self, logits, targets):
    return tf.nn.relu((logits - targets)**2 - self.cutoff * self.cutoff)


class VideoModalityL2Raw(VideoModalityL2):
  """Modality with L2 loss and raw input (sequences of frames)."""

  def convert_rgb_to_real(self, prediction, targets):
    """Convert prediction and target from rgb to real."""
    prediction = tf.squeeze(prediction, axis=-1)
    prediction = common_layers.convert_rgb_to_real(prediction)
    targets = common_layers.convert_rgb_to_real(targets)
    return prediction, targets

  def bottom(self, x):
    common_video.gif_summary("inputs", x)
    return common_layers.convert_rgb_to_real(x)

  def targets_bottom(self, x):  # pylint: disable=arguments-differ
    common_video.gif_summary("targets_bottom", x)
    return common_layers.convert_rgb_to_real(x)

  def top(self, body_output, _):
    frames = body_output
    if isinstance(body_output, list):
      frames = tf.stack(body_output, axis=1)
    rgb_frames = common_layers.convert_real_to_rgb(frames)
    common_video.gif_summary("body_output", rgb_frames)
    return tf.expand_dims(rgb_frames, axis=-1)

  def loss(self, top_out, targets):
    prediction, groundtruth = self.convert_rgb_to_real(top_out, targets)
    loss = tf.compat.v1.losses.mean_squared_error(prediction, groundtruth)
    return loss, tf.constant(1.0)


class VideoModalityL1Raw(VideoModalityL2Raw):
  """Modality with L1 loss and raw input (sequences of frames)."""

  def loss(self, top_out, targets):
    prediction, groundtruth = self.convert_rgb_to_real(top_out, targets)
    loss = tf.compat.v1.losses.absolute_difference(prediction, groundtruth)
    return loss, tf.constant(1.0)


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
