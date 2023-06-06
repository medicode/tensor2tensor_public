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

"""Models defined in T2T. Imports here force registration."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# pylint: disable=unused-import

from tensor2tensor.layers import modalities  # pylint: disable=g-import-not-at-top
from tensor2tensor.models import bytenet
from tensor2tensor.models import distillation
from tensor2tensor.models import lstm
from tensor2tensor.models import neural_gpu
from tensor2tensor.models import resnet
from tensor2tensor.models import revnet
from tensor2tensor.models import transformer
from tensor2tensor.models import vanilla_gan
from tensor2tensor.models import xception

# FATHOM
# comment out so gym not required
#from tensor2tensor.models.research import rl
from tensor2tensor.models.research import similarity_transformer
from tensor2tensor.models.research import super_lm
from tensor2tensor.models.research import transformer_moe
from tensor2tensor.models.research import transformer_nat
from tensor2tensor.models.research import transformer_revnet
from tensor2tensor.models.research import transformer_symshard
from tensor2tensor.models.research import transformer_vae
from tensor2tensor.models.research import universal_transformer
from tensor2tensor.models.research import vqa_attention
from tensor2tensor.models.research import vqa_recurrent_self_attention
from tensor2tensor.models.research import vqa_self_attention


from tensor2tensor.utils import registry

# pylint: enable=unused-import

# bug reproduction
from tensor2tensor.models import lstm2

def model(name):
  return registry.model(name)
