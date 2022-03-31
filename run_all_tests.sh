# Runs t2t unit tests.
# Intended strictly to run on circleci--will not work locally.
#
# TODO: converge w/ _local.sh unit test script.

#!/usr/bin/env bash

set -euxo pipefail

T2T=tensor2tensor
DT=diseaseTools
DTC=diseaseTools-config
IMAGE=us.gcr.io/fathom-containers/t2t_test
GCS_KEY_NAME=GOOGLE_APPLICATION_CREDENTIALS
GCS_KEY_PATH=/usr/src/diseaseTools/gcloud/keys/google-auth.json

docker pull $IMAGE

docker run -it \
       -v $HOME/$DT:/usr/src/diseaseTools \
       -v $HOME/$DTC:/usr/src/diseaseTools-config \
       -v $HOME/gdm:/usr/src/diseaseTools/gcloud/gdm \
       -v $HOME/$T2T:/usr/src/t2t \
       -w /usr/src/t2t \
       --env PYTHONPATH=/usr/src/t2t:/usr/src/diseaseTools:/usr/src/diseaseTools-config \
       --env $GCS_KEY_NAME=$GCS_KEY_PATH \
       $IMAGE \
       python3 -m pytest -vv \
       --ignore=/usr/src/t2t/tensor2tensor/utils/registry_test.py \
       --ignore=/usr/src/t2t/tensor2tensor/utils/trainer_lib_test.py \
       --ignore=/usr/src/t2t/tensor2tensor/visualization/visualization_test.py \
       --ignore=/usr/src/t2t/tensor2tensor/bin/t2t_trainer_test.py \
       --ignore=/usr/src/t2t/tensor2tensor/data_generators/algorithmic_math_test.py \
       --ignore=/usr/src/t2t/tensor2tensor/rl/ \
       --ignore=/usr/src/t2t/tensor2tensor/data_generators/allen_brain_test.py \
       --ignore=/usr/src/t2t/tensor2tensor/problems_test.py \
       --ignore=/usr/src/t2t/tensor2tensor/data_generators/gym_problems_test.py \
       --ignore=/usr/src/t2t/tensor2tensor/utils/checkpoint_compatibility_test.py \
       --ignore=/usr/src/t2t/tensor2tensor/models/video/ \
       --ignore=/usr/src/t2t/tensor2tensor/models/research/next_frame_test.py \
       --ignore=/usr/src/t2t/tensor2tensor/rl/trainer_model_based_stochastic_test.py \
       --ignore=/usr/src/t2t/tensor2tensor/rl/trainer_model_based_sv2p_test.py \
       --ignore=/usr/src/t2t/tensor2tensor/models/research/glow_test.py \
       --deselect=/usr/src/t2t/tensor2tensor/layers/common_video_test.py::CommonVideoTest::testGifSummary \
       --ignore=/usr/src/t2t/tensor2tensor/data_generators/image_utils_test.py \
       --ignore=/usr/src/t2t/tensor2tensor/data_generators/video_utils_test.py \
       --ignore=/usr/src/t2t/tensor2tensor/layers/common_video_test.py \
       --ignore=/usr/src/t2t/tensor2tensor/data_generators/common_voice_test.py \
       --ignore=/usr/src/t2t/tensor2tensor/data_generators/gym_env_test.py \
       --junitxml=/usr/src/t2t/test_results/pytest/unittests.xml \
       /usr/src/t2t/tensor2tensor/

