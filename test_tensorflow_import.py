"""Module to test tensorflow wrapper"""
import tensorflow

# These imports are part of the test, to make sure they do not fail
# pylint:disable=unused-import
from tensorflow.core import example
import tensorflow.estimator as tf_est


# pylint:enable=unused-import


def test_tensorflow_wrapper() -> None:
    """Testing our tensorflow wrapper"""
    assert (
        tensorflow.real_tf is not None
    ), "We expect 'import tensorflow' to use the wrapper, not the original tensorflow."
    assert (
        not tensorflow.compat.v1.executing_eagerly()
    ), "We expect V2 behaviors to be disabled."
