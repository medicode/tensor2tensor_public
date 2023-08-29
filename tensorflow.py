"""Wrapper for tensorflow to ensure we disable TF2 behaviours"""
import os
import sys

old_path = sys.path
# Hack path to exclude Fathom sources and import the real tensorflow.
# The `and p` catches the case when Python is invoked as a repl and the path contains
# the empty string (current directory).
sys.path = [p for p in sys.path if p != os.path.dirname(__file__) and p]
# before the import, also need to remove this module from the cache
assert (
    __name__ in sys.modules
), "Sanity check: this module should have been set in the module cache"
old_entry = sys.modules.pop(__name__)

# Import all of TF
# pylint:disable=wildcard-import,import-self
from tensorflow import *
import tensorflow as real_tf

# pylint:enable=wildcard-import,import-self

# restore the path and module cache
sys.path = old_path
# insert this wrapper into the imported modules cache instead of the real tensorflow
sys.modules[__name__] = old_entry

real_tf.compat.v1.disable_v2_behavior()

# set the path of this module to be the same as real tensorflow - thus Python will treat
# this module as a package, and any import statements referencing this module will
# search the same paths as the real tensorflow
__path__ = real_tf.__path__
