import os

import tensorflow as tf

from chitra.core import get_basename, load_imagenet_labels, remove_dsstore


def test_remove_dsstore():
    os.makedirs("chitra_temp", exist_ok=True)
    ds_store = "chitra_temp/.DS_Store"
    open(ds_store, "w").close()
    assert os.path.exists(ds_store)
    remove_dsstore("chitra_temp")
    assert not os.path.exists(ds_store)
    os.removedirs("chitra_temp")


def test_get_basename():
    assert get_basename(tf.constant("hello/world")) == "world"


def test_load_imagenet_labels():
    labels = load_imagenet_labels()

    assert "\n" not in labels
    assert len(labels) == 1000 + 1
