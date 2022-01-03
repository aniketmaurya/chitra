import numpy as np
import tensorflow as tf

from chitra.dataloader import Clf

shape = (32, 32)
class_names = ("a", "b")
clf = Clf()
clf.shape = shape
clf.CLASS_NAMES = class_names


def test__encode_classes():
    clf._encode_classes()
    assert clf.class_to_idx == {"a": 0, "b": 1}


def test__ensure_shape():
    img, label = tf.random.normal((32, 32, 3)), 0
    img1, label1 = clf._ensure_shape(img, label)
    assert np.all(img1 == img)
    assert np.all(label1 == label)


def test__get_image_list():
    assert len(clf._get_image_list(".")) != 0
