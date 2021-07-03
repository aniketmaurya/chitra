from unittest.mock import MagicMock

import numpy as np
import pytest
import tensorflow as tf
from tensorflow import keras

from chitra.trainer import Dataset, InterpretModel, Trainer, create_cnn

dataset = Dataset("./")
cnn = create_cnn("mobilenetv2", num_classes=1000, keras_applications=False)
trainer = Trainer(dataset, cnn)
model_interpret = InterpretModel(True, trainer)
image_tensor = tf.random.normal((24, 24, 1))


def test_create_cnn():
    assert cnn.trainable is True
    assert isinstance(cnn, keras.models.Model)


def test_trainer():
    assert isinstance(trainer.model, keras.models.Model)


def test_interpret_model():
    assert isinstance(model_interpret.learner, Trainer)


def test_build():
    with pytest.raises(NotImplementedError):
        trainer.build()


def test_warmup():
    with pytest.raises(NotImplementedError):
        trainer.warmup()


def test_prewhiten():
    rescaled_img = trainer.prewhiten(image_tensor).numpy()
    restored_img = (rescaled_img + 1) * 127.5

    assert np.allclose(restored_img, image_tensor.numpy(), 1e-3, 1e-3)


def test_rescale():
    assert trainer.rescale(image_tensor, 0)[1] == 0


def test_summary():
    trainer.model.summary = MagicMock()
    trainer.summary()
    trainer.model.summary.assert_called_once()


def test_fit():
    trainer.model.fit = MagicMock()
    train_ds = MagicMock()
    trainer.fit(train_ds)
    trainer.model.fit.assert_called_once()


def test__get_optimizer():
    optim = trainer._get_optimizer(tf.optimizers.Adam)()
    assert isinstance(optim, tf.optimizers.Adam)
