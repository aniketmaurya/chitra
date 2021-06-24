from tensorflow import keras

from chitra.trainer import create_cnn
from chitra.trainer import Dataset
from chitra.trainer import InterpretModel
from chitra.trainer import Trainer

dataset = Dataset('./')
cnn = create_cnn('mobilenetv2', num_classes=1000, keras_applications=False)
trainer = Trainer(dataset, cnn)
model_interpret = InterpretModel(True, trainer)


def test_create_cnn():
    assert cnn.trainable is True
    assert isinstance(cnn, keras.models.Model)


def test_trainer():
    assert isinstance(trainer.model, keras.models.Model)


def test_interpret_model():
    assert isinstance(model_interpret.learner, Trainer)
