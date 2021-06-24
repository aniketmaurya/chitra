from tensorflow import keras

from chitra.trainer import create_cnn
from chitra.trainer import Trainer

trainer = Trainer(
    ds, create_cnn('mobilenetv2', num_classes=1000, keras_applications=False))
# model_interpret = InterpretModel(True, trainer)
#
# image = Image.fromarray(image)
# model_interpret(image)
# print(IMAGENET_LABELS[285])


def test_create_cnn():
    cnn = create_cnn('mobilenetv2', 1, weights=None)
    assert cnn.trainable is True
    assert isinstance(cnn, keras.models.Model)
