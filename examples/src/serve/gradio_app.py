import tensorflow as tf

from chitra.core import load_imagenet_labels
from chitra.serve.app import GradioApp

image_shape = (224, 224)
model = tf.keras.applications.MobileNetV2(weights="imagenet")
IMAGENET_LABELS = load_imagenet_labels()


def postprocess(preds):
    preds = tf.argmax(preds, 1).numpy()
    label = IMAGENET_LABELS[preds[0]]
    return label


app = GradioApp(
    "image-classification",
    model=model,
    image_shape=image_shape,
    postprocess_fn=postprocess,
)

app.run(share=True)
