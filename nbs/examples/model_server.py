from chitra.serve import create_api

from chitra.trainer import create_cnn

model = create_cnn('mobilenetv2', num_classes=2)
app = create_api(model, run=True, model_type='image-classification')
