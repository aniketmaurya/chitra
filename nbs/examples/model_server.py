from chitra.serve import create_app
from chitra.trainer import create_cnn

model = create_cnn('mobilenetv2', num_classes=2)

app = create_app(model, run=True)
