from chitra.serve import create_api

# model = create_cnn('mobilenetv2', num_classes=2)
model = lambda x: x
# app = create_api(model, run=True, model_type='image-classification')
app = create_api(model, run=True, api_type='question-ans')
