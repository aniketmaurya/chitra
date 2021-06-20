# Chitra Model Server

## Serve your models easily with `chitra.serve`

```python
pip install chitra==0.0.23
```

```python
from chitra.serve import create_api
from chitra.trainer import create_cnn

```

## What can it do?

- Serve Any Learning Model (framework agnostic)
- Predefined processing functions for image classification (NLP processing functions coming soon)
- Override custom preprocessing and Postprocessing function with your own.

## Image Classification Example

```python
model = create_cnn('mobilenetv2', num_classes=2)

create_api(model, run=True, api_type='image-classification')
```

Open `http://127.0.0.1:8000/docs` in your browser and try out the API. You can upload any image to try out the API.


If you want to try out Text Classification or Question Answering task then all you have to do is just change `api-type="text-classification"` or `api_type="question-ans"` then pass your model and you are all set.

Request Response Schema (JSON body) will be changed based on the `api_type`.


#### Preview
![png](preview.png)