# Serve Deep Learning Model with CHITRA Model Server

Create Rest API or Interactive UI app for Any Learning Model - ML, DL, Image Classification, NLP, Tensorflow or PyTorch.

## What can it do?

- Serve Any Learning Model
- Predefined processing functions for image classification (NLP processing functions coming soon)
- Override custom preprocessing and Postprocessing function with your own.
- Request Response Schema (JSON body) will be changed based on the `api_type`.

> pip install -U "chitra[serve]"

### Default available API types are:

1. Image Classification
1. Object Detection
1. Text Classification
1. Question Answering

To get a full list of available API types you can call `chitra.serve.get_available_api_types()`.

## Text Classification API

You can easily create Sentiment Analysis API. In this example, I will use HuggingFace to load the Sentiment Analysis
Model but feel free to use other models as well.

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

from chitra.serve import create_api

tokenizer = AutoTokenizer.from_pretrained("finiteautomata/beto-sentiment-analysis")
model = AutoModelForSequenceClassification.from_pretrained(
    "finiteautomata/beto-sentiment-analysis"
)
classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

create_api(classifier, run=True, api_type="text-classification")
```

You can open `http://127.0.0.1:8000/docs` Swagger UI in your browser to test the API 🔥

## Image Classification API

```python
from chitra.serve import create_api
from chitra.trainer import create_cnn


model = create_cnn('mobilenetv2', num_classes=2)

create_api(model, run=True, api_type='image-classification')
```

Open in your browser and try out the API. You can upload any image to try.

#### Swagger UI Preview

![png](preview-qna.png)
