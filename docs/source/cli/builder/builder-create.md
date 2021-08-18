---
title: Automatic Docker Image Creation for Machine Learning Model APIs
description: "Chitra automatically creates API and Generate Docker image for deployment."
---

# Automatic Docker Image Creation for Any ML/DL Model 🐳

`chitra` CLI can build docker image for any kind of Machine Learning or Deep Learning Model.

You need to create a `main.py` file which will contain an object of type `chitra.serve.ModelServer` and
its name should be `app`.
If you have any external Python dependency then create a `requirements.txt` file and keep in the same directory.

If the above conditions are satisfied then just run `chitra builder run --path MAIN_FILEPATH`


## Usage

```
chitra builder create [OPTIONS]

Options:
  --path TEXT  [default: ./]
  --port TEXT
  --tag TEXT
  --help       Show this message and exit.
```

`path` is the file location where `main.py` and `requirements.txt` is present.
You can specify which port to run your app on. By default, it is 8080.
To set the tag of the docker image use `tag` argument.


## Example: Auto Build Docker Image for HuggingFace Text Classification Model API

### Create ModelServer
First create Text Classification model from HuggingFace. `chitra` provides `create_api` method
to create API for any kind of ML/DL model. We specify the `api_type="text-classification"` and
get an `app` object.


```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

from chitra.serve import create_api

tokenizer = AutoTokenizer.from_pretrained("microsoft/xtremedistil-l6-h256-uncased")
model = AutoModelForSequenceClassification.from_pretrained(
    "microsoft/xtremedistil-l6-h256-uncased"
)
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

app = create_api(classifier, run=False, api_type="text-classification").app
```

### Use Chitra CLI to auto-build Docker Image
Go to terminal and change the directory to the path where `main.py` is present.
Run `chitra builder create --path ./FILEPATH --tag chitra-server`.

That's all you need to do! You can run the docker image using `docker run -p 8080:8080 chitra-server`
from terminal.
