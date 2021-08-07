---
title: Automatic Docker Image Creation for Machine Learning Model APIs
description: "Chitra automatically creates API and Generate Docker image for deployment."
---

# Automatic Docker Image Creation for Any ML/DL Model

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
