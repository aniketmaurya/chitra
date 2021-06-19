# Chitra Model Server

> Serve your models easily with `chitra.serve`

```python
# For latest update install from master
!pip install git+https://github.com/aniketmaurya/chitra@master -q
```

```python
from chitra.serve import create_app
from chitra.trainer import create_cnn

```

## What can it do?

- Serve your Deep Learning Model
- Override custom preprocessing and Postprocessing function.

```python
model = create_cnn('mobilenetv2', num_classes=2)

create_app(model, run=True)
```

Open http://127.0.0.1:8000/docs in your browser and try out the API.

#### Preview
![png](preview.png)