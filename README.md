# Chitra
> Image processing library for creating data pipeline.


.
<!-- This file will become your README and also the index of your documentation. -->

## Install

`pip install chitra`

## How to use

<!-- Fill me in please! Don't forget code examples: -->
## Loading data for image classification

```python
from chitra.dataloader import Clf
```

```python
path = '/Users/aniketmaurya/Pictures/cats'

clf_dl = Clf()
data = clf_dl.from_folder(path)
```

```python
clf_dl.CLASS_NAMES
```




    (b'whitecat', b'blackcat')



```python
for e in data.take(1):
    print('image shape:', e[0].shape)
    print('class:', e[1].numpy())
```

    image shape: (1, 160, 160, 3)
    class: b'whitecat'

