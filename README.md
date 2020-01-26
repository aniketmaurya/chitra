# Chitra
> Library for creating data input pipeline in pure Tensorflow 2.x


.
<!-- This file will become your README and also the index of your documentation. -->

## Install

`pip install chitra`

## How to use

<!-- Fill me in please! Don't forget code examples: -->
## Loading data for image classification

```python
from chitra.dataloader import Clf

path = '/Users/aniketmaurya/Pictures/cats'

clf_dl = Clf()
data = clf_dl.from_folder(path)

print('class names:', clf_dl.CLASS_NAMES)

for e in data.take(1):
    print('image shape:', e[0].shape)
    print('class:', e[1].numpy())
```

    class names: (b'whitecat', b'blackcat')
    image shape: (1, 160, 160, 3)
    class: b'whitecat'

