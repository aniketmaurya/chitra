# Chitra
> Library for creating data input pipeline in Tensorflow


## Install

`pip install chitra`

## How to use

<!-- Fill me in please! Don't forget code examples: -->
## Loading data for image classification

```
import tensorflow as tf
import chitra
from chitra.dataloader import Clf, show_batch

path = '/Users/aniketmaurya/Pictures/cats'

clf_dl = Clf()
data = clf_dl.from_folder(path, target_shape=(224, 224))

clf_dl.show_batch(6, figsize=(6,6))
```

    CLASSES ENCODED: {'blackcat': 0, 'whitecat': 1}



![png](docs/images/output_4_1.png)

